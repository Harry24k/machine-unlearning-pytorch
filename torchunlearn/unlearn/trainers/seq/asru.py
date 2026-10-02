"""ASRU -- Activation Steering meets Reinforcement Unlearning (ICML 2026, arXiv 2605.15687; code guangjh/ASRU).

Stage 1  ``ASRUSteer`` (non-trainer, closed form).  At decoder layer L*, the *knowledge-absence direction* is
         v = mean_{x in D_target} h(x) - mean_{x in D_f} h(x), where D_target are prompts with unseen images
         (the paper uses 400 DigiFace-1M faces).  Only the MLP down-projection W of layer L* is updated so that
         forget activations move along v while retain activations stay:
             t_f = a(x) + lambda v,   t_r = a(x)
             W* = ([H_f;H_r]^T [H_f;H_r] + gamma I)^-1 [H_f;H_r]^T [T_f;T_r]
         (H = inputs of the down-projection on answer positions, a = its outputs).  Repo defaults: layer 17,
         forget scale 3.

Stage 2  ``ASRU`` (GRPO).  Rule-based reward (paper Sec. 3.3):
         forget prompt:   exact ground truth -> 0.0 | valid refusal -> 1.0 | 0 < ROUGE-L <= 0.4 -> 0.5 | else 0.1
         boundary prompt: exact ground truth -> 1.0 | refusal -> 0.0 | ROUGE-L > 0.6 -> 0.5 | else 0.1
         Boundary set = retain samples similar to the forget set (paper: 27.3 % of the retain set); pass them as
         the Retain loader.  Paper: lr 1e-6, KL coef 0.1, 90-120 steps, batch 2.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

import torch

from ....metrics.text import REFUSAL_PATTERNS, exact_match, is_refusal, normalize, rouge_l
from .grpo import GRPOUnlearner

class ASRU(GRPOUnlearner):
    """Stage 2 of ASRU: GRPO with the paper's rule-based refusal / boundary reward."""

    def reward(self, sample, text: str, role: str) -> float:
        gold = sample.answer or ""
        em = exact_match(text, gold) > 0 or (normalize(gold) and normalize(gold) in normalize(text))
        rl = rouge_l(text, gold)
        if role == "Forget":
            if em:
                return 0.0
            if is_refusal(text):
                return 1.0
            if 0.0 < rl <= 0.4:
                return 0.5
            return 0.1
        if em:
            return 1.0
        if is_refusal(text):
            return 0.0
        if rl > 0.6:
            return 0.5
        return 0.1


class ASRUSteer:
    """Stage 1 of ASRU: closed-form activation steering of one MLP down-projection.

    Arguments
        rmodel        SeqRobModel
        layer_id      decoder block index L* (paper/repo: 17 for Qwen3-VL-8B)
        lam           steering strength lambda along v (repo "forget scale" 3)
        gamma         ridge term
        positions     "answer" (label positions) | "all" (every real text token)
    fit(train_loaders, target_samples, collator): train_loaders = MergedLoaders(Forget, Retain);
        target_samples = prompts with *unseen* images (knowledge-absence anchor).
    """

    def __init__(self, rmodel, layer_id=17, lam=3.0, gamma=1e-2, positions="answer", device=None):
        self.rmodel, self.layer_id, self.lam, self.gamma, self.positions = rmodel, int(layer_id), float(lam), float(gamma), positions
        prefix, layers = rmodel.decoder_layers()
        self.layer = layers[self.layer_id]
        names = rmodel.mlp_out_proj_names([self.layer_id])
        mods = dict(rmodel.model.named_modules())
        self.down_name = names[0][: -len(".weight")]
        self.down = mods[self.down_name]
        self.device = device or next(rmodel.model.parameters()).device

    def setup(self, **kw):
        return self

    def record_rob(self, *a, **k):
        return self

    @torch.no_grad()
    def _collect(self, loader_or_samples, collator, want_hidden=False):
        """Returns (H: inputs of down_proj at selected positions, A: outputs, Hid: residual output of the block)."""
        from torch.utils.data import DataLoader
        from ...seq_data import SeqDataset
        Hs, As, Hid = [], [], []
        store = {}
        h1 = self.down.register_forward_hook(lambda m, i, o: store.update(inp=i[0].detach(), out=o.detach()))
        h2 = self.layer.register_forward_hook(lambda m, i, o: store.update(hid=(o[0] if isinstance(o, tuple) else o).detach()))
        try:
            if isinstance(loader_or_samples, DataLoader):
                it = loader_or_samples
            else:
                it = DataLoader(SeqDataset(list(loader_or_samples)), batch_size=4, shuffle=False, collate_fn=collator)
            for batch in it:
                inputs = {k: v.to(self.device) for k, v in batch.items()
                          if torch.is_tensor(v) and not k.startswith("alt_") and k not in ("idx", "alt_k", "labels", "answer_labels", "full_labels")}
                self.rmodel.model(**inputs, use_cache=False)
                if self.positions == "answer" and (batch["answer_labels"] != -100).any():
                    m = batch["answer_labels"] != -100
                else:
                    m = batch["full_labels"] != -100
                m = m.to(self.device)
                Hs.append(store["inp"][m].float().cpu()); As.append(store["out"][m].float().cpu()); Hid.append(store["hid"][m].float().cpu())
        finally:
            h1.remove(); h2.remove()
        return torch.cat(Hs), torch.cat(As), torch.cat(Hid)

    @torch.no_grad()
    def fit(self, train_loaders, target_samples: Sequence, collator, **kw):
        loaders = getattr(train_loaders, "loaders", train_loaders)
        self.rmodel.model.eval()
        Hf, Af, hid_f = self._collect(loaders["Forget"], collator)
        Ht, At, hid_t = self._collect(target_samples, collator)
        v = hid_t.mean(0) - hid_f.mean(0)                              # knowledge-absence direction at layer L*
        self.direction = v
        Tf = Af + self.lam * v
        if "Retain" in loaders:
            Hr, Ar, _ = self._collect(loaders["Retain"], collator)
            H, T = torch.cat([Hf, Hr]), torch.cat([Tf, Ar])
        else:
            H, T = Hf, Tf
        H, T = H.double(), T.double()
        d_in = H.shape[1]
        W = torch.linalg.solve(H.T @ H + self.gamma * torch.eye(d_in, dtype=H.dtype), H.T @ T)   # [d_in, d_out]
        w = self.down.weight                                           # nn.Linear: [d_out, d_in]
        w.copy_(W.T.to(w.dtype).to(w.device))
        self.rmodel.model.train()
        self.results = {"layer": self.layer_id, "down_proj": self.down_name, "n_forget_tok": int(Hf.shape[0]),
                        "n_target_tok": int(Ht.shape[0]), "direction_norm": float(v.norm())}
        return self
