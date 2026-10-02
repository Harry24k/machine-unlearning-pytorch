# -*- coding: utf-8 -*-
r"""REVS: Rank Editing in the Vocabulary Space (Ashuach et al., ACL 2025) — PII-specific, non-gradient.

Port of ``REVS/revs/revs.py`` + ``utils/hidden_state_ops.py`` + ``utils/data.py`` (token selection)
for Llama-structured HF causal LMs (``model.model.layers[i].mlp.down_proj``, ``model.model.norm``,
``model.lm_head``; e.g. Llama-3.1, Qwen2, Mistral).  OLMo-2 (post-norm residual) and Qwen3.5
(hybrid attention) violate ``residual_after = residual_before + attn_out + mlp_out`` and are rejected.

Algorithm (per target token, for every layer):
  1. run the prompt (document prefix up to the mention, plus already-processed target tokens), collect at the
     last position: residual_before, attn_out, mlp_out (fc_out), down_proj input (fc_out_act), residual_after
  2. rank of the target token in lm_head(norm(·)) of the residual and of the MLP output;
     skip the layer if residual rank >= residual_bottom_rank_margin or mlp rank >= mlp_top_rank_margin
  3. select neurons = down_proj columns: keep the ``act_filter`` (top_100) most active, score them by the
     target-token rank of lm_head(norm(column)), take the ``n_neurons`` lowest ranks
  4. for a growing/shrinking number of those neurons (start 5; x1.6 / x0.8 / +-1 adjustments) compute
     column deltas that push the target-token rank of each column into
     [neuron_bottom_rank_margin, neuron_top_rank_margin] by editing its logit (-10 init, x1.3 / x0.8)
     and inverting lm_head (pinv) + RMSNorm; stop when the edited MLP-output rank and residual rank
     fall inside their margins
  5. apply ``down_proj.weight[:, idx] += delta``

Target tokens: ``token_method='rarest'`` sorts the mention's tokens by token id (descending) and keeps
``max_tokens`` of them (reference behaviour); each kept token is edited with the prefix that precedes it.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from ...nn.robmodel import RobModel


class REVS:
    """Non-trainer.  ``REVS(rmodel, **hparams).fit(forget_items, docs)`` edits ``rmodel.model`` in place
    (``copy_model=True`` edits a deep copy left in ``self.rmodel``)."""

    def __init__(self, rmodel, n_neurons=30, max_tokens=2, token_method="rarest",
                 residual_bottom_rank_margin=10000, residual_top_rank_margin=20000,
                 mlp_bottom_rank_margin=10000, mlp_top_rank_margin=10000,
                 neuron_bottom_rank_margin=90000, neuron_top_rank_margin=100000,
                 max_iter_mlp_rank=100, max_iter_neuron_rank=100, act_filter="top_100",
                 neurons_score_method="rank", skip_tokens=None, stop_tokens=None,
                 max_prompt_tokens=1024, layers=None, copy_model=False, device=None, seed=0):
        assert isinstance(rmodel, RobModel)
        self.rmodel = rmodel
        self.tok = rmodel.tokenizer
        self.device = device or next(rmodel.parameters()).device
        self.hp = dict(n_neurons=n_neurons, max_tokens=max_tokens, token_method=token_method,
                       residual_bottom_rank_margin=residual_bottom_rank_margin,
                       residual_top_rank_margin=residual_top_rank_margin,
                       mlp_bottom_rank_margin=mlp_bottom_rank_margin, mlp_top_rank_margin=mlp_top_rank_margin,
                       neuron_bottom_rank_margin=neuron_bottom_rank_margin,
                       neuron_top_rank_margin=neuron_top_rank_margin,
                       max_iter_mlp_rank=max_iter_mlp_rank, max_iter_neuron_rank=max_iter_neuron_rank,
                       act_filter=act_filter, neurons_score_method=neurons_score_method,
                       skip_tokens=skip_tokens or [], stop_tokens=stop_tokens or [],
                       max_prompt_tokens=max_prompt_tokens, layers=layers, copy_model=copy_model, seed=seed)
        self.evaluator = None
        self.results: Dict[str, dict] = {}
        self.edit_dicts = []          # [(layer -> edit_dict, target_token)]
        self._pinv = None
        self._check_structure()

    # ------------------------------------------------------------------ API parity
    def setup(self, **kw):
        self.hp.update(kw)
        return self

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator
        return self

    def record_rob(self, *a, **k):
        return self

    # ------------------------------------------------------------------ structure
    @property
    def model(self):
        return self.rmodel.model

    def _check_structure(self):
        m = self.model
        mt = getattr(m.config, "model_type", "")
        if mt in ("olmo2", "qwen3_5", "qwen3_next"):
            raise NotImplementedError(f"REVS needs pre-norm Llama-style residual structure; model_type={mt}")
        for attr in ("model.layers", "model.norm", "lm_head"):
            o = m
            for a in attr.split("."):
                o = getattr(o, a, None)
                if o is None:
                    raise NotImplementedError(f"REVS: missing {attr}")
        if not hasattr(m.model.layers[0].mlp, "down_proj"):
            raise NotImplementedError("REVS: mlp.down_proj not found")
        self.n_layers = len(m.model.layers)

    @torch.no_grad()
    def _pinv_lm_head(self):
        if self._pinv is None:
            self._pinv = torch.linalg.pinv(self.model.lm_head.weight.detach().float())  # (d, V)
        return self._pinv

    @torch.no_grad()
    def hs_to_logits(self, hs):
        return self.model.lm_head(self.model.model.norm(hs.to(self.model.lm_head.weight.dtype))).float()

    @torch.no_grad()
    def logits_to_hs(self, logits, mean, var):
        normed = logits @ self._pinv_lm_head().T                  # invert lm_head (no bias)
        norm = self.model.model.norm
        eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
        rms = torch.sqrt(var + eps)
        return (normed * rms.unsqueeze(1)) / norm.weight.float() + mean.unsqueeze(1)   # invert_llama_layer_norm

    @staticmethod
    def rank_of(logits, tid):
        if logits.dim() == 2:
            return (logits > logits[:, tid].unsqueeze(1)).sum(1)
        return (logits > logits[tid]).sum()

    # ------------------------------------------------------------------ activations
    @torch.no_grad()
    def collect(self, ids):
        """last-position activations per layer: residual_in/out, attn_out, mlp_out, mlp_in(down_proj input)."""
        acts = defaultdict(dict)
        hds = []
        layers = self.model.model.layers

        def mk(l, name, which):
            def hook(mod, inp, out):
                t = (inp[0] if which == "in" else (out[0] if isinstance(out, tuple) else out))
                acts[l][name] = t[0, -1].detach().float()
            return hook
        for l, layer in enumerate(layers):
            hds.append(layer.register_forward_hook(mk(l, "residual_in", "in")))
            hds.append(layer.register_forward_hook(mk(l, "residual_out", "out")))
            hds.append(layer.self_attn.register_forward_hook(mk(l, "attn_out", "out")))
            hds.append(layer.mlp.register_forward_hook(mk(l, "mlp_out", "out")))
            hds.append(layer.mlp.down_proj.register_forward_hook(mk(l, "mlp_in", "in")))
        try:
            self.model(input_ids=ids, attention_mask=torch.ones_like(ids))
        finally:
            for h in hds:
                h.remove()
        return acts

    # ------------------------------------------------------------------ neuron edit (calc_deltas_delete)
    @torch.no_grad()
    def calc_deltas_delete(self, neuron_values, tid, bottom, top, max_iter):
        logits = self.hs_to_logits(neuron_values)
        ranks = self.rank_of(logits, tid)
        to_edit = ranks < bottom
        deltas = torch.zeros(neuron_values.shape[0], device=neuron_values.device)
        deltas[to_edit] = -10
        edited = neuron_values.clone()
        for _ in range(max_iter):
            logits[:, tid] += deltas
            edited = self.logits_to_hs(logits, neuron_values.mean(-1), neuron_values.var(-1))
            logits[:, tid] -= deltas
            ranks = self.rank_of(self.hs_to_logits(edited), tid)
            below, above = ranks < bottom, ranks > top
            if below.any():
                deltas[below] *= 1.3
            if above.any():
                deltas[above] *= 0.8
            if not (below | above).any():
                break
        return edited - neuron_values

    @torch.no_grad()
    def select_neurons(self, W, act, tid):
        """W: down_proj.weight (d, n); act: (n,) down_proj input at the last position."""
        f = self.hp["act_filter"]
        if f == "positive":
            idx = torch.nonzero(act > 0, as_tuple=False).squeeze(-1)
        elif f.startswith("top_"):
            idx = torch.topk(act, int(f.split("_")[-1])).indices
        else:
            idx = torch.arange(act.shape[0], device=act.device)
        if self.hp["neurons_score_method"] == "rank":
            scores = self.rank_of(self.hs_to_logits(W[:, idx].T.float()), tid)
            largest = False
        elif self.hp["neurons_score_method"] == "act":
            scores, largest = act[idx], True
        else:
            raise ValueError("neurons_score_method must be 'rank' or 'act'")
        k = min(self.hp["n_neurons"], idx.shape[0])
        top = torch.topk(scores.float(), k, largest=largest).indices
        return idx[top]

    # ------------------------------------------------------------------ layer edit (edit_layer_rank_iter)
    @torch.no_grad()
    def edit_layer(self, acts, layer, tid):
        hp = self.hp
        a = acts[layer]
        residual_before, residual_after = a["residual_in"], a["residual_out"]
        fc_out, fc_out_act, attn_out = a["mlp_out"], a["mlp_in"], a["attn_out"]
        mlp_rank = self.rank_of(self.hs_to_logits(fc_out), tid).item()
        res_rank = self.rank_of(self.hs_to_logits(residual_after), tid).item()
        if res_rank >= hp["residual_bottom_rank_margin"] or mlp_rank >= hp["mlp_top_rank_margin"]:
            return None
        W = self.model.model.layers[layer].mlp.down_proj.weight.detach().clone().float()
        bias = getattr(self.model.model.layers[layer].mlp.down_proj, "bias", None)
        bias = torch.zeros(W.shape[0], device=W.device) if bias is None else bias.detach().float()
        sel = self.select_neurons(W, fc_out_act, tid)
        values = W[:, sel].T.clone()                     # (k, d)
        max_n, n_edit, tried = sel.shape[0], 5, set()
        deltas, cur_idx = None, None
        for _ in range(hp["max_iter_mlp_rank"]):
            n_edit = min(n_edit, max_n)
            cur_idx = sel[:n_edit]
            deltas = self.calc_deltas_delete(values[:n_edit], tid, hp["neuron_bottom_rank_margin"],
                                             hp["neuron_top_rank_margin"], hp["max_iter_neuron_rank"])
            W[:, cur_idx] += deltas.T
            mlp_new = fc_out_act @ W.T + bias
            res_new = residual_before + mlp_new + attn_out
            W[:, cur_idx] -= deltas.T
            mlp_rank = self.rank_of(self.hs_to_logits(mlp_new), tid).item()
            res_rank = self.rank_of(self.hs_to_logits(res_new), tid).item()
            mlp_ok = hp["mlp_bottom_rank_margin"] <= mlp_rank <= hp["mlp_top_rank_margin"]
            res_ok = hp["residual_bottom_rank_margin"] <= res_rank <= hp["residual_top_rank_margin"]
            if mlp_ok:
                if res_ok:
                    break
                elif res_rank < hp["residual_bottom_rank_margin"]:
                    n_edit = int(min(n_edit + 1, max_n))
                elif res_rank > hp["residual_top_rank_margin"]:
                    n_edit = int(max(n_edit - 1, 1))
            elif mlp_rank < hp["mlp_bottom_rank_margin"]:
                n_edit = int(min(n_edit * 1.6, max_n))
            elif mlp_rank > hp["mlp_top_rank_margin"]:
                n_edit = int(max(n_edit * 0.8, 2))
            if n_edit in tried:
                break
            tried.add(n_edit)
        return {"neurons_deltas": deltas.T, "neuron_indices": cur_idx, "mlp_rank": mlp_rank, "res_rank": res_rank}

    @torch.no_grad()
    def apply_edit(self, layer, ed, sign=1.0):
        w = self.model.model.layers[layer].mlp.down_proj.weight
        w.data[:, ed["neuron_indices"]] += (sign * ed["neurons_deltas"]).to(w.dtype)

    # ------------------------------------------------------------------ target tokens (create_concat_prompts_target)
    def concat_pairs(self, prompt_ids: List[int], target: str):
        hp = self.hp
        t_ids = self.tok(target, add_special_tokens=False)["input_ids"]
        pairs = []
        prefix = list(prompt_ids)
        for t in t_ids:
            s = self.tok.decode([t])
            if hp["stop_tokens"] and any(st in s for st in hp["stop_tokens"]):
                break
            if not (hp["skip_tokens"] and any(sk in s for sk in hp["skip_tokens"])):
                pairs.append((list(prefix), t))
            prefix.append(t)
        if hp["max_tokens"]:
            m = hp["token_method"]
            if m in ("rarest", None):
                pairs.sort(key=lambda x: x[1], reverse=True)
            elif m == "frequent":
                pairs.sort(key=lambda x: x[1])
            elif m == "random":
                import random
                random.Random(hp["seed"]).shuffle(pairs)
            pairs = pairs[:hp["max_tokens"]]
        return pairs

    # ------------------------------------------------------------------ fit
    def fit(self, forget_items, docs, evaluator=None, verbose=True, save_path=None):
        """forget_items: [SpanItem]; docs: {doc_id: text}."""
        if evaluator is not None:
            self.evaluator = evaluator
        if self.hp["copy_model"]:
            self.rmodel = copy.deepcopy(self.rmodel)
        self.model.eval()
        if self.evaluator is not None and "before" not in self.results:
            self.results["before"] = self.evaluator.evaluate(self.model, quick=False)
            print(self.evaluator.table(self.results["before"], "[REVS] before"))
        layers = self.hp["layers"] or list(range(self.n_layers))
        n_edits = 0
        it = tqdm(forget_items, desc="[REVS] targets", disable=not verbose)
        for item in it:
            prompt = docs[item.doc_id][:item.abs_start]
            p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"][-self.hp["max_prompt_tokens"]:]
            for prefix, tid in self.concat_pairs(p_ids, item.surface):
                ids = torch.tensor([prefix], device=self.device)
                layer_edits = {}
                for l in layers:
                    acts = self.collect(ids)          # re-collect after each layer edit (reference behaviour)
                    ed = self.edit_layer(acts, l, tid)
                    if ed is not None:
                        self.apply_edit(l, ed)
                        layer_edits[l] = ed
                        n_edits += 1
                self.edit_dicts.append((layer_edits, tid))
        self.results["n_edits"] = n_edits
        self.results["n_neurons_edited"] = self.total_edited_neurons()
        if self.evaluator is not None:
            self.results["after"] = self.evaluator.evaluate(self.model, quick=False)
            print(self.evaluator.table(self.results["after"], "[REVS] after"))
        if save_path:
            self.model.save_pretrained(save_path)
        return self

    def restore_all_edits(self):
        for layer_edits, _ in self.edit_dicts:
            for l, ed in layer_edits.items():
                self.apply_edit(l, ed, sign=-1.0)
        self.edit_dicts = []

    def total_edited_neurons(self):
        s = defaultdict(set)
        for layer_edits, _ in self.edit_dicts:
            for l, ed in layer_edits.items():
                s[l].update(ed["neuron_indices"].tolist())
        return sum(len(v) for v in s.values())
