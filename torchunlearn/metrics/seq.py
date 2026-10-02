"""SeqUnlearningEvaluator -- forget / retain / test / probe evaluation for text and image+text models.

Roles are arbitrary names mapped to sample lists, e.g.

    SeqUnlearningEvaluator({"Forget": forget, "Retain": retain, "Test": test,
                            "Forget_noimg": strip_images(forget)},   # cross-modal leakage probe
                           collator, batch_size=8)

Per role:
    lp_mean      mean log-prob per answer token (quick metric; trainers use it for trajectories / early stop)
    nll_seq      mean summed NLL per sequence
    rouge_l / exact_match / includes   greedy generation vs the gold answer (quick=False only)
    gen          (prompt id, generation) pairs for inspection (first ``keep_gen`` items)
    mink         Min-k% token log-prob (k=20) -- membership-inference style score (lower = more "unseen")
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..unlearn.seq_data import SeqCollator, SeqDataset, SeqSample
from .text import GENERATION_METRICS


class SeqUnlearningEvaluator:
    def __init__(self, role_samples: Dict[str, Sequence[SeqSample]], collator: SeqCollator, batch_size: int = 8,
                 gen: bool = True, max_new_tokens: int = 32, metrics: Iterable[str] = ("rouge_l", "exact_match", "includes"),
                 n_limit: Optional[int] = None, keep_gen: int = 20, mink: float = 0.2, gen_kwargs: Optional[dict] = None):
        self.roles = {r: list(s)[: n_limit] if n_limit else list(s) for r, s in role_samples.items()}
        self.collator, self.batch_size, self.gen, self.max_new_tokens = collator, batch_size, gen, max_new_tokens
        self.metrics = [m for m in metrics if m in GENERATION_METRICS]
        self.keep_gen, self.mink, self.gen_kwargs = keep_gen, mink, dict(gen_kwargs or {})

    # ------------------------------------------------------------------ helpers
    def _loader(self, role):
        return DataLoader(SeqDataset(self.roles[role]), batch_size=self.batch_size, shuffle=False, collate_fn=self.collator)

    @torch.no_grad()
    def _logprob_stats(self, hf, role):
        dev = next(hf.parameters()).device
        tok_lp, seq_nll, mink = [], [], []
        for batch in self._loader(role):
            inputs = {k: v.to(dev) for k, v in batch.items()
                      if torch.is_tensor(v) and not k.startswith("alt_") and k not in ("idx", "alt_k", "labels", "answer_labels", "full_labels", "samples")}
            logits = hf(**inputs, use_cache=False).logits[:, :-1].float()
            lab = batch["answer_labels"][:, 1:].to(dev)
            mask = lab != -100
            lp = -F.cross_entropy(logits.transpose(1, 2), lab.clamp(min=0), reduction="none")
            for j in range(lp.shape[0]):
                v = lp[j][mask[j]]
                if v.numel() == 0:
                    continue
                tok_lp.append(v.mean().item())
                seq_nll.append(-v.sum().item())
                k = max(1, int(round(self.mink * v.numel())))
                mink.append(torch.topk(v, k, largest=False).values.mean().item())
        n = len(tok_lp)
        return {"n": n, "lp_mean": sum(tok_lp) / max(n, 1), "nll_seq": sum(seq_nll) / max(n, 1),
                "mink": sum(mink) / max(n, 1)}

    @torch.no_grad()
    def _generate(self, hf, role) -> List[str]:
        dev = next(hf.parameters()).device
        tok = self.collator.tok
        outs: List[str] = []
        samples = self.roles[role]
        for i in range(0, len(samples), self.batch_size):
            chunk = samples[i: i + self.batch_size]
            enc = self.collator.prompts(chunk)
            enc = {k: v.to(dev) for k, v in enc.items() if torch.is_tensor(v)}
            # trainers set config.use_cache=False for training; generation must re-enable the KV cache
            kw = {"max_new_tokens": self.max_new_tokens, "do_sample": False, "pad_token_id": tok.pad_token_id,
                  "use_cache": True}
            kw.update(self.gen_kwargs)
            gen = hf.generate(**enc, **kw)
            n_in = enc["input_ids"].shape[1]
            outs.extend(tok.batch_decode(gen[:, n_in:], skip_special_tokens=True))
        return outs

    # ------------------------------------------------------------------ public
    def evaluate(self, model, quick: bool = False, roles: Optional[Iterable[str]] = None) -> Dict[str, dict]:
        hf = model.model if hasattr(model, "modality") else model
        was_training = hf.training
        hf.eval()
        out: Dict[str, dict] = {}
        for role in (roles or self.roles):
            if role not in self.roles or not self.roles[role]:
                continue
            r = self._logprob_stats(hf, role)
            if not quick and self.gen:
                gens = self._generate(hf, role)
                golds = [s.answer or "" for s in self.roles[role]]
                for m in self.metrics:
                    fn = GENERATION_METRICS[m]
                    r[m] = sum(fn(p, g) for p, g in zip(gens, golds)) / max(len(gens), 1)
                r["gen"] = [(s.id, g) for s, g in list(zip(self.roles[role], gens))[: self.keep_gen]]
            out[role] = r
        if was_training:
            hf.train()
        return out

    def table(self, results: Dict[str, dict], title: str = "") -> str:
        cols = ["n", "lp_mean", "nll_seq", "mink"] + self.metrics
        lines = [title] if title else []
        lines.append("role".ljust(16) + "".join(c.rjust(12) for c in cols))
        for role, r in results.items():
            if not isinstance(r, dict):
                continue
            row = role.ljust(16)
            for c in cols:
                v = r.get(c)
                row += (f"{v:12.4f}" if isinstance(v, float) else str(v if v is not None else "-").rjust(12))
            lines.append(row)
        return "\n".join(lines)


def summarise_delta(before: Dict[str, dict], after: Dict[str, dict], keys=("lp_mean", "rouge_l", "exact_match", "includes")):
    out = {}
    for role in after:
        if role in before and isinstance(after[role], dict):
            out[role] = {k: after[role][k] - before[role][k] for k in keys if k in after[role] and k in before[role]}
    return out
