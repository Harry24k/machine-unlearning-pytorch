"""SIU -- Single Image Unlearning (Li et al., NeurIPS 2024, arXiv 2405.12523).

Fine-tune on *Multifaceted Fine-tuning Data* (see ``torchunlearn.unlearn.recipes.siu.build_siu_samples``) with

    L = ce_weight * CE(all fine-tuning text)  +  dmk_weight * L_DMK

    L_DMK = sum_t  K_S(t) * sum_v K_V(v) * p_ref(v | ...) * log( p_ref(v | ...) / p_theta(v | ...) )

K_S (token-level mask): 0 on the positions of the *specified tokens* (the substitute name / new description that
    contradicts the original knowledge, given per sample in ``meta["specified_text"]``), 1 elsewhere.
K_V (vocabulary-level mask): 0 on the token ids of the concept's name(s) (``concept_names``), 1 elsewhere.
p_ref is the original (pre-unlearning) model, so ``ref_mode="model"`` is forced.  Both the Forget batch (targets
1-3 of the paper) and the Retain batch (target 4) go through the same CE + DMK.  Paper: alpha=0.9, beta=0.75,
LoRA on LLaVA, Adam lr 3e-4, batch 4, ~6 steps for one concept.
"""
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F

from .base import SeqUnlearner


def _find_subsequence(seq: List[int], sub: List[int]) -> List[int]:
    out = []
    n, m = len(seq), len(sub)
    if m == 0:
        return out
    for i in range(n - m + 1):
        if seq[i:i + m] == sub:
            out.extend(range(i, i + m))
    return out


class SIU(SeqUnlearner):
    """Dual Masked KL-divergence + CE on multifaceted fine-tuning data."""

    def __init__(self, rmodel, concept_names: Optional[Iterable[str]] = None, ce_weight=0.9, dmk_weight=0.75, **kw):
        kw["ref_mode"] = "model"
        kw.setdefault("retain_loss_type", "none")
        super().__init__(rmodel, **kw)
        self.ce_weight, self.dmk_weight = float(ce_weight), float(dmk_weight)
        names = [concept_names] if isinstance(concept_names, str) else list(concept_names or [])
        ids = set()
        for name in names:
            for variant in (name, " " + name):
                ids.update(rmodel.tokenizer(variant, add_special_tokens=False)["input_ids"])
            for w in name.split():
                for variant in (w, " " + w):
                    ids.update(rmodel.tokenizer(variant, add_special_tokens=False)["input_ids"])
        self.concept_ids = sorted(ids)
        if not self.concept_ids:
            print("[SIU] warning: no concept_names given -> vocabulary mask K_V is all ones")

    def _token_mask(self, batch, labels):
        """K_S over label positions: 0 where the label token belongs to the sample's specified text."""
        ks = torch.ones_like(labels, dtype=torch.float)
        samples = batch.get("samples")
        if samples is None:
            return ks
        for i, s in enumerate(samples):
            spec = s.meta.get("specified_text") if s.meta else None
            if not spec:
                continue
            specs = [spec] if isinstance(spec, str) else list(spec)
            row = labels[i].tolist()
            for sp in specs:
                for variant in (sp, " " + sp):
                    sub = self.rmodel.tokenizer(variant, add_special_tokens=False)["input_ids"]
                    for pos in _find_subsequence(row, sub):
                        ks[i, pos] = 0.0
        return ks

    def _ce_dmk(self, batch):
        labels = self._labels(batch, "Forget")
        logits = self._logits(batch)
        nll, mask = self._token_nll(logits, labels)
        ce = nll.sum() / mask.sum().clamp(min=1)
        with torch.no_grad():
            ref_logits = self._logits(batch, model=self._ref())[:, :-1].float().to(logits.device)
            p_ref = F.softmax(ref_logits, -1)
        logq = F.log_softmax(logits[:, :-1].float(), -1)
        logp = torch.log(p_ref.clamp(min=1e-12))
        kv = torch.ones(logq.shape[-1], device=logq.device)
        if self.concept_ids:
            kv[torch.tensor(self.concept_ids, device=logq.device)] = 0.0
        ks = self._token_mask(batch, labels)[:, 1:].to(logq.device) * mask.float()
        kl_tok = (p_ref * (logp - logq) * kv).sum(-1)          # [B, L-1]
        dmk = (kl_tok * ks).sum() / ks.sum().clamp(min=1)
        return ce, dmk

    def calculate_cost(self, train_data, reduction="mean"):
        ce_f, dmk_f = self._ce_dmk(train_data["Forget"])
        cost = self.ce_weight * ce_f + self.dmk_weight * dmk_f
        rb = train_data.get("Retain")
        if rb is not None and self.alpha != 0:
            ce_r, dmk_r = self._ce_dmk(rb)
            cost = cost + self.alpha * (self.ce_weight * ce_r + self.dmk_weight * dmk_r)
            self.add_record_item("RTLoss", float((ce_r + dmk_r).detach()))
        self.add_record_item("FGLoss", float(ce_f.detach()))
        self.add_record_item("DMK", float(dmk_f.detach()))
        self.add_record_item("Cost", float(cost.detach()))
        if self.eval_every and self.stop_lp is not None and self.accumulated_iter % self.eval_every == 0:
            self._check_stop()
        return cost

    def forget_loss(self, batch):
        ce, dmk = self._ce_dmk(batch)
        return self.ce_weight * ce + self.dmk_weight * dmk
