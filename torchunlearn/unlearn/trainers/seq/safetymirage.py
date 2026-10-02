"""Safety Mirage (Chen et al., ICLR 2026, arXiv 2503.11832): VLM safety alignment *by unlearning* instead of
supervised safety fine-tuning, which removes the spurious "first-word -> refusal" correlation.

Data (VLGuard train split):
    forget  D_u = unsafe (image, query) pairs.  RMU pairs each unsafe query with a harmful response (the paper uses
                  Llama-2-13B-Chat generations); NPO uses the unsafe query alone.
    retain  D_r = safe (image, query, answer) pairs.
Objective:  l_u(theta, D_u) + l_ft(theta, D_r) + alpha * l_mu,r(theta, D_r)
    NPO:  l_u = E[-2/beta log sigma(-beta log pi_theta(x)/pi_ref(x))]        (beta: NPO temperature)
    RMU:  l_u = E[|| M_theta(x) - c v ||^2 ],  l_mu,r = || M_theta(x_r) - M_ref(x_r) ||^2  (layer + c tuned for VLMs)
Evaluation (metrics.mm_bench.safety_mirage_metrics): attack success rate before/after the one-word attack, rejection
rate on safe queries (over-prudence), VQA utility.

The two classes below are the NPO / RMU trainers with the paper's data convention baked in: the forget sequence is
the *whole* unsafe text (query or query+harmful response, ``retain_loss_on`` governs the retain side only), and the
retain term is NLL fine-tuning on safe answers plus the method's own retain regulariser.
``build_vlguard_sets`` turns the VLGuard JSON into the two sample lists.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from .npo import NPO
from .rmu import RMU


class SafetyMirageNPO(NPO):
    """NPO on unsafe multimodal queries (label = the unsafe query text itself, or query + harmful response)."""

    def __init__(self, rmodel, beta=0.1, **kw):
        kw.setdefault("retain_loss_type", "NLL")
        kw.setdefault("alpha", 1.0)
        super().__init__(rmodel, beta=beta, **kw)

    def _labels(self, batch, key):
        if key == "Forget":
            return batch["full_labels"]          # the whole unsafe text is the thing to unlearn
        return super()._labels(batch, key)


class SafetyMirageRMU(RMU):
    """RMU on unsafe query + harmful response, EMBED_DIFF retain on safe data."""

    def __init__(self, rmodel, layer_id=7, steering_coeff=20.0, **kw):
        super().__init__(rmodel, layer_id=layer_id, steering_coeff=steering_coeff, **kw)

    def _labels(self, batch, key):
        if key == "Forget":
            return batch["full_labels"]
        return super()._labels(batch, key)


def build_vlguard_sets(train_json: str, image_root: Optional[str] = None, harmful_responses: Optional[Dict[str, str]] = None,
                       unsafe_answer: Optional[str] = None) -> Tuple[List, List]:
    """VLGuard ``train.json`` -> (forget, retain) SeqSample lists.

    VLGuard record: {"id", "image", "safe": bool, "instr-resp": [{"instruction", "response"} | {"safe_instruction",
    "response"} | {"unsafe_instruction", "response"}]}.  Unsafe images: every instruction is unsafe; safe images
    carry one safe and one unsafe instruction.  harmful_responses maps "<id>#<k>" -> harmful text (RMU); when absent,
    ``unsafe_answer`` (or the query itself) is used as the forget answer.
    """
    from ...seq_data import SeqSample
    data = json.load(open(train_json))
    forget, retain = [], []
    for r in data:
        img = r.get("image")
        imgs = [os.path.join(image_root, img) if (image_root and img and not os.path.isabs(img)) else img] if img else []
        for k, ir in enumerate(r.get("instr-resp", [])):
            sid = f"{r.get('id')}#{k}"
            if "safe_instruction" in ir:
                retain.append(SeqSample(prompt=ir["safe_instruction"], answer=ir["response"], images=imgs, group=str(r.get("id")), id=sid,
                                        meta={"safe": True}))
            else:
                q = ir.get("unsafe_instruction") or ir.get("instruction")
                if r.get("safe", False) and "unsafe_instruction" not in ir:
                    retain.append(SeqSample(prompt=q, answer=ir["response"], images=imgs, group=str(r.get("id")), id=sid, meta={"safe": True}))
                    continue
                ans = (harmful_responses or {}).get(sid) or unsafe_answer or q
                forget.append(SeqSample(prompt=q, answer=ans, images=imgs, group=str(r.get("id")), id=sid,
                                        meta={"safe": False, "safe_response": ir.get("response")}))
    return forget, retain
