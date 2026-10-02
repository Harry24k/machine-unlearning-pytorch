"""CLEAR (Dontsov et al., 2024) -- TOFU persons with synthetic faces; TOFU-style forget quality / model utility.

HF ``therem/CLEAR`` configs: full, full+tofu, forget01/05/10, retain99/95/90, forget*_perturbed, retain_perturbed,
real_faces, real_world.  Visual rows: image, caption (+ name); QA rows (tofu): question, answer, paraphrased_answer,
perturbed_answer.  Field names are matched tolerantly because the card does not fix them.

    forget = qa_samples(load_split("forget05"))           # image-caption + QA as SeqSamples
    pert   = qa_samples(load_split("forget05_perturbed")) # paraphrased / perturbed answers for truth ratios
    real   = choice_samples(load_split("real_faces"))     # multiple choice (meta choices/label)
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence

from ..seq_data import SeqSample
from .mllmu_bench import _get, _j

CAPTION_PROMPT = "Who is shown in this image? Describe the person."


def load_split(config: str, name: str = "therem/CLEAR", split: str = "train", **kw) -> List[dict]:
    from datasets import load_dataset
    return [dict(r) for r in load_dataset(name, config, split=split, **kw)]


def qa_samples(rows: Sequence[dict], caption_prompt: str = CAPTION_PROMPT) -> List[SeqSample]:
    out = []
    for i, r in enumerate(rows):
        img = _get(r, "image")
        name = _get(r, "name", "person", default=None)
        q = _get(r, "question", "prompt")
        a = _get(r, "answer", "caption")
        if q is None and _get(r, "caption"):
            q = caption_prompt
        if q is None or a is None:
            continue
        pa = _get(r, "paraphrased_answer")
        pe = _get(r, "perturbed_answer", "perturbed_answers")
        pe = _j(pe) if isinstance(pe, str) else pe
        if isinstance(pe, str):
            pe = [pe]
        out.append(SeqSample(prompt=str(q), answer=str(a), images=[img] if img is not None else [], group=str(name) if name else None,
                             id=str(_get(r, "id", default=i)),
                             meta={"paraphrased_answer": pa if isinstance(pa, str) else (pa[0] if pa else None),
                                   "perturbed_answers": [str(x) for x in (pe or [])]}))
    return out


def choice_samples(rows: Sequence[dict]) -> List[SeqSample]:
    """real_faces / real_world: question + options + answer -> meta choices/label."""
    out = []
    for i, r in enumerate(rows):
        img = _get(r, "image")
        q = _get(r, "question", default="Who is shown in this image?")
        opts = _j(_get(r, "options", "choices", "perturbed_answer", default=[])) or []
        ans = _get(r, "answer")
        if isinstance(opts, dict):
            opts = list(opts.values())
        choices = [str(o) for o in opts]
        if ans is not None and str(ans) not in choices:
            choices = [str(ans)] + choices
        label = choices.index(str(ans)) if ans is not None else 0
        out.append(SeqSample(prompt=str(q), answer=choices[label], images=[img] if img is not None else [], id=str(_get(r, "id", default=i)),
                             meta={"choices": choices, "label": label}))
    return out
