"""MLLMU-Bench (Liu et al., 2025; used by ASRU ICML 2026, MMUnlearner, ...) and UMU-Bench (Wang et al., NeurIPS 2025).

MLLMU-Bench HF ``MLLMMU/MLLMU-Bench`` configs: forget_5/10/15, retain_95/90/85, Test_Set, Retain_Set (real), ft_Data,
Full_Set.  Row: image, ID, biography, question, answer, Classification_Task{Image_Textual_Questions[], Pure_Text_Questions[]}
(each {Question, Options{A..D}, Correct_Answer}), Generation_Task[{Question, Ground_Truth, Type}], Mask_Task[...].
Test_Set rows carry ``images`` (a list) instead of ``image``.

UMU-Bench HF ``chengyewang/UMU-bench`` has the same split names with ID, image, Biography, MM_QA, UM_QA, Classify, Cloze,
Generation stored as JSON strings; the same parser handles both (keys are matched case-insensitively).

    ft   = finetune_samples(load_split("ft_Data"))          # stage I: (image, "Tell me about..." , biography answer)
    ev   = eval_tasks(load_split("forget_5"))               # {"cls": [...], "gen": [...], "cloze": [...]}
    evaluator = MLLMUBenchEvaluator({"forget": ev, "retain": eval_tasks(load_split("retain_95")), ...}, collator)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from ..seq_data import SeqSample

LETTERS = "ABCDEFGH"


def load_split(config: str, name: str = "MLLMMU/MLLMU-Bench", split: str = "train", **kw) -> List[dict]:
    from datasets import load_dataset
    return [dict(r) for r in load_dataset(name, config, split=split, **kw)]


def _j(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:  # noqa: BLE001
            return v
    return v


def _get(d: dict, *keys, default=None):
    low = {k.lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in low:
            return low[k.lower()]
    return default


def _images(r: dict) -> list:
    if r.get("image") is not None:
        return [r["image"]]
    ims = r.get("images")
    return list(ims) if ims else []


def finetune_samples(rows: Sequence[dict]) -> List[SeqSample]:
    """Stage-I data: the biography QA (MLLMU ``question``/``answer``) + every generation-task QA."""
    out = []
    for r in rows:
        ims = _images(r)
        gid = str(r.get("ID"))
        q, a = _get(r, "question"), _get(r, "answer")
        if q and a:
            out.append(SeqSample(prompt=q, answer=a, images=ims[:1], group=gid, id=f"{gid}#bio"))
        for k, g in enumerate(_j(_get(r, "Generation_Task", "Generation", default=[])) or []):
            gq, ga = _get(g, "Question"), _get(g, "Ground_Truth", "Answer")
            if gq and ga:
                mm = str(_get(g, "Type", default="Image_Textual")).lower().startswith("image")
                out.append(SeqSample(prompt=gq, answer=ga, images=ims[:1] if mm else [], group=gid, id=f"{gid}#gen{k}"))
    return out


def eval_tasks(rows: Sequence[dict]) -> Dict[str, List[SeqSample]]:
    cls, gen, cloze = [], [], []
    for r in rows:
        ims = _images(r)
        gid = str(r.get("ID"))
        ct = _j(_get(r, "Classification_Task", "Classify", default={})) or {}
        if isinstance(ct, dict):
            groups = [(True, ct.get("Image_Textual_Questions", [])), (False, ct.get("Pure_Text_Questions", []))]
        else:
            groups = [(True, ct)]
        for mm, qs in groups:
            for k, q in enumerate(qs or []):
                opts = _get(q, "Options", default={})
                if isinstance(opts, dict):
                    keys = sorted(opts)
                    choices = [opts[x] for x in keys]
                    ans = str(_get(q, "Correct_Answer", "Answer", default="A")).strip()
                    label = keys.index(ans) if ans in keys else choices.index(ans) if ans in choices else 0
                else:
                    choices = list(opts); ans = str(_get(q, "Correct_Answer", "Answer", default=""))
                    label = LETTERS.index(ans) if ans in LETTERS[: len(choices)] else (choices.index(ans) if ans in choices else 0)
                cls.append(SeqSample(prompt=_get(q, "Question"), answer=choices[label], images=ims[:1] if mm else [], group=gid,
                                     id=f"{gid}#cls{'mm' if mm else 'txt'}{k}", meta={"choices": choices, "label": label}))
        for k, g in enumerate(_j(_get(r, "Generation_Task", "Generation", default=[])) or []):
            mm = str(_get(g, "Type", default="Image_Textual")).lower().startswith("image")
            gen.append(SeqSample(prompt=_get(g, "Question"), answer=_get(g, "Ground_Truth", "Answer"), images=ims[:1] if mm else [], group=gid,
                                 id=f"{gid}#gen{k}", meta={"type": _get(g, "Type")}))
        for k, g in enumerate(_j(_get(r, "Mask_Task", "Cloze", default=[])) or []):
            mm = str(_get(g, "Type", default="Image_Textual")).lower().startswith("image")
            cloze.append(SeqSample(prompt=_get(g, "Question"), answer=_get(g, "Ground_Truth", "Answer"), images=ims[:1] if mm else [], group=gid,
                                   id=f"{gid}#cloze{k}"))
    return {"cls": cls, "gen": gen, "cloze": cloze}


def umu_qa_samples(rows: Sequence[dict], field: str = "MM_QA") -> List[SeqSample]:
    """UMU-Bench MM_QA / UM_QA JSON strings -> QA samples (multimodal for MM_QA, text-only for UM_QA)."""
    out = []
    for r in rows:
        ims = _images(r)
        gid = str(r.get("ID"))
        qa = _j(r.get(field)) or []
        if isinstance(qa, dict):
            qa = [{"Question": q, "Answer": a} for q, a in qa.items()]
        for k, q in enumerate(qa):
            out.append(SeqSample(prompt=_get(q, "Question", "question"), answer=_get(q, "Answer", "answer"),
                                 images=ims[:1] if field.upper().startswith("MM") else [], group=gid, id=f"{gid}#{field}{k}"))
    return out
