"""FIUBench (Ma et al., ICLR 2025) -- fictitious facial identities with private attributes.

HF ``gray311/FIUBench`` rows: image_path | image, name, gender, caption, qa_list[{question, paraphrased_question,
answer, paraphrased_answer, perturbed_answer, keywords}], raw_data.  One identity = one group; the paper's default
split forgets 5 % of the identities (also 1 % / 10 %).

    rows = load_rows()                                  # HF (or a local JSON/parquet via path=)
    train = to_samples(rows, image_root=...)            # stage I fine-tuning data (all QA pairs)
    forget, retain = split(train, ratio=0.05, seed=0)   # by identity
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..seq_data import SeqSample, split_forget_retain


def load_rows(path: Optional[str] = None, split: str = "test", **kw) -> List[dict]:
    if path and (path.endswith(".json") or path.endswith(".jsonl")):
        with open(path) as fh:
            return json.load(fh) if path.endswith(".json") else [json.loads(l) for l in fh if l.strip()]
    from datasets import load_dataset
    ds = load_dataset(path or "gray311/FIUBench", split=split, **kw) if not (path and os.path.isdir(path)) else load_dataset("parquet", data_dir=path, split="train")
    return [dict(r) for r in ds]


def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        try:
            v = json.loads(x)
            if isinstance(v, list):
                return [str(t) for t in v]
        except Exception:  # noqa: BLE001
            pass
        return [x]
    return [str(t) for t in x]


def to_samples(rows: Sequence[dict], image_root: Optional[str] = None) -> List[SeqSample]:
    out = []
    for r in rows:
        img = r.get("image") if r.get("image") is not None else r.get("image_path")
        if isinstance(img, str) and image_root and not os.path.isabs(img):
            img = os.path.join(image_root, img)
        gid = str(r.get("name") or r.get("id") or r.get("image_path"))
        qa_list = r.get("qa_list") or []
        if isinstance(qa_list, str):
            qa_list = json.loads(qa_list)
        for k, qa in enumerate(qa_list):
            out.append(SeqSample(prompt=qa["question"], answer=qa.get("answer"), images=[img] if img is not None else [], group=gid,
                                 id=f"{gid}#{k}",
                                 meta={"paraphrased_questions": _as_list(qa.get("paraphrased_question")),
                                       "paraphrased_answer": (_as_list(qa.get("paraphrased_answer")) or [None])[0],
                                       "perturbed_answers": _as_list(qa.get("perturbed_answer")),
                                       "keywords": _as_list(qa.get("keywords")), "gender": r.get("gender")}))
    return out


def split(samples: Sequence[SeqSample], ratio: float = 0.05, seed: int = 0) -> Tuple[List[SeqSample], List[SeqSample]]:
    return split_forget_retain(samples, by="group", ratio=ratio, seed=seed)
