"""MLUBench (ICML 2026, arXiv 2606.12809; repo lihe-maxsize/Lifelong_Unlearning_main): 127 entities / 9 classes,
5,105 images, 15,414 VQA pairs, four sequential unlearning tasks A-D, each with a forget set and a retain set.

The release ships ``Tasks/`` (QA per task) and ``Entities/`` (QA per entity) as JSON plus an image archive.  The exact
JSON keys are not documented on the card, so this loader is tolerant: every record needs a question, an answer and an
image path (keys: question|Question|prompt, answer|Answer|Ground_Truth|response, image|image_path|images, entity|
Entity|name, class|category|type).

    tasks = load_tasks(root)                     # {"A": (forget, retain), "B": ..., ...}
    lumoe.add_request("A", *tasks["A"])
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from ..seq_data import SeqSample
from .mllmu_bench import _get


def records_to_samples(recs: Sequence[dict], image_root: Optional[str] = None, prefix: str = "") -> List[SeqSample]:
    out = []
    for i, r in enumerate(recs):
        q = _get(r, "question", "prompt", "instruction")
        a = _get(r, "answer", "Ground_Truth", "response", "output")
        img = _get(r, "image", "image_path", "images")
        if isinstance(img, list):
            img = img[0] if img else None
        if isinstance(img, str) and image_root and not os.path.isabs(img):
            img = os.path.join(image_root, img)
        ent = _get(r, "entity", "name", "Entity")
        if q is None or a is None:
            continue
        out.append(SeqSample(prompt=str(q), answer=str(a), images=[img] if img else [], group=str(ent) if ent else None,
                             id=f"{prefix}{_get(r, 'id', default=i)}",
                             meta={"entity": ent, "class": _get(r, "class", "category", "type")}))
    return out


def _load_json(p: str):
    with open(p) as fh:
        d = json.load(fh)
    if isinstance(d, dict):
        for k in ("data", "records", "qa", "QA"):
            if k in d and isinstance(d[k], list):
                return d[k]
        return [dict(v, id=k) if isinstance(v, dict) else v for k, v in d.items()]
    return d


def load_tasks(root: str, image_root: Optional[str] = None, tasks: Sequence[str] = ("A", "B", "C", "D")
               ) -> Dict[str, Tuple[List[SeqSample], List[SeqSample]]]:
    """root/Tasks/<task>/forget*.json and retain*.json (case-insensitive, nested dirs allowed)."""
    out = {}
    for t in tasks:
        cand = glob.glob(os.path.join(glob.escape(root), "**", f"*{t}*"), recursive=True)
        dirs = [c for c in cand if os.path.isdir(c)]
        f_files = [p for d in dirs for p in glob.glob(os.path.join(glob.escape(d), "**", "*forget*.json"), recursive=True)]
        r_files = [p for d in dirs for p in glob.glob(os.path.join(glob.escape(d), "**", "*retain*.json"), recursive=True)]
        if not f_files:
            raise FileNotFoundError(f"MLUBench task {t}: no *forget*.json under {root}")
        forget = [s for p in f_files for s in records_to_samples(_load_json(p), image_root, prefix=f"{t}F:")]
        retain = [s for p in r_files for s in records_to_samples(_load_json(p), image_root, prefix=f"{t}R:")]
        out[t] = (forget, retain)
    return out
