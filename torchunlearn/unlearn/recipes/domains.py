"""Domain-labelled image folders for ADU (NeurIPS 2025): Office-Home / DomainNet / mini-DomainNet / ImageNet+Sketch.

Layout  root/<domain>/<class>/*.jpg   ->  items (path, class_idx, domain_idx), class_names, domain_names.
``few_shot(items, k, seed)`` keeps k images per (domain, class) as in the paper (k = 8).
"""
from __future__ import annotations

import glob
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def from_domain_folders(root: str, domains: Optional[Sequence[str]] = None, classes: Optional[Sequence[str]] = None
                        ) -> Tuple[List[tuple], List[str], List[str]]:
    dom_dirs = sorted(d for d in glob.glob(os.path.join(glob.escape(root), "*")) if os.path.isdir(d))
    domain_names = [os.path.basename(d) for d in dom_dirs if (domains is None or os.path.basename(d) in domains)]
    if not domain_names:
        raise FileNotFoundError(f"no domain folders under {root}")
    cls_set = set()
    for dn in domain_names:
        for c in glob.glob(os.path.join(glob.escape(os.path.join(root, dn)), "*")):
            if os.path.isdir(c):
                cls_set.add(os.path.basename(c))
    class_names = sorted(c for c in cls_set if (classes is None or c in classes))
    items = []
    for di, dn in enumerate(domain_names):
        for ci, cn in enumerate(class_names):
            for p in sorted(glob.glob(os.path.join(glob.escape(os.path.join(root, dn, cn)), "*"))):
                if p.lower().endswith(EXTS):
                    items.append((p, ci, di))
    return items, class_names, domain_names


def few_shot(items: Sequence[tuple], k: int = 8, seed: int = 0) -> List[tuple]:
    rng = random.Random(seed)
    by: Dict[tuple, list] = {}
    for it in items:
        by.setdefault((it[2], it[1]), []).append(it)
    out = []
    for key in sorted(by):
        xs = list(by[key]); rng.shuffle(xs)
        out.extend(xs[:k])
    return out


def train_test_split(items: Sequence[tuple], test_ratio: float = 0.2, seed: int = 0):
    rng = random.Random(seed)
    xs = list(items); rng.shuffle(xs)
    n = int(len(xs) * test_ratio)
    return xs[n:], xs[:n]
