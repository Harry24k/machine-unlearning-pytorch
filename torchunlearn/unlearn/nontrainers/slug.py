"""SLUG -- Targeted Unlearning with Single Layer Unlearning Gradient (Cai et al., ICML 2025, arXiv 2407.11867;
code CSIPlab/SLUG).  Contrastive (CLIP) track; the updated vision encoder can be transplanted into an LVLM.

    1. one gradient computation on the original model:
           g_f = grad L_forget (cosine embedding loss  mean(1 - cos(v_i, t_i)) over the forget pairs)
           g_r = grad L_retain (CLIP contrastive loss over the retain pairs)
    2. per layer l:  importance_l = ||g_f,l||_2 / ||theta_l||_2,   alignment_l = cos(g_f,l, g_r,l)
       candidate layers = Pareto front (max importance, min alignment); the paper picks one layer from it
       (identity unlearning in CLIP ViT: e.g. ``vision_model.encoder.layers.9.self_attn.out_proj``)
    3. theta_l <- theta_l^(0) - lambda * g_f,l  with lambda found by binary search until the forget accuracy is
       ~0 while the test accuracy stays high (about 10 evaluations).

Loaders yield dicts {"pixel_values", "input_ids", "attention_mask"} (see :func:`make_clip_loader`).
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from ...nn.clipmodel import CLIPRobModel


class CLIPPairDataset(Dataset):
    """(image, text) pairs; images are PIL or paths."""

    def __init__(self, pairs: Sequence[Tuple], image_root: Optional[str] = None):
        self.pairs, self.image_root = list(pairs), image_root

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        from PIL import Image
        import os
        im, t = self.pairs[i][0], self.pairs[i][1]
        if isinstance(im, str):
            p = im if (os.path.isabs(im) or not self.image_root) else os.path.join(self.image_root, im)
            im = Image.open(p).convert("RGB")
        return im, t


def make_clip_loader(pairs, rmodel: CLIPRobModel, batch_size=32, shuffle=False, image_root=None) -> DataLoader:
    def collate(items):
        ims, ts = zip(*items)
        enc = rmodel.processor(images=list(ims), text=list(ts), return_tensors="pt", padding=True, truncation=True)
        return dict(enc)
    return DataLoader(CLIPPairDataset(pairs, image_root), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


class SLUG:
    def __init__(self, rmodel: CLIPRobModel, forget_loss: str = "cosine", layer_regex: str = r"vision_model\.encoder\.layers\.\d+\.(self_attn|mlp)\..*weight",
                 device=None):
        self.rmodel, self.forget_loss_name, self.layer_regex = rmodel, forget_loss, re.compile(layer_regex)
        self.device = device or rmodel.get_device()
        self.grads: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.table: List[dict] = []
        self.results: dict = {}

    def setup(self, **kw):
        return self

    # ------------------------------------------------------------------ gradients
    def _loss(self, batch, kind):
        pv = batch["pixel_values"].to(self.device)
        ids, am = batch["input_ids"].to(self.device), batch.get("attention_mask")
        am = am.to(self.device) if am is not None else None
        if kind == "forget" and self.forget_loss_name == "cosine":
            return self.rmodel.cosine_embedding_loss(pv, ids, am)
        return self.rmodel.contrastive_loss(pv, ids, am)

    def compute_gradients(self, forget_loader, retain_loader):
        params = {n: p for n, p in self.rmodel.model.named_parameters() if self.layer_regex.fullmatch(n)}
        if not params:
            raise ValueError(f"layer_regex matched no parameter")
        for p in self.rmodel.model.parameters():
            p.requires_grad_(False)
        for p in params.values():
            p.requires_grad_(True)
        self.rmodel.model.eval()
        acc = {}
        for kind, loader in (("forget", forget_loader), ("retain", retain_loader)):
            g = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
            nb = 0
            for batch in loader:
                self.rmodel.model.zero_grad(set_to_none=True)
                self._loss(batch, kind).backward()
                for n, p in params.items():
                    if p.grad is not None:
                        g[n] += p.grad.float()
                nb += 1
            for n in g:
                g[n] /= max(nb, 1)
            acc[kind] = g
        self.rmodel.model.zero_grad(set_to_none=True)
        self.grads = {n: (acc["forget"][n], acc["retain"][n]) for n in params}
        self.table = []
        for n, (gf, gr) in self.grads.items():
            theta = params[n].detach().float()
            imp = (gf.norm() / theta.norm().clamp(min=1e-12)).item()
            ali = torch.nn.functional.cosine_similarity(gf.flatten(), gr.flatten(), dim=0).item()
            self.table.append({"layer": n, "importance": imp, "alignment": ali})
        return self.table

    def pareto_front(self) -> List[dict]:
        rows = sorted(self.table, key=lambda r: -r["importance"])
        front, best_align = [], float("inf")
        for r in rows:                      # higher importance first; keep rows that improve (lower) alignment
            if r["alignment"] < best_align:
                front.append(r)
                best_align = r["alignment"]
        return front

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def _set(self, name, lam):
        p = dict(self.rmodel.model.named_parameters())[name]
        p.copy_((self._theta0 - lam * self.grads[name][0]).to(p.dtype))

    def fit(self, forget_loader=None, retain_loader=None, layer: Optional[str] = None, lam: Optional[float] = None,
            eval_fn: Optional[Callable[[CLIPRobModel], Tuple[float, float]]] = None, forget_target: float = 0.05,
            test_drop_tol: float = 0.02, lam_init: float = 1.0, n_search: int = 10, **kw):
        """eval_fn(rmodel) -> (forget_accuracy, test_accuracy).  Without eval_fn, ``lam`` is applied directly."""
        if not self.grads:
            if forget_loader is None or retain_loader is None:
                raise ValueError("compute_gradients needs forget_loader and retain_loader")
            self.compute_gradients(forget_loader, retain_loader)
        if layer is None:
            layer = self.pareto_front()[0]["layer"]
        self.layer = layer
        self._theta0 = dict(self.rmodel.model.named_parameters())[layer].detach().float().clone()
        trace = []
        if eval_fn is None:
            if lam is None:
                raise ValueError("give lam or eval_fn")
            self._set(layer, lam)
            self.results = {"layer": layer, "lam": lam, "trace": trace}
            return self
        f0, t0 = eval_fn(self.rmodel)
        trace.append({"lam": 0.0, "forget_acc": f0, "test_acc": t0})
        hi = lam_init
        for _ in range(8):                      # grow until the forget accuracy collapses
            self._set(layer, hi)
            f, t = eval_fn(self.rmodel)
            trace.append({"lam": hi, "forget_acc": f, "test_acc": t})
            if f <= forget_target:
                break
            hi *= 2
        lo, best = 0.0, hi
        for _ in range(n_search):
            mid = 0.5 * (lo + hi)
            self._set(layer, mid)
            f, t = eval_fn(self.rmodel)
            trace.append({"lam": mid, "forget_acc": f, "test_acc": t})
            if f <= forget_target and t >= t0 - test_drop_tol:
                best, hi = mid, mid
            elif f <= forget_target:
                hi = mid
            else:
                lo = mid
        self._set(layer, best)
        f, t = eval_fn(self.rmodel)
        self.results = {"layer": layer, "lam": best, "forget_acc": f, "test_acc": t, "forget_acc_0": f0, "test_acc_0": t0, "trace": trace}
        return self

    # ------------------------------------------------------------------ transfer to an LVLM
    @torch.no_grad()
    def apply_to_vlm(self, seq_rmodel) -> List[str]:
        """Copy the updated CLIP vision parameters into an LVLM whose vision tower is the same CLIP (e.g. LLaVA-1.5
        with openai/clip-vit-large-patch14-336).  Returns the LVLM parameter names that were overwritten."""
        src = dict(self.rmodel.model.named_parameters())
        tgt = dict(seq_rmodel.model.named_parameters())
        done = []
        for name in [self.layer] if getattr(self, "layer", None) else []:
            suffix = name.split("vision_model.", 1)[-1]
            hits = [n for n in tgt if n.endswith("vision_model." + suffix) and seq_rmodel.component_of(n) == "vision"]
            if len(hits) != 1:
                raise ValueError(f"could not map {name} into the LVLM (matches: {hits[:3]})")
            if tgt[hits[0]].shape != src[name].shape:
                raise ValueError(f"shape mismatch {name}: {src[name].shape} vs {tgt[hits[0]].shape}")
            tgt[hits[0]].copy_(src[name].to(tgt[hits[0]].dtype))
            done.append(hits[0])
        return done
