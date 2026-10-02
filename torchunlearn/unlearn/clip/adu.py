"""ADU -- Approximate Domain Unlearning for Vision-Language Models (Kawamura et al., NeurIPS 2025 spotlight,
arXiv 2510.08132).  Contrastive (CLIP) track.

Task: reduce zero-shot recognition accuracy on *forget domains* (e.g. illustrations) while keeping it on *retain
domains* (e.g. photos), over a shared class set.  CLIP stays frozen; trained parts are
    * deep vision prompts: n_ctx learnable tokens inserted after CLS in the first ``depth`` encoder layers
    * InstaPG: instance-wise prompt generator = cross-attention (query: prompts, key/value: patch tokens) in one
      intermediate block, so the prompt adapts to the style of each image
    * an auxiliary domain classifier on the image feature
Losses (paper):
    L_memorize = CE(zero-shot logits, y)                 on retain-domain images
    L_forget   = CE(zero-shot logits, uniform over C)    on forget-domain images   (maximise entropy)
    L_domain   = gamma * CE(domain classifier, d) - lambda * MMD^2(features of different domains)   (DDL)
    L = L_memorize + L_forget + L_domain,  gamma=30, lambda=10, few-shot (8 per domain), SGD lr 0.0025, 50 epochs.
Metrics: Mem = accuracy on retain domains, For = error rate on forget domains, H = harmonic mean.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ...nn.clipmodel import CLIPRobModel


class DomainImageDataset(Dataset):
    """items: (image | path, class_idx, domain_idx)."""

    def __init__(self, items: Sequence, rmodel: CLIPRobModel, image_root: Optional[str] = None):
        self.items, self.rmodel, self.image_root = list(items), rmodel, image_root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import os
        from PIL import Image
        im, y, d = self.items[i]
        if isinstance(im, str):
            p = im if (os.path.isabs(im) or not self.image_root) else os.path.join(self.image_root, im)
            im = Image.open(p).convert("RGB")
        pv = self.rmodel.processor(images=im, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pv, "labels": torch.tensor(y), "domains": torch.tensor(d)}


def make_domain_loader(items, rmodel, batch_size=32, shuffle=True, image_root=None):
    return DataLoader(DomainImageDataset(items, rmodel, image_root), batch_size=batch_size, shuffle=shuffle)


def _mmd2(x: torch.Tensor, y: torch.Tensor, sigmas=(1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    def k(a, b):
        d = torch.cdist(a, b) ** 2
        return sum(torch.exp(-d / (2 * s * s)) for s in sigmas) / len(sigmas)
    return k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()


class _InstaPG(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, prompts, patches):
        out, _ = self.attn(self.norm(prompts), self.norm(patches), self.norm(patches))
        return prompts + out


class ADU(nn.Module):
    def __init__(self, rmodel: CLIPRobModel, class_names: Sequence[str], n_domains: int, template: str = "a photo of a {}.",
                 n_ctx: int = 8, depth: int = 9, insta_layer: int = 1, gamma: float = 30.0, lam: float = 10.0, device=None):
        super().__init__()
        self.rmodel = rmodel
        self.device = device or rmodel.get_device()
        for p in rmodel.model.parameters():
            p.requires_grad_(False)
        vm = rmodel.model.vision_model
        dim = vm.config.hidden_size
        self.n_ctx, self.depth, self.insta_layer = int(n_ctx), int(depth), int(insta_layer)
        self.prompts = nn.ParameterList([nn.Parameter(torch.randn(self.n_ctx, dim) * 0.02) for _ in range(self.depth)])
        self.insta = _InstaPG(dim, n_heads=max(1, vm.config.num_attention_heads // 2))
        self.domain_clf = nn.Linear(rmodel.model.config.projection_dim, n_domains)
        self.gamma, self.lam = float(gamma), float(lam)
        self.class_names, self.template = list(class_names), template
        self.to(self.device)
        self._hooks = []
        with torch.no_grad():
            self.text_feats = rmodel.text_embeds([template.format(c) for c in class_names]).detach()

    # ------------------------------------------------------------------ prompt injection
    def _install(self):
        layers = self.rmodel.vision_encoder_layers()

        def make_hook(l):
            def hook(module, args, kwargs):
                h = args[0] if args else kwargs["hidden_states"]
                B = h.shape[0]
                P = self.prompts[l].unsqueeze(0).expand(B, -1, -1).to(h.dtype)
                if l == 0:
                    h = torch.cat([h[:, :1], P, h[:, 1:]], 1)
                else:
                    if l == self.insta_layer:
                        P = self.insta(P.float(), h[:, 1 + self.n_ctx:].float()).to(h.dtype)
                    h = torch.cat([h[:, :1], P, h[:, 1 + self.n_ctx:]], 1)
                if args:
                    return (h,) + tuple(args[1:]), kwargs
                kwargs = dict(kwargs); kwargs["hidden_states"] = h
                return args, kwargs
            return hook
        for l in range(self.depth):
            self._hooks.append(layers[l].register_forward_pre_hook(make_hook(l), with_kwargs=True))

    def _uninstall(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __enter__(self):
        self._install(); return self

    def __exit__(self, *a):
        self._uninstall()

    def image_feats(self, pixel_values):
        return self.rmodel.image_embeds(pixel_values)

    def logits(self, pixel_values):
        return self.rmodel.logit_scale().float() * self.image_feats(pixel_values) @ self.text_feats.T

    # ------------------------------------------------------------------ training
    def losses(self, batch, forget_domains: Iterable[int]):
        pv, y, d = batch["pixel_values"].to(self.device), batch["labels"].to(self.device), batch["domains"].to(self.device)
        feats = self.image_feats(pv)
        logits = self.rmodel.logit_scale().float() * feats @ self.text_feats.T
        fd = torch.tensor(sorted(set(forget_domains)), device=self.device)
        is_f = torch.isin(d, fd)
        out = {}
        out["memorize"] = F.cross_entropy(logits[~is_f], y[~is_f]) if (~is_f).any() else logits.new_zeros(())
        if is_f.any():
            logp = F.log_softmax(logits[is_f], -1)
            out["forget"] = -logp.mean()                        # CE towards the uniform distribution
        else:
            out["forget"] = logits.new_zeros(())
        out["dom_ce"] = F.cross_entropy(self.domain_clf(feats), d)
        mmd = logits.new_zeros(())
        doms = d.unique().tolist()
        for i in range(len(doms)):
            for j in range(i + 1, len(doms)):
                mmd = mmd + _mmd2(feats[d == doms[i]], feats[d == doms[j]])
        out["mmd2"] = mmd
        out["domain"] = self.gamma * out["dom_ce"] - self.lam * out["mmd2"]
        out["total"] = out["memorize"] + out["forget"] + out["domain"]
        return out

    def fit(self, loader, forget_domains: Iterable[int], n_epochs: int = 50, lr: float = 0.0025, log_every: int = 10):
        opt = torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=lr, momentum=0.9)
        self.history = []
        self._install()
        try:
            for ep in range(n_epochs):
                tot = {}
                for batch in loader:
                    l = self.losses(batch, forget_domains)
                    opt.zero_grad(); l["total"].backward(); opt.step()
                    for k, v in l.items():
                        tot[k] = tot.get(k, 0.0) + float(v.detach())
                row = {"epoch": ep + 1, **{k: v / max(len(loader), 1) for k, v in tot.items()}}
                self.history.append(row)
                if log_every and (ep + 1) % log_every == 0:
                    print(f"[ADU] epoch {ep + 1}: " + " ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "epoch"))
        finally:
            self._uninstall()
        return self

    @torch.no_grad()
    def evaluate(self, loader, forget_domains: Iterable[int]) -> Dict[str, float]:
        fd = set(forget_domains)
        self._install()
        try:
            c_r = n_r = c_f = n_f = 0
            for batch in loader:
                pred = self.logits(batch["pixel_values"].to(self.device)).argmax(-1).cpu()
                y, d = batch["labels"], batch["domains"]
                for p_, y_, d_ in zip(pred.tolist(), y.tolist(), d.tolist()):
                    if d_ in fd:
                        n_f += 1; c_f += int(p_ == y_)
                    else:
                        n_r += 1; c_r += int(p_ == y_)
        finally:
            self._uninstall()
        mem = c_r / max(n_r, 1)
        forget_err = 1.0 - c_f / max(n_f, 1)
        h = 2 * mem * forget_err / max(mem + forget_err, 1e-12)
        return {"Mem": mem, "For": forget_err, "H": h, "n_retain": n_r, "n_forget": n_f}

    def save(self, path: str):
        torch.save({"prompts": [p.detach().cpu() for p in self.prompts], "insta": self.insta.state_dict(),
                    "domain_clf": self.domain_clf.state_dict(), "class_names": self.class_names, "template": self.template,
                    "n_ctx": self.n_ctx, "depth": self.depth, "insta_layer": self.insta_layer}, path)
