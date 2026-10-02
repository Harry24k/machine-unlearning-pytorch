"""CLIPRobModel -- wrapper for dual-encoder vision-language models (CLIP / SigLIP) = the *contrastive* track.

The token-level methods in ``trainers/seq`` do not apply to these models (no next-token likelihood); the CLIP
track has its own methods: SLUG (ICML 2025, ``nontrainers/slug.py``) and ADU (NeurIPS 2025, ``unlearn/clip/adu.py``).

Surface used by those methods:
    rmodel.model                 HF CLIPModel / SiglipModel
    rmodel.processor             AutoProcessor (image_processor + tokenizer)
    rmodel.image_embeds(pixel_values)   L2-normalised [B, D]
    rmodel.text_embeds(texts | input_ids, attention_mask)   L2-normalised [N, D]
    rmodel.logit_scale()
    rmodel.zero_shot_logits(pixel_values, class_texts)      [B, C]
    rmodel.contrastive_loss(pixel_values, input_ids, attention_mask)  symmetric InfoNCE (the CLIP pre-training loss)
    rmodel.cosine_embedding_loss(pixel_values, input_ids, attention_mask)  mean(1 - cos(v_i, t_i))  (SLUG forget loss)
    rmodel.apply_trainable("vision" | "text" | "all" | "none" | "re:<regex>")
"""
from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .robmodel import RobModel
from .seqmodel import DTYPES, CapabilityError


class CLIPRobModel(RobModel):
    def __init__(self, model, processor, device=None, name: Optional[str] = None):
        nn.Module.__init__(self)
        if device is not None:
            model = model.to(device)
        self.model, self.processor = model, processor
        self.name = name or getattr(model.config, "_name_or_path", model.__class__.__name__)
        self.device = device or next(model.parameters()).device
        self.register_buffer("n_classes", torch.tensor(0))
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    @classmethod
    def from_pretrained(cls, path: str, dtype: str = "fp32", device=None, trainable: str = "all",
                        local_files_only: bool = False, **kw) -> "CLIPRobModel":
        from transformers import AutoModel, AutoProcessor
        load_kw = {"local_files_only": local_files_only}
        if DTYPES[dtype] != "auto":
            load_kw["dtype"] = DTYPES[dtype]
        load_kw.update(kw)
        model = AutoModel.from_pretrained(path, **load_kw)
        if not (hasattr(model, "get_image_features") and hasattr(model, "get_text_features")):
            raise CapabilityError(f"{path} is not a dual-encoder (needs get_image_features / get_text_features)")
        processor = AutoProcessor.from_pretrained(path, local_files_only=local_files_only)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj = cls(model, processor, device=device, name=path)
        obj.apply_trainable(trainable)
        return obj

    # ------------------------------------------------------------------ surface
    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def forward(self, *a, **k):
        return self.model(*a, **k)

    def get_device(self):
        return next(self.model.parameters()).device

    def logit_scale(self) -> torch.Tensor:
        ls = getattr(self.model, "logit_scale", None)
        return ls.exp() if ls is not None else torch.tensor(100.0, device=self.get_device())

    def preprocess_images(self, images) -> torch.Tensor:
        return self.processor(images=list(images), return_tensors="pt")["pixel_values"].to(self.get_device())

    def tokenize(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        return {k: v.to(self.get_device()) for k, v in enc.items()}

    def image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        e = self.model.get_image_features(pixel_values=pixel_values.to(self.get_device(), next(self.model.parameters()).dtype))
        return F.normalize(e.float(), dim=-1)

    def text_embeds(self, texts=None, input_ids=None, attention_mask=None) -> torch.Tensor:
        if texts is not None:
            enc = self.tokenize(texts)
            input_ids, attention_mask = enc["input_ids"], enc.get("attention_mask")
        kw = {"input_ids": input_ids.to(self.get_device())}
        if attention_mask is not None:
            kw["attention_mask"] = attention_mask.to(self.get_device())
        e = self.model.get_text_features(**kw)
        return F.normalize(e.float(), dim=-1)

    def zero_shot_logits(self, pixel_values: torch.Tensor, class_texts: Sequence[str]) -> torch.Tensor:
        return self.logit_scale().float() * self.image_embeds(pixel_values) @ self.text_embeds(class_texts).T

    def contrastive_loss(self, pixel_values, input_ids, attention_mask=None) -> torch.Tensor:
        v, t = self.image_embeds(pixel_values), self.text_embeds(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.logit_scale().float() * v @ t.T
        target = torch.arange(len(v), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))

    def cosine_embedding_loss(self, pixel_values, input_ids, attention_mask=None) -> torch.Tensor:
        v, t = self.image_embeds(pixel_values), self.text_embeds(input_ids=input_ids, attention_mask=attention_mask)
        return (1.0 - (v * t).sum(-1)).mean()

    # ------------------------------------------------------------------ parameters
    def component_of(self, name: str) -> str:
        if name.startswith(("vision_model", "visual_projection")):
            return "vision"
        if name.startswith(("text_model", "text_projection")):
            return "text"
        return "other"

    def apply_trainable(self, spec: Optional[str]) -> Dict[str, int]:
        if spec is None:
            return self.trainable_summary()
        if spec.startswith("re:"):
            rx = re.compile(spec[3:])
            for n, p in self.model.named_parameters():
                p.requires_grad_(rx.fullmatch(n) is not None)
        elif spec in ("all", "none"):
            for p in self.model.parameters():
                p.requires_grad_(spec == "all")
        else:
            comps = set(spec.split("+"))
            if comps - {"vision", "text"}:
                raise ValueError(f"unknown trainable spec {spec!r}")
            for n, p in self.model.named_parameters():
                p.requires_grad_(self.component_of(n) in comps)
        if not any(p.requires_grad for p in self.model.parameters()) and spec != "none":
            raise CapabilityError(f"trainable={spec!r} matched nothing")
        return self.trainable_summary()

    def trainable_summary(self) -> Dict[str, int]:
        out = {"vision": 0, "text": 0, "other": 0}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                out[self.component_of(n)] += p.numel()
        return out

    def vision_encoder_layers(self) -> nn.ModuleList:
        return self.model.vision_model.encoder.layers

    # ------------------------------------------------------------------ evaluation helpers
    @torch.no_grad()
    def zero_shot_accuracy(self, loader, class_texts: Sequence[str]) -> float:
        """loader yields (pixel_values, labels) or dicts with those keys."""
        self.model.eval()
        t = self.text_embeds(class_texts)
        correct, n = 0, 0
        for batch in loader:
            if isinstance(batch, dict):
                pv, y = batch["pixel_values"], batch["labels"]
            else:
                pv, y = batch[0], batch[1]
            logits = self.logit_scale().float() * self.image_embeds(pv) @ t.T
            correct += int((logits.argmax(-1).cpu() == y.cpu()).sum())
            n += len(y)
        return correct / max(n, 1)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    def save_dict(self, save_path):
        self.save_pretrained(save_path)

    def eval_accuracy(self, *a, **k):
        raise NotImplementedError("use zero_shot_accuracy(loader, class_texts)")

    def __repr__(self):
        return f"CLIPRobModel(name={self.name!r}, trainable={self.trainable_summary()})"
