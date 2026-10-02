"""Contrastive (CLIP / SigLIP) track: methods that act on dual-encoder vision-language models."""
from .adu import ADU, DomainImageDataset, make_domain_loader

__all__ = ["ADU", "DomainImageDataset", "make_domain_loader"]
