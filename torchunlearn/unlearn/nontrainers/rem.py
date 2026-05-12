"""REM: Redirection for Erasing Memory.

Reference:
    "Towards a Universal Unlearning Method for Corrupted Data",
    Google DeepMind. Official code: https://github.com/google-deepmind/rem

Algorithm overview:
    REM is designed for CORRUPTED DATA unlearning (not the standard random/class
    forgetting scenario). It handles discovered mislabelled or poisoned samples.

    Core mechanism:
    Each Linear/Conv layer gains a second branch theta2 (expansion) in addition
    to the original base branch theta1. A per-sample binary mask table routes
    each training sample to a unique mask vector over theta2.
    All discovered forget samples Df share ONE mask id (mask 0).

    Four steps:
    Step 1 - Wrap: replace eligible layers with REM wrappers (theta1 + theta2).
    Step 2 - Remove: gradient ascent on Df through theta1 only until forget
             accuracy drops below threshold gamma.
    Step 3 - Repair: jointly train theta1 and theta2.
             - Full model (theta1 + theta2): repair utility on all of Dtr.
             - Base-only barrier: gradient ascent on Df through theta1 to
               prevent corruption from creeping back into the base branch.
    Step 4 - Drop: discard theta2, keep theta1 as the clean unlearned model.

Dataloader format:
    Both train_loader and forget_loader must return (idx, x, y) triples
    where idx is the sample index in the full training dataset.
    Use IndexedDataset to wrap a standard (x, y) dataset.

Adapted from the provided implementation skeleton.
For exact paper reproduction use the official repo linked above.
"""

from __future__ import annotations

import copy
import itertools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Global REM forward-pass state
# ---------------------------------------------------------------------------

class _REMState:
    use_expansion: bool = False
    mask_ids: Optional[torch.Tensor] = None


@contextmanager
def rem_forward(mask_ids: Optional[torch.Tensor] = None,
                use_expansion: bool = False):
    """Context manager: controls whether the expansion branch theta2 is active.

    Base-only (theta1):
        with rem_forward(use_expansion=False):
            logits = model(x)

    Full (theta1 + theta2):
        with rem_forward(mask_ids=batch_mask_ids, use_expansion=True):
            logits = model(x)
    """
    old_use = _REMState.use_expansion
    old_mask = _REMState.mask_ids
    _REMState.use_expansion = use_expansion
    _REMState.mask_ids = mask_ids
    try:
        yield
    finally:
        _REMState.use_expansion = old_use
        _REMState.mask_ids = old_mask


# ---------------------------------------------------------------------------
# Mask table helper
# ---------------------------------------------------------------------------

def _make_binary_mask_table(num_masks: int, width: int,
                             active_ratio: float) -> torch.Tensor:
    if not (0.0 < active_ratio <= 1.0):
        raise ValueError("active_ratio must be in (0, 1].")
    masks = (torch.rand(num_masks, width) < active_ratio).float()
    empty = masks.sum(dim=1) == 0
    if empty.any():
        rand_cols = torch.randint(0, width, (int(empty.sum().item()),))
        masks[empty, rand_cols] = 1.0
    return masks


# ---------------------------------------------------------------------------
# REM Layer wrappers
# ---------------------------------------------------------------------------

class REMLinear(nn.Module):
    """Linear layer with REM expansion branch theta2."""

    def __init__(self, base: nn.Linear, num_masks: int, active_ratio: float = 0.2):
        super().__init__()
        self.base = copy.deepcopy(base)
        self.expansion = nn.Linear(base.in_features, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.expansion.weight, a=5 ** 0.5)
        self.register_buffer(
            "mask_table",
            _make_binary_mask_table(num_masks, base.out_features, active_ratio),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not _REMState.use_expansion:
            return y
        if _REMState.mask_ids is None:
            raise RuntimeError("mask_ids must be set via rem_forward()")
        e = self.expansion(x)
        mask = self.mask_table[_REMState.mask_ids.to(e.device)]
        while mask.dim() < e.dim():
            mask = mask.unsqueeze(1)
        return y + e * mask


class REMConv2d(nn.Module):
    """Conv2d layer with REM expansion branch theta2."""

    def __init__(self, base: nn.Conv2d, num_masks: int, active_ratio: float = 0.2):
        super().__init__()
        self.base = copy.deepcopy(base)
        self.expansion = nn.Conv2d(
            base.in_channels, base.out_channels,
            base.kernel_size, base.stride, base.padding,
            base.dilation, base.groups, bias=False,
            padding_mode=base.padding_mode,
        )
        nn.init.kaiming_normal_(self.expansion.weight, nonlinearity="relu")
        self.register_buffer(
            "mask_table",
            _make_binary_mask_table(num_masks, base.out_channels, active_ratio),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not _REMState.use_expansion:
            return y
        if _REMState.mask_ids is None:
            raise RuntimeError("mask_ids must be set via rem_forward()")
        e = self.expansion(x)
        mask = self.mask_table[_REMState.mask_ids.to(e.device)].view(
            e.size(0), e.size(1), 1, 1)
        return y + e * mask


# ---------------------------------------------------------------------------
# Model wrapping / unwrapping
# ---------------------------------------------------------------------------

def expand_model_for_rem(
    model: nn.Module,
    num_masks: int,
    active_ratio: float = 0.2,
    wrap_linear: bool = True,
    wrap_conv2d: bool = True,
) -> nn.Module:
    """Deep-copy model and replace eligible layers with REM wrappers.

    The model forward(x) API is preserved unchanged.
    """
    model = copy.deepcopy(model)
    for name, child in list(model.named_children()):
        if wrap_linear and isinstance(child, nn.Linear):
            setattr(model, name, REMLinear(child, num_masks, active_ratio))
        elif wrap_conv2d and isinstance(child, nn.Conv2d):
            setattr(model, name, REMConv2d(child, num_masks, active_ratio))
        else:
            setattr(model, name, expand_model_for_rem(
                child, num_masks, active_ratio, wrap_linear, wrap_conv2d))
    return model


def drop_rem_expansion(model: nn.Module) -> nn.Module:
    """Drop theta2 in-place; return a standard model containing only theta1."""
    for name, child in list(model.named_children()):
        if isinstance(child, (REMLinear, REMConv2d)):
            setattr(model, name, child.base)
        else:
            drop_rem_expansion(child)
    return model


def set_rem_trainable(model: nn.Module,
                      train_base: bool,
                      train_expansion: bool) -> None:
    """Control which REM branch receives gradients."""
    for module in model.modules():
        if isinstance(module, (REMLinear, REMConv2d)):
            for p in module.base.parameters():
                p.requires_grad_(train_base)
            for p in module.expansion.parameters():
                p.requires_grad_(train_expansion)


# ---------------------------------------------------------------------------
# Mask assignment
# ---------------------------------------------------------------------------

def make_mask_assignment(
    dataset_size: int,
    forget_indices: Sequence[int],
    num_masks: int,
    shared_forget_mask_id: int = 0,
    seed: int = 0,
) -> torch.Tensor:
    """Assign mask IDs to all training samples.

    All forget samples share mask 0.
    Other samples receive random masks from 1..num_masks-1.
    """
    if num_masks < 2:
        raise ValueError("num_masks must be >= 2 (mask 0 reserved for Df).")
    g = torch.Generator().manual_seed(seed)
    assignment = torch.randint(1, num_masks, (dataset_size,), generator=g)
    assignment[list(forget_indices)] = shared_forget_mask_id
    return assignment.long()


# ---------------------------------------------------------------------------
# IndexedDataset wrapper
# ---------------------------------------------------------------------------

class IndexedDataset(torch.utils.data.Dataset):
    """Wraps any (x, y) dataset into one returning (idx, x, y)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx][0], self.dataset[idx][1]
        return idx, x, y


# ---------------------------------------------------------------------------
# REM Config and training loop
# ---------------------------------------------------------------------------

@dataclass
class REMConfig:
    lr: float = 5e-4
    max_epochs: int = 5
    gamma: float = 0.2
    lambda_barrier: float = 1.0
    removal_steps_per_epoch: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def accuracy_base_only(model: nn.Module, loader, device: str) -> float:
    """Evaluate base-only accuracy (theta1 only, expansion OFF)."""
    model.eval()
    correct = total = 0
    with rem_forward(use_expansion=False):
        for batch in loader:
            _, x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def rem_unlearn(
    rem_model: nn.Module,
    train_loader,
    forget_loader,
    mask_assignment: torch.Tensor,
    cfg: REMConfig,
) -> nn.Module:
    """REM-style unlearning loop.

    Steps 2 and 3 of the REM algorithm (Step 1 = expand_model_for_rem,
    Step 4 = drop_rem_expansion). Both loaders must return (idx, x, y).

    Returns:
        The rem_model after training (still contains theta2).
        Call drop_rem_expansion(rem_model) afterward.
    """
    device = cfg.device
    rem_model.to(device)
    forget_cycle = itertools.cycle(forget_loader)
    mask_assignment = mask_assignment.cpu()
    optimizer = torch.optim.AdamW(
        [p for p in rem_model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=1e-4,
    )

    for epoch in range(cfg.max_epochs):
        # --- Step 2: remove corruption from theta1 (gradient ascent on Df) ---
        set_rem_trainable(rem_model, train_base=True, train_expansion=False)
        rem_model.train()
        forget_acc = accuracy_base_only(rem_model, forget_loader, device)
        for _ in range(cfg.removal_steps_per_epoch):
            if forget_acc <= cfg.gamma:
                break
            for batch in forget_loader:
                _, x_f, y_f = batch
                x_f, y_f = x_f.to(device), y_f.to(device)
                optimizer.zero_grad(set_to_none=True)
                with rem_forward(use_expansion=False):
                    logits_f = rem_model(x_f)
                loss = -F.cross_entropy(logits_f, y_f)
                loss.backward()
                optimizer.step()
            forget_acc = accuracy_base_only(rem_model, forget_loader, device)

        # --- Step 3: repair utility on Dtr; redirect Df into theta2 ---
        set_rem_trainable(rem_model, train_base=True, train_expansion=True)
        rem_model.train()
        for batch in train_loader:
            idx, x, y = batch
            idx = idx.long()
            x, y = x.to(device), y.to(device)
            mask_ids = mask_assignment[idx.cpu()].to(device)
            _, x_f, y_f = next(forget_cycle)
            x_f, y_f = x_f.to(device), y_f.to(device)
            optimizer.zero_grad(set_to_none=True)
            # 3.1 full model: utility on Dtr
            with rem_forward(mask_ids=mask_ids, use_expansion=True):
                logits = rem_model(x)
            loss_redirect = F.cross_entropy(logits, y)
            # 3.2 base-only barrier: prevent Df from returning to theta1
            with rem_forward(use_expansion=False):
                logits_f_base = rem_model(x_f)
            loss_barrier = -F.cross_entropy(logits_f_base, y_f)
            loss = loss_redirect + cfg.lambda_barrier * loss_barrier
            loss.backward()
            optimizer.step()

        fa = accuracy_base_only(rem_model, forget_loader, device)
        print(f"[REM] epoch {epoch+1}/{cfg.max_epochs}  forget_acc(base)={fa:.4f}")

    return rem_model


__all__ = [
    "rem_forward", "expand_model_for_rem", "drop_rem_expansion",
    "set_rem_trainable", "make_mask_assignment", "IndexedDataset",
    "REMConfig", "accuracy_base_only", "rem_unlearn",
    "REMLinear", "REMConv2d",
]
