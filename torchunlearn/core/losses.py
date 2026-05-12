"""Reusable loss functions shared across unlearning algorithms.

Kept side-effect free per the project refactoring guide (§1.4 Minimize
side effects). Recording / logging is the engine's responsibility, not
the loss's.
"""
import torch
import torch.nn as nn


def classification_loss(model, x, y, device, reduction="mean"):
    """Standard cross-entropy classification loss.

    Args:
        model: callable returning logits of shape (B, C).
        x: input tensor.
        y: target labels of shape (B,).
        device: torch.device to move x, y to.
        reduction: "mean" (default) returns a scalar; otherwise returns the
            per-sample loss tensor of shape (B,).

    Returns:
        torch.Tensor: scalar loss if reduction == "mean", else (B,) tensor.
    """
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    per_sample = nn.CrossEntropyLoss(reduction="none")(logits, y)
    return per_sample.mean() if reduction == "mean" else per_sample


def l1_parameter_penalty(model):
    """Sum of L1 norms of every parameter tensor in the model.

    Used by :class:`L1Sparse` and others that apply weight-sparsity
    regularization. Centralized here so the regularizer is unit-testable
    in isolation.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        total = total + p.abs().sum()
    return total
