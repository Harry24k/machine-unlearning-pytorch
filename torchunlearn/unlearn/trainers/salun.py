"""SalUn: Saliency-Based Weight Masking for Machine Unlearning.

Reference:
    Fan et al., "SalUn: Empowering Machine Unlearning via Gradient-Based
    Weight Saliency in Both Image Classification and Generation",
    ICLR 2024.

Algorithm overview:
    SalUn identifies a sparse set of SALIENT weights whose gradients are
    largest on the forget set. Only those weights are updated during
    unlearning, protecting non-salient weights from corruption.

    Two phases:
    1. SALIENCY MASK COMPUTATION (offline, one-shot):
       - Compute gradient magnitudes of CE loss w.r.t. each parameter on Df.
       - Keep the top-k% of weights by gradient magnitude (saliency mask M).
       - Mask is binary: M_i = 1 if weight i is salient, else 0.

    2. MASKED RANDOM-LABEL UNLEARNING (online, per-step):
       - Apply random labels to forget samples.
       - Compute CE on both retain (normal labels) and forget (random labels).
       - Zero out gradients of non-salient weights before each optimizer step.
       - Only salient weights are updated; the rest remain frozen in-place.
"""

import torch
import torch.nn as nn
from typing import Optional

from .unlearner import Unlearner
from ...core.losses import classification_loss


class SalUn(Unlearner):
    """SalUn: saliency-masked random-label unlearning.

    Arguments:
        rmodel: model to unlearn (RobModel).
        saliency_threshold (float): fraction of weights to keep in the
            saliency mask (top-k by gradient magnitude). Default 0.5.
        retain_lambda (float): weight on the retain CE loss (default 1.0).
        forget_lambda (float): weight on the random-label CE loss (default 1.0).
        enforce_frozen (bool): restore non-salient weights after each
            optimizer step. Needed because SGD applies weight decay inside
            ``step()``, after gradient masking. Default True.
    """

    def __init__(
        self,
        rmodel,
        saliency_threshold: float = 0.5,
        retain_lambda: float = 1.0,
        forget_lambda: float = 1.0,
        enforce_frozen: bool = True,
    ):
        super().__init__(rmodel)
        self.saliency_threshold = saliency_threshold
        self.retain_lambda = retain_lambda
        self.forget_lambda = forget_lambda
        self.enforce_frozen = enforce_frozen
        self._saliency_mask: Optional[dict] = None

    # ------------------------------------------------------------------ API
    def compute_saliency_mask(self, forget_loader) -> None:
        """Compute and store the binary saliency mask.

        Call this ONCE before ``fit()``, passing the forget data loader.
        The mask is stored in ``self._saliency_mask`` and automatically
        applied during each ``calculate_cost`` call via the hook registered
        in ``fit``.

        Arguments:
            forget_loader: DataLoader over the forget set.
        """
        self.rmodel.eval()
        grad_accum = {
            n: torch.zeros_like(p)
            for n, p in self.rmodel.named_parameters()
            if p.requires_grad
        }

        n_batches = 0
        for x, y in forget_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.rmodel.zero_grad()
            loss = nn.CrossEntropyLoss()(self.rmodel(x), y)
            loss.backward()
            for n, p in self.rmodel.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grad_accum[n] += p.grad.abs()
            n_batches += 1

        if n_batches > 0:
            for n in grad_accum:
                grad_accum[n] /= n_batches

        # Flatten all gradients, find top-k threshold
        all_grads = torch.cat([g.view(-1) for g in grad_accum.values()])
        k = max(1, int(self.saliency_threshold * all_grads.numel()))
        threshold_val = all_grads.topk(k).values.min()

        self._saliency_mask = {
            n: (g >= threshold_val).float()
            for n, g in grad_accum.items()
        }
        self.rmodel.train()

    def calculate_cost(self, train_data, reduction: str = "mean"):
        """Compute SalUn loss (random-label CE on forget + CE on retain).

        After the backward pass, non-salient gradients are zeroed so that
        only salient weights receive updates. This is handled automatically
        by the gradient-masking hook registered by ``_register_grad_hook``.
        """
        if self._saliency_mask is None:
            raise RuntimeError(
                "Call compute_saliency_mask(forget_loader) before fit()."
            )

        x_r, y_r = train_data["Retain"]
        x_f, y_f = train_data["Forget"]
        x_r, y_r = x_r.to(self.device), y_r.to(self.device)
        x_f = x_f.to(self.device)

        # Random labels for the forget batch
        y_rand = torch.randint_like(y_f, 0, self.rmodel.n_classes).to(self.device)

        retain_loss = classification_loss(self.rmodel, x_r, y_r, self.device)
        forget_loss = nn.CrossEntropyLoss()(self.rmodel(x_f), y_rand)

        cost = self.retain_lambda * retain_loss + self.forget_lambda * forget_loss

        self.add_record_item("RetainLoss", retain_loss.item())
        self.add_record_item("ForgetLoss", forget_loss.item())
        self.add_record_item("Cost", cost.item())
        return cost

    def _update_weight(self, *inputs):
        """Overridden so the saliency mask is actually applied.

        The engine's default ``_update_weight`` goes straight from
        ``backward()`` to ``optimizer.step()``, which would leave the mask
        computed but unused -- reducing SalUn to plain random-label
        fine-tuning. We insert the mask between the two.
        """
        if self.minimizer is not None:
            self.minimizer.step(self.calculate_cost, *inputs)
            return

        cost = self.calculate_cost(*inputs)
        self.optimizer.zero_grad()
        cost.backward()
        self.apply_saliency_mask_to_grads()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.rmodel.parameters(), self.clip_grad_norm
            )

        # Zeroing .grad is not sufficient on its own: SGD adds
        # weight_decay * p to the gradient INSIDE step(), so non-salient
        # weights would still decay every iteration -- exactly the drift the
        # mask exists to prevent. Snapshot and restore them.
        if self.enforce_frozen and self._saliency_mask is not None:
            frozen = {
                n: p.detach().clone()
                for n, p in self.rmodel.named_parameters()
                if n in self._saliency_mask
            }
            self.optimizer.step()
            with torch.no_grad():
                for n, p in self.rmodel.named_parameters():
                    if n in frozen:
                        keep = self._saliency_mask[n].to(p.device)
                        p.copy_(keep * p + (1 - keep) * frozen[n])
        else:
            self.optimizer.step()

    def apply_saliency_mask_to_grads(self) -> None:
        """Zero out gradients of non-salient parameters.

        Called by :meth:`_update_weight` between ``backward()`` and
        ``optimizer.step()``.
        """
        if self._saliency_mask is None:
            return
        for name, param in self.rmodel.named_parameters():
            if param.grad is not None and name in self._saliency_mask:
                param.grad.mul_(self._saliency_mask[name].to(param.grad.device))
