import torch
import torch.nn as nn

from .unlearner import Unlearner
from ...core.losses import l1_parameter_penalty


class NegGrad(Unlearner):
    r"""Negative-gradient unlearning (Golatkar et al., 2020).

    Applies a *negative* cross-entropy gradient on the forget batch, optionally
    blended with a positive gradient on a retain batch (`retain_lambda`) and an
    L1 weight-sparsity penalty (`l1_penalty_lambda`).

    Arguments:
        rmodel (nn.Module): model to unlearn.
        retain_lambda (float): weight in [0, 1] for the retain-set loss.
        l1_penalty_lambda (float): coefficient of the L1 parameter penalty.
    """

    def __init__(self, rmodel, retain_lambda=0.0, l1_penalty_lambda=0.0):
        super().__init__(rmodel)
        self.retain_lambda = retain_lambda
        self.l1_penalty_lambda = l1_penalty_lambda

    def calculate_cost(self, train_data, reduction="mean"):
        """Overridden. Returns the (possibly per-sample) NegGrad cost."""
        x, y, n_forget = self._build_batch(train_data)
        logits = self.rmodel(x.to(self.device))
        y = y.to(self.device)

        ce_none = nn.CrossEntropyLoss(reduction="none")
        fg_loss = -ce_none(logits[:n_forget], y[:n_forget])

        rt_loss = torch.zeros_like(fg_loss)
        if self.retain_lambda > 0:
            rt_loss = ce_none(logits[n_forget:], y[n_forget:])
            self.add_record_item("RTLoss", rt_loss.mean().item())

        l1_loss = l1_parameter_penalty(self.rmodel)
        cost = ((1 - self.retain_lambda) * fg_loss
                + self.retain_lambda * rt_loss
                + self.l1_penalty_lambda * l1_loss)

        self.add_record_item("FGLoss", fg_loss.mean().item())
        self.add_record_item("L1Loss", l1_loss.item())
        self.add_record_item("Cost", cost.mean().item())
        return cost.mean() if reduction == "mean" else cost

    def _build_batch(self, train_data):
        """Concatenate forget (+ optionally retain) batches; return (x, y, n_forget)."""
        x_forget, y_forget = train_data["Forget"]
        n_forget = len(y_forget)
        if self.retain_lambda > 0:
            x_retain, y_retain = train_data["Retain"]
            x = torch.cat([x_forget, x_retain])
            y = torch.cat([y_forget, y_retain])
        else:
            x, y = x_forget, y_forget
        return x, y, n_forget
