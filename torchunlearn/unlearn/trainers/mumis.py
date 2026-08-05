"""MU-Mis: Machine Unlearning by Minimizing input sensitivity.

Reference:
    Cheng, Huang, Zhou, He, Yang, Wu & Huang,
    "Remaining-data-free Machine Unlearning by Suppressing Sample
    Contribution", ICLR 2026.  arXiv:2402.15109
    Official code: github.com/poppopbean0903/MU-Mis

Algorithm overview:
    A sample's contribution to training shows up as the trained model's
    increased *input sensitivity* to it. Unlearning should therefore
    withdraw that contribution by suppressing the sensitivity directly,
    rather than by heuristics like random relabeling or distillation --
    which damage utility and then need retain data to repair it.

    For each forget sample x with label c, MU-Mis minimizes

        L = mean_x [ ||grad_x f_c(x)||^2  -  ||grad_x f_c'(x)||^2 ]

    where f_c is the target-class logit and f_c' the logits of the other
    classes. This lowers sensitivity along the true class while raising it
    elsewhere.

    The distinguishing property is that the loss touches ONLY the forget
    set: MU-Mis is remaining-data-free, and was the first such method
    reported to match retain-dependent methods.

Notes on this implementation:
    Reimplemented from the paper description, not a port of the official
    code. Requires a double backward (grad-of-grad), so the model must be
    twice differentiable -- avoid in-place ReLU in custom architectures.
"""

import torch

from .unlearner import Unlearner


class MUMis(Unlearner):
    r"""MU-Mis: retain-free unlearning by minimizing input sensitivity.

    Because no retain data is used, pass the Forget loader directly to
    ``fit`` (a ``MergedLoaders`` dict is also accepted and its ``Forget``
    entry is used).

    Arguments:
        rmodel (RobModel): model to unlearn.
        other_lambda (float): weight on the non-target sensitivity term
            that is *maximized*. Default 1.0.
        n_other (int): how many non-target classes to sample per example.
            ``None`` uses all classes. Sampling keeps the double backward
            affordable on datasets with many classes.
    """

    def __init__(self, rmodel, other_lambda: float = 1.0, n_other: int = 1):
        super().__init__(rmodel)
        self.other_lambda = other_lambda
        self.n_other = n_other

    @staticmethod
    def _sensitivity_sq(logit_sum, x):
        """Squared L2 norm of d(logit_sum)/dx, per sample, keeping the graph."""
        grad = torch.autograd.grad(
            logit_sum, x, create_graph=True, retain_graph=True
        )[0]
        return grad.flatten(1).pow(2).sum(dim=1)

    def calculate_cost(self, train_data, reduction: str = "mean"):
        """||grad_x f_c||^2 - lambda * ||grad_x f_c'||^2 over the forget set."""
        if isinstance(train_data, dict):
            x, y = train_data["Forget"]
        else:
            x, y = train_data
        x = x.to(self.device).requires_grad_(True)
        y = y.to(self.device)

        logits = self.rmodel(x)
        n_classes = logits.size(1)

        target_logit = logits.gather(1, y.view(-1, 1)).sum()
        target_sens = self._sensitivity_sq(target_logit, x)

        # Pick non-target classes to push sensitivity toward.
        if self.n_other is None:
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, y.view(-1, 1), False)
            other_logit = logits[mask].sum()
        else:
            offset = torch.randint(
                1, n_classes, (x.size(0), self.n_other), device=self.device
            )
            other_idx = (y.view(-1, 1) + offset) % n_classes
            other_logit = logits.gather(1, other_idx).sum()
        other_sens = self._sensitivity_sq(other_logit, x)

        per_sample = target_sens - self.other_lambda * other_sens

        self.add_record_item("TargetSens", target_sens.mean().item())
        self.add_record_item("OtherSens", other_sens.mean().item())
        self.add_record_item("Cost", per_sample.mean().item())

        return per_sample.mean() if reduction == "mean" else per_sample
