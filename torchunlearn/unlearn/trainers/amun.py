"""AMUN: Adversarial Machine UNlearning.

Reference:
    Ebrahimpour-Boroojeny, Sundaram & Chandrasekaran,
    "Not All Wrong is Bad: Using Adversarial Examples for Unlearning",
    ICML 2025.  https://icml.cc/virtual/2025/poster/46097

Algorithm overview:
    For each forget sample x, find the *nearest* adversarial example x_adv
    (a minimally-perturbed input the model misclassifies) and record the
    label y_adv the model assigns to it. Fine-tuning on (x_adv, y_adv)
    pulls the decision boundary across x itself, which lowers the model's
    confidence on x while leaving the global decision surface -- and hence
    test accuracy -- largely intact.

    Because the adversarial examples lie on the model's own data manifold,
    the edit is local: unlike random relabeling, it does not inject
    off-manifold targets that damage retain performance.

Notes on this implementation:
    This is a reimplementation from the paper description, not a port of
    the authors' code. The paper uses a minimum-norm attack (DeepFool
    family) to find the closest adversarial example. We default to
    DeepFool for fidelity and expose PGD as a cheaper alternative.
"""

import torch
import torch.nn as nn

from .unlearner import Unlearner
from ...core.losses import classification_loss


class AMUN(Unlearner):
    r"""Adversarial Machine UNlearning (Ebrahimpour-Boroojeny et al., ICML 2025).

    Arguments:
        rmodel (RobModel): model to unlearn.
        attack (str): "deepfool" (minimum-norm, faithful to the paper) or
            "pgd" (fixed-budget, faster). Default "deepfool".
        eps (float): L-inf budget, used only when ``attack="pgd"``.
        steps (int): attack iterations.
        retain_lambda (float): weight on the retain cross-entropy term.
        adv_lambda (float): weight on the adversarial-example term.
    """

    def __init__(
        self,
        rmodel,
        attack: str = "deepfool",
        eps: float = 8 / 255,
        steps: int = 20,
        retain_lambda: float = 1.0,
        adv_lambda: float = 1.0,
    ):
        super().__init__(rmodel)
        self.attack_name = attack.lower()
        self.eps = eps
        self.steps = steps
        self.retain_lambda = retain_lambda
        self.adv_lambda = adv_lambda
        self._atk = None

    def _build_attack(self):
        if self._atk is not None:
            return self._atk
        if self.attack_name == "deepfool":
            from ...attacks.attacks.deepfool import DeepFool
            self._atk = DeepFool(self.rmodel, steps=self.steps)
        elif self.attack_name == "pgd":
            from ...attacks.attacks.pgd import PGD
            self._atk = PGD(self.rmodel, eps=self.eps,
                            alpha=self.eps / 4, steps=self.steps)
        else:
            raise ValueError("attack must be 'deepfool' or 'pgd'.")
        return self._atk

    @torch.no_grad()
    def _adv_labels(self, x_adv):
        """Label each adversarial example with the model's own prediction."""
        was_training = self.rmodel.training
        self.rmodel.eval()
        y_adv = self.rmodel(x_adv).argmax(dim=1)
        if was_training:
            self.rmodel.train()
        return y_adv

    def calculate_cost(self, train_data, reduction: str = "mean"):
        """Retain CE + CE on (nearest adversarial example, adversarial label)."""
        if not isinstance(train_data, dict):
            raise TypeError(
                "%s needs Retain and Forget batches together. Wrap your loaders "
                "with MergedLoaders({'Retain': ..., 'Forget': ...}) and pass that "
                "to fit()." % type(self).__name__
            )
        x_r, y_r = train_data["Retain"]
        x_f, y_f = train_data["Forget"]
        x_f, y_f = x_f.to(self.device), y_f.to(self.device)

        # Generating x_adv requires grads w.r.t. the input, not the weights.
        was_training = self.rmodel.training
        self.rmodel.eval()
        atk = self._build_attack()
        with torch.enable_grad():
            x_adv = atk(x_f, y_f).detach()
        if was_training:
            self.rmodel.train()

        y_adv = self._adv_labels(x_adv)

        retain_loss = classification_loss(self.rmodel, x_r, y_r, self.device)
        adv_loss = nn.CrossEntropyLoss()(self.rmodel(x_adv), y_adv)
        cost = self.retain_lambda * retain_loss + self.adv_lambda * adv_loss

        self.add_record_item("RetainLoss", retain_loss.item())
        self.add_record_item("AdvLoss", adv_loss.item())
        self.add_record_item("Cost", cost.item())
        return cost
