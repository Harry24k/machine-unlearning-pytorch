import torch
import torch.nn as nn

from .unlearner import Unlearner
from ...core.losses import classification_loss, l1_parameter_penalty


class L1Sparse(Unlearner):
    r"""L1-sparsity regularized unlearning (Jia et al., 2023).

    Fine-tunes on the Retain split with an L1 weight-penalty term to push
    parameters toward zero, encouraging a sparser (more "forgotten") model.

    Arguments:
        rmodel (nn.Module): model to unlearn.
        gamma (float): L1 regularization coefficient (default: 1e-5).
    """

    def __init__(self, rmodel, gamma: float = 1e-5):
        super().__init__(rmodel)
        self.gamma = gamma

    # -------------------------------------------------------------- cost
    def calculate_cost(self, train_data, reduction: str = "mean"):
        """Cross-entropy on Retain + L1 parameter penalty."""
        x, y = train_data["Retain"]

        retain_loss = classification_loss(
            self.rmodel, x, y, self.device, reduction="none"
        )
        l1_loss = l1_parameter_penalty(self.rmodel)
        cost = retain_loss + self.gamma * l1_loss

        self.add_record_item("RTLoss", retain_loss.mean().item())
        self.add_record_item("L1Loss", l1_loss.item())
        self.add_record_item("Cost", cost.mean().item())

        return cost.mean() if reduction == "mean" else cost
