"""Boundary Shrink: Unlearning by Shrinking the Decision Boundary.

Reference:
    Chen et al., "Boundary Unlearning: Rapid Forgetting of Deep Networks
    via Shifting the Decision Boundary", CVPR 2023.

Algorithm overview:
    For class-wise forgetting, the model is pushed to misclassify forget-class
    samples into NEARBY (adversarially closest) classes instead of random ones.
    This "shrinks" the decision boundary of the forget class.

    Two interleaved objectives:
    1. BOUNDARY SHRINK on Df:
       Find the nearest non-forget class c* for each forget sample x_f using
       the model's current logits, then minimise CE(model(x_f), c*).
       This pulls the decision boundary away from the forget class.

    2. RETAIN REPAIR on Dr:
       Standard cross-entropy on retain samples to prevent catastrophic
       forgetting of retained classes.

Notes:
    - Designed primarily for class-wise forgetting (omit_label must be set).
    - For random forgetting, nearest-class re-targeting still applies but
      the forget set contains samples from mixed classes.
"""

import torch
import torch.nn as nn

from .unlearner import Unlearner
from ...core.losses import classification_loss


class BoundaryShrink(Unlearner):
    """Boundary Shrink unlearning via nearest-class re-targeting.

    Arguments:
        rmodel: model to unlearn (RobModel).
        omit_label (int or None): the class label to forget.
            If None the nearest-class redirect applies to all forget samples
            regardless of their true class.
        retain_lambda (float): weight on the retain CE loss (default 1.0).
        forget_lambda (float): weight on the boundary-shrink loss (default 1.0).
    """

    def __init__(
        self,
        rmodel,
        omit_label: int = None,
        retain_lambda: float = 1.0,
        forget_lambda: float = 1.0,
    ):
        super().__init__(rmodel)
        self.omit_label = omit_label
        self.retain_lambda = retain_lambda
        self.forget_lambda = forget_lambda

    # ------------------------------------------------------------------ API
    def calculate_cost(self, train_data, reduction: str = "mean"):
        """Compute Boundary Shrink loss for one step.

        train_data must contain both "Retain" and "Forget" keys.
        """
        x_r, y_r = train_data["Retain"]
        x_f, y_f = train_data["Forget"]
        x_r = x_r.to(self.device)
        y_r = y_r.to(self.device)
        x_f = x_f.to(self.device)
        y_f = y_f.to(self.device)

        # --- Boundary shrink: re-target forget samples to nearest class ---
        nearest_labels = self._nearest_class(x_f, y_f)
        shrink_loss = nn.CrossEntropyLoss(reduction="none")(self.rmodel(x_f), nearest_labels)

        # --- Retain: standard cross-entropy ---
        retain_loss = classification_loss(self.rmodel, x_r, y_r, self.device, reduction="none")

        cost = (self.forget_lambda * shrink_loss.mean()
                + self.retain_lambda * retain_loss.mean())

        self.add_record_item("ShrinkLoss", shrink_loss.mean().item())
        self.add_record_item("RetainLoss", retain_loss.mean().item())
        self.add_record_item("Cost", cost.item())
        return cost

    # ------------------------------------------------------------ helpers
    @torch.no_grad()
    def _nearest_class(
        self,
        x_f: torch.Tensor,
        y_f: torch.Tensor,
    ) -> torch.Tensor:
        """Find the nearest non-forget class for each forget sample.

        For each sample, the nearest class c* is the class (other than the
        forget label) with the highest logit score under the current model.
        This corresponds to the boundary being closest in logit space.
        """
        self.rmodel.eval()
        logits = self.rmodel(x_f)   # (B, C)
        self.rmodel.train()

        if self.omit_label is not None:
            # Mask out the forget class so argmax finds the next best
            logits[:, self.omit_label] = float("-inf")
        else:
            # Mask out each sample's own true label
            for i, label in enumerate(y_f):
                logits[i, label.item()] = float("-inf")

        return logits.argmax(dim=1)
