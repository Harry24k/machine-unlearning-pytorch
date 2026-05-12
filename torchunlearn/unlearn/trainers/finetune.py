from .unlearner import Unlearner
from ...core.losses import classification_loss


class Finetune(Unlearner):
    r"""Baseline unlearning method: fine-tune on the Retain split only.

    The forget set is never seen during this step; the model simply
    continues training on the data we want to keep.

    Attributes:
        rmodel: the robust model being unlearned.
        device: device where rmodel lives.
        optimizer / scheduler / curr_epoch / curr_iter: see
            :class:`Trainer`. All are auto-updated by the engine.

    Arguments:
        rmodel (nn.Module): model to unlearn.
    """

    def __init__(self, rmodel):
        super().__init__(rmodel)

    def calculate_cost(self, train_data, reduction="mean"):
        """Cross-entropy on the Retain batch. Overrides :meth:`Trainer.calculate_cost`."""
        x, y = train_data["Retain"]
        cost = classification_loss(self.rmodel, x, y, self.device, reduction)
        # Engine-facing record (kept here for back-compat with existing logs)
        scalar_cost = cost.item() if reduction == "mean" else cost.mean().item()
        self.add_record_item("Cost", scalar_cost)
        return cost
