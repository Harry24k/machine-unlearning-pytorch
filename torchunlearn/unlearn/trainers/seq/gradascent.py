"""GradAscent: loss = -CE on the answer tokens (Jang et al., 2023; open-unlearning GradAscent)."""
from .base import SeqUnlearner


class GradAscent(SeqUnlearner):
    """Gradient ascent on the answer tokens: loss = -CE."""

    def __init__(self, rmodel, **kw):
        kw.setdefault("alpha", 0.0)
        kw.setdefault("retain_loss_type", "none")
        super().__init__(rmodel, **kw)

    def forget_loss(self, batch):
        ce, *_ = self.forget_nll(batch)
        return -ce
