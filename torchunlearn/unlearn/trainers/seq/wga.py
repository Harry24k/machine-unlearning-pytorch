"""WGA (Wang et al., 2025): weighted gradient ascent, w = p^beta (detached); loss = -(w * CE).mean()."""
from .base import SeqUnlearner


class WGA(SeqUnlearner):
    """Weighted Gradient Ascent."""

    def __init__(self, rmodel, beta=1.0, **kw):
        super().__init__(rmodel, **kw)
        self.beta = float(beta)

    def forget_loss(self, batch):
        _, logits, nll, mask = self.forget_nll(batch)
        w = ((-nll).exp().detach()) ** self.beta
        return -(w * nll)[mask].mean()
