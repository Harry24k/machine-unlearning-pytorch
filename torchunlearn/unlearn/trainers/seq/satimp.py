"""SatImp (Wang et al., 2025): w = p^beta1 * (1-p)^beta2 (detached); loss = -(w * CE).mean(); gamma=0.1."""
from .base import SeqUnlearner


class SatImp(SeqUnlearner):
    """Saturation x Importance weighted gradient ascent."""

    def __init__(self, rmodel, beta1=5.0, beta2=1.0, **kw):
        kw.setdefault("gamma", 0.1)
        super().__init__(rmodel, **kw)
        self.beta1, self.beta2 = float(beta1), float(beta2)

    def forget_loss(self, batch):
        _, logits, nll, mask = self.forget_nll(batch)
        p = (-nll).exp().detach()
        w = (p ** self.beta1) * ((1 - p) ** self.beta2)
        return -(w * nll)[mask].mean()
