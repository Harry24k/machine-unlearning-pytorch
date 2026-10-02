"""SimNPO (Fan et al., 2024): reference-free NPO on length-normalised NLL (beta=4.5, delta=0, gamma=0.125)."""
import torch.nn.functional as F

from .base import SeqUnlearner


class SimNPO(SeqUnlearner):
    """-2/beta * logsigmoid(beta * (NLL_seq / n_tok - delta)).mean()."""

    def __init__(self, rmodel, beta=4.5, delta=0.0, **kw):
        kw.setdefault("gamma", 0.125)
        super().__init__(rmodel, **kw)
        self.beta, self.delta = float(beta), float(delta)

    def forget_loss(self, batch):
        _, logits, nll, mask = self.forget_nll(batch)
        x = nll.sum(-1) / mask.sum(-1).clamp(min=1) - self.delta
        return -F.logsigmoid(self.beta * x).mean() * 2 / self.beta
