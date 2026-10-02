"""NPO (Zhang et al., 2024): -2/beta * logsigmoid(beta * (NLL_seq - NLL_seq_ref)).mean() + alpha * retain."""
import torch
import torch.nn.functional as F

from .base import SeqUnlearner


class NPO(SeqUnlearner):
    """Negative Preference Optimization on the sequence-level log-ratio vs the reference model."""
    ref_needs = ("forget_seq_nll",)

    def __init__(self, rmodel, beta=0.1, **kw):
        super().__init__(rmodel, **kw)
        self.beta = float(beta)

    def forget_loss(self, batch):
        _, logits, nll, mask = self.forget_nll(batch)
        seq = nll.sum(-1)
        if self.ref_mode == "model":
            with torch.no_grad():
                rn, _ = self._token_nll(self._logits(batch, model=self._ref()), self._labels(batch, "Forget"))
                ref = rn.sum(-1).to(seq.device)
        else:
            ref = torch.tensor(self._cached("Forget", batch, "seq_nll"), device=seq.device)
        lose_log_ratio = -(seq - ref)
        return -2 / self.beta * F.logsigmoid(self.beta * (0.0 - lose_log_ratio)).mean()
