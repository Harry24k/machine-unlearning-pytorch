"""UNDIAL (Dong et al., 2025): self-distillation with the teacher logit of the true token reduced by beta;
loss = CE(student, softmax(teacher - beta * onehot)) on answer positions (beta=10, alpha=0)."""
import torch
import torch.nn.functional as F

from .base import SeqUnlearner


class UNDIAL(SeqUnlearner):
    """Self-distillation with a penalised teacher logit on the true token."""
    ref_needs = ("forget_tok_logits",)

    def __init__(self, rmodel, beta=10.0, **kw):
        kw.setdefault("alpha", 0.0)
        super().__init__(rmodel, **kw)
        self.beta = float(beta)

    def forget_loss(self, batch):
        logits = self._logits(batch)
        sl = logits[:, :-1].float()
        lab = self._labels(batch, "Forget")[:, 1:].to(sl.device)
        m = lab != -100
        if self.ref_mode == "model":
            with torch.no_grad():
                t = self._logits(batch, model=self._ref())[:, :-1].float().to(sl.device)[m]
        else:
            t = torch.cat(self._cached("Forget", batch, "tok_logits")).to(sl.device).float()
        onehot = F.one_hot(lab[m], sl.shape[-1]).float()
        soft = F.softmax(t - onehot * self.beta, -1)
        return F.cross_entropy(sl[m], soft)
