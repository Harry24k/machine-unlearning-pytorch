"""FLAT (Wang et al., ICLR 2025): f-divergence contrastive loss between the probability of the alternate
answer and of the forget answer (loss_type="cl"); default divergence Total-Variation, no retain term."""
import math

import torch
import torch.nn.functional as F

from .base import SeqUnlearner


class FLAT(SeqUnlearner):
    """activation(-mean p_good) - conjugate(-mean p_unlearn)."""
    needs_alt = True
    DIVS = {
        "KL": (lambda x: -torch.mean(x), lambda x: -torch.mean(torch.exp(x - 1.0))),
        "Reverse-KL": (lambda x: -torch.mean(-torch.exp(x)), lambda x: -torch.mean(-1.0 - x)),
        "Jeffrey": (lambda x: -torch.mean(x), lambda x: -torch.mean(x + x * x / 4.0 + x * x * x / 16.0)),
        "Squared-Hellinger": (lambda x: -torch.mean(1.0 - torch.exp(x)), lambda x: -torch.mean((1.0 - torch.exp(x)) / torch.exp(x))),
        "Pearson": (lambda x: -torch.mean(x), lambda x: -torch.mean(x * x / 4.0 + x)),
        "Neyman": (lambda x: -torch.mean(1.0 - torch.exp(x)), lambda x: -torch.mean(2.0 - 2.0 * torch.sqrt(1.0 - x))),
        "Jenson-Shannon": (lambda x: -torch.mean(-torch.log(1.0 + torch.exp(-x))) - math.log(2.0),
                           lambda x: -torch.mean(x + torch.log(1.0 + torch.exp(-x))) + math.log(2.0)),
        "Total-Variation": (lambda x: -torch.mean(torch.tanh(x) / 2.0), lambda x: -torch.mean(torch.tanh(x) / 2.0)),
    }

    def __init__(self, rmodel, div="Total-Variation", **kw):
        kw.setdefault("alpha", 0.0)
        kw.setdefault("retain_loss_type", "none")
        super().__init__(rmodel, **kw)
        if div not in self.DIVS:
            raise ValueError(f"div must be one of {list(self.DIVS)}")
        self.div = div

    def _prob_loss(self, batch, prefix):
        logits = self._logits(batch, prefix=prefix)
        sl = logits[:, :-1].float()
        labels = batch["alt_labels"] if prefix == "alt_" else self._labels(batch, "Forget")
        lab = labels[:, 1:].to(sl.device)
        probs = F.softmax(sl, -1)
        return F.nll_loss(probs.transpose(1, 2), lab, ignore_index=-100, reduction="none").mean()

    def forget_loss(self, batch):
        if "alt_input_ids" not in batch:
            raise ValueError("FLAT needs alternate answers (SeqCollator(alt_text=...) or SeqSample.alt_answers)")
        loss_sum_unlearn = self._prob_loss(batch, "")
        loss_sum_good = self._prob_loss(batch, "alt_")
        activation, conjugate = self.DIVS[self.div]
        return activation(-loss_sum_good) - conjugate(-loss_sum_unlearn)
