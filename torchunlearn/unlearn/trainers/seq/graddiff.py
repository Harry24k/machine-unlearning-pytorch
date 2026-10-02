"""GradDiff: gamma * (-CE_forget) + alpha * retain (Liu et al., 2022; open-unlearning GradDiff)."""
from .base import SeqUnlearner


class GradDiff(SeqUnlearner):
    """-CE(forget) + alpha * retain (NLL or KL vs reference)."""

    def forget_loss(self, batch):
        ce, *_ = self.forget_nll(batch)
        return -ce
