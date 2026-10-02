"""DPO unlearning (Rafailov et al., 2023 / Maini et al., 2024 IdkDPO): an alternate answer is preferred over
the forget answer.  Alternate = SeqSample.alt_answers[k] or SeqCollator(alt_text="[REDACTED]")."""
import torch
import torch.nn.functional as F

from .base import SeqUnlearner


class DPO(SeqUnlearner):
    """-2/beta * logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean() + alpha * retain."""
    needs_alt = True
    ref_needs = ("forget_seq_nll", "alt_seq_nll")

    def __init__(self, rmodel, beta=0.1, **kw):
        super().__init__(rmodel, **kw)
        self.beta = float(beta)

    def _seq(self, batch, prefix, model=None):
        logits = self._logits(batch, prefix=prefix, model=model)
        labels = batch["alt_labels"] if prefix == "alt_" else self._labels(batch, "Forget")
        nll, _ = self._token_nll(logits, labels)
        return nll.sum(-1)

    def _refs(self, batch, lose):
        if self.ref_mode == "model":
            with torch.no_grad():
                lose_ref = self._seq(batch, "", self._ref()).to(lose.device)
                win_ref = self._seq(batch, "alt_", self._ref()).to(lose.device)
        else:
            lose_ref = torch.tensor(self._cached("Forget", batch, "seq_nll"), device=lose.device)
            win_ref = torch.tensor(self._cached("Forget", batch, "alt_seq_nll"), device=lose.device)
        return lose_ref, win_ref

    def forget_loss(self, batch):
        if "alt_input_ids" not in batch:
            raise ValueError("DPO needs alternate answers (SeqCollator(alt_text=...) or SeqSample.alt_answers)")
        lose = self._seq(batch, "")
        win = self._seq(batch, "alt_")
        lose_ref, win_ref = self._refs(batch, lose)
        win_log_ratio = -(win - win_ref)
        lose_log_ratio = -(lose - lose_ref)
        return -2 / self.beta * F.logsigmoid(self.beta * (win_log_ratio - lose_log_ratio)).mean()
