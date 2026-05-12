"""SCRUB: Towards Adversarial Robustness of Machine Unlearning.

Reference:
    Kurmanji et al., "Towards Adversarial Robustness of Machine Unlearning",
    NeurIPS 2023.

Algorithm overview:
    SCRUB alternates two objectives each epoch:

    1. MAXIMIZATION step (ascent on forget set):
       Maximize KL-divergence between the student (unlearned) model and a
       frozen teacher (original model) on forget samples. This pushes the
       student away from remembering the forget data.

    2. MINIMIZATION step (descent on retain set):
       Minimize KL-divergence between student and teacher on retain samples
       PLUS standard cross-entropy on retain samples. This preserves utility
       and keeps the student close to the original on retained knowledge.

    The interplay of these two objectives produces a model that forgets Df
    while staying close to the retrained gold standard on Dr.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unlearner import Unlearner


class SCRUB(Unlearner):
    """SCRUB unlearning via alternating KL-maximization / KL-minimization.

    Arguments:
        rmodel: model to unlearn (RobModel).
        retain_lambda (float): weight on the retain KL + CE terms (default 1.0).
        forget_lambda (float): weight on the forget KL-ascent term (default 1.0).
        sgda_smoothing (float): label smoothing on retain CE (default 0.0).
        msteps (int): number of maximization (forget-ascent) steps per epoch.
    """

    def __init__(
        self,
        rmodel,
        retain_lambda: float = 1.0,
        forget_lambda: float = 1.0,
        sgda_smoothing: float = 0.0,
        msteps: int = 1,
    ):
        super().__init__(rmodel)
        self.retain_lambda = retain_lambda
        self.forget_lambda = forget_lambda
        self.sgda_smoothing = sgda_smoothing
        self.msteps = msteps
        # Frozen teacher copy — set during the first calculate_cost call
        self._teacher = None

    # ------------------------------------------------------------------ API
    def calculate_cost(self, train_data, reduction: str = "mean"):
        """Compute SCRUB loss for one training step.

        train_data must contain both "Retain" and "Forget" keys.

        The teacher snapshot is created lazily on the first call so that
        the model has already been moved to the correct device.
        """
        if self._teacher is None:
            self._teacher = self._make_frozen_teacher()

        x_r, y_r = train_data["Retain"]
        x_f, y_f = train_data["Forget"]

        x_r = x_r.to(self.device)
        y_r = y_r.to(self.device)
        x_f = x_f.to(self.device)

        # --- Retain loss: CE + KL towards teacher ---
        logits_r = self.rmodel(x_r)
        ce_loss = nn.CrossEntropyLoss(label_smoothing=self.sgda_smoothing)(logits_r, y_r)

        with torch.no_grad():
            teacher_logits_r = self._teacher(x_r)
        kl_retain = self._kl_div(logits_r, teacher_logits_r)

        # --- Forget loss: negative KL (ascent) towards teacher ---
        logits_f = self.rmodel(x_f)
        with torch.no_grad():
            teacher_logits_f = self._teacher(x_f)
        kl_forget = self._kl_div(logits_f, teacher_logits_f)

        cost = (self.retain_lambda * (ce_loss + kl_retain)
                - self.forget_lambda * kl_forget)

        self.add_record_item("CELoss(R)", ce_loss.item())
        self.add_record_item("KL(R)", kl_retain.item())
        self.add_record_item("KL(F)", kl_forget.item())
        self.add_record_item("Cost", cost.item())
        return cost

    # ------------------------------------------------------------ helpers
    def _make_frozen_teacher(self) -> nn.Module:
        """Deep-copy the current model and freeze all parameters."""
        teacher = copy.deepcopy(self.rmodel)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    @staticmethod
    def _kl_div(student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                temperature: float = 1.0) -> torch.Tensor:
        """KL( teacher || student ) averaged over the batch."""
        T = temperature
        p = F.softmax(teacher_logits / T, dim=1)
        log_q = F.log_softmax(student_logits / T, dim=1)
        return F.kl_div(log_q, p, reduction="batchmean") * (T ** 2)
