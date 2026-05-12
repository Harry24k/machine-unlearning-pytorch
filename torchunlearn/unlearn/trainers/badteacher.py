"""Bad Teacher: Unlearning via Competent-Teacher / Bad-Teacher Distillation.

Reference:
    Chundawat et al., "Can Bad Teaching Induce Forgetting? Unlearning in Deep
    Networks Using an Incompetent Teacher", AAAI 2023.

Algorithm overview:
    Two teacher models are used:
      - Competent Teacher (CT): the original trained model (frozen). Guides
        the student on RETAIN samples to preserve accuracy.
      - Bad Teacher (BT): a randomly initialised model of the same architecture
        (frozen). Guides the student on FORGET samples to produce uniformly
        random, uninformative predictions.

    For each batch the student minimises:
        L = KL(student(x_r) || CT(x_r))   [retain: stay close to competent teacher]
          + KL(student(x_f) || BT(x_f))   [forget: match bad/random teacher]

    After training the student has forgotten Df (its outputs on forget samples
    resemble random noise) while retaining competent performance on Dr.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unlearner import Unlearner


class BadTeacher(Unlearner):
    """Unlearning via competent-teacher / bad-teacher knowledge distillation.

    Arguments:
        rmodel: model to unlearn (RobModel).
        retain_lambda (float): weight on the retain KL loss (default 1.0).
        forget_lambda (float): weight on the forget KL loss (default 1.0).
        temperature (float): softmax temperature for KL distillation (default 1.0).
    """

    def __init__(
        self,
        rmodel,
        retain_lambda: float = 1.0,
        forget_lambda: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__(rmodel)
        self.retain_lambda = retain_lambda
        self.forget_lambda = forget_lambda
        self.temperature = temperature
        # Both teachers are created lazily on first use
        self._competent_teacher = None
        self._bad_teacher = None

    # ------------------------------------------------------------------ API
    def calculate_cost(self, train_data, reduction: str = "mean"):
        """Compute Bad-Teacher distillation loss for one step.

        train_data must contain both "Retain" and "Forget" keys.
        """
        if self._competent_teacher is None:
            self._competent_teacher = self._make_competent_teacher()
        if self._bad_teacher is None:
            self._bad_teacher = self._make_bad_teacher()

        x_r, y_r = train_data["Retain"]
        x_f, _ = train_data["Forget"]
        x_r = x_r.to(self.device)
        x_f = x_f.to(self.device)

        # --- Retain: match competent teacher ---
        logits_r = self.rmodel(x_r)
        with torch.no_grad():
            ct_logits_r = self._competent_teacher(x_r)
        kl_retain = self._kl_div(logits_r, ct_logits_r)

        # --- Forget: match bad (random) teacher ---
        logits_f = self.rmodel(x_f)
        with torch.no_grad():
            bt_logits_f = self._bad_teacher(x_f)
        kl_forget = self._kl_div(logits_f, bt_logits_f)

        cost = self.retain_lambda * kl_retain + self.forget_lambda * kl_forget

        self.add_record_item("KL(R)", kl_retain.item())
        self.add_record_item("KL(F)", kl_forget.item())
        self.add_record_item("Cost", cost.item())
        return cost

    # ------------------------------------------------------------ helpers
    def _make_competent_teacher(self) -> nn.Module:
        """Frozen deep copy of the original rmodel = competent teacher."""
        teacher = copy.deepcopy(self.rmodel)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    def _make_bad_teacher(self) -> nn.Module:
        """Randomly re-initialised copy of the rmodel = bad teacher."""
        bad = copy.deepcopy(self.rmodel)
        # Reset all parameters to random (Xavier / Kaiming defaults)
        for module in bad.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        bad.eval()
        for p in bad.parameters():
            p.requires_grad_(False)
        return bad

    def _kl_div(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL( teacher || student ) averaged over the batch."""
        T = self.temperature
        p = F.softmax(teacher_logits / T, dim=1)
        log_q = F.log_softmax(student_logits / T, dim=1)
        return F.kl_div(log_q, p, reduction="batchmean") * (T ** 2)
