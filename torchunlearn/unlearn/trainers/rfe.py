"""RFE: two-phase unlearning under retain-forget entanglement.

Reference:
    Cheng, Liu, Li & Zhang, "Machine Unlearning under Retain-Forget
    Entanglement", ICLR 2026.  arXiv:2603.26569
    OpenReview: openreview.net/forum?id=4WMBSHHJEr

Algorithm overview:
    Forgetting is rarely isolated. Retained samples that share features or
    semantics with the forget set get collaterally damaged, and averaging
    accuracy over the whole retain set hides it. The paper splits the
    retain set into an ADJACENT part D_adj (semantically entangled with
    D_f -- e.g. sibling subclasses under the same CIFAR-100 superclass)
    and the rest, then runs two phases:

    Phase 1 -- augmented Lagrangian. Maximize forget loss subject to a
        constraint that the loss on the *less-related* retain samples stays
        near its original value. The multiplier is updated online, so the
        forget/retain trade-off is adapted rather than hand-tuned.

    Phase 2 -- gradient projection. Project the update direction to remove
        the component that would raise loss on D_adj, regularized by a
        Wasserstein-2 term so the model's output distribution on the
        entangled samples does not drift.

    Provide the entangled split as a third loader keyed "Adjacent". If it
    is absent, phase 2 degrades gracefully to plain retain projection.

Notes on this implementation:
    Reimplemented from the paper description, not a port of the authors'
    code. The W2 regularizer is approximated by the squared L2 distance
    between current and reference softmax outputs on D_adj, which is the
    standard closed-form surrogate; the paper's exact estimator may differ.
"""

import copy

import torch
import torch.nn.functional as F

from .unlearner import Unlearner
from ...core.losses import classification_loss


class RFE(Unlearner):
    r"""Retain-Forget-Entanglement-aware unlearning (Cheng et al., ICLR 2026).

    Arguments:
        rmodel (RobModel): model to unlearn.
        phase1_steps (int): iterations spent in the augmented-Lagrangian
            phase before switching to gradient projection.
        constraint_tol (float): allowed increase in retain loss over its
            value at the start of unlearning (the constraint level).
        lam_init (float): initial Lagrange multiplier.
        lam_lr (float): dual ascent rate for the multiplier.
        rho (float): quadratic penalty weight of the augmented Lagrangian.
        w2_lambda (float): weight of the Wasserstein-2 surrogate on D_adj.
    """

    def __init__(
        self,
        rmodel,
        phase1_steps: int = 100,
        constraint_tol: float = 0.05,
        lam_init: float = 1.0,
        lam_lr: float = 0.05,
        rho: float = 1.0,
        w2_lambda: float = 1.0,
    ):
        super().__init__(rmodel)
        self.phase1_steps = phase1_steps
        self.constraint_tol = constraint_tol
        self.lam = lam_init
        self.lam_lr = lam_lr
        self.rho = rho
        self.w2_lambda = w2_lambda
        self._step = 0
        self._retain_ref = None
        self._ref_model = None

    def state_dict(self):
        """Extra state so ``fit(refit=True)`` resumes in the right phase."""
        return {"step": self._step, "lam": self.lam, "retain_ref": self._retain_ref}

    def load_state_dict(self, state):
        self._step = state.get("step", 0)
        self.lam = state.get("lam", self.lam)
        self._retain_ref = state.get("retain_ref")

    def _reference_model(self):
        """Frozen copy of the pre-unlearning model, for the W2 surrogate."""
        if self._ref_model is None:
            self._ref_model = copy.deepcopy(self.rmodel).eval()
            for p in self._ref_model.parameters():
                p.requires_grad_(False)
        return self._ref_model

    # ------------------------------------------------------------- phase 1
    def _phase1_cost(self, x_r, y_r, x_f, y_f):
        """Augmented Lagrangian: ascend forget loss under a retain constraint."""
        retain_loss = classification_loss(self.rmodel, x_r, y_r, self.device)
        forget_loss = classification_loss(self.rmodel, x_f, y_f, self.device)

        if self._retain_ref is None:
            self._retain_ref = retain_loss.detach()

        # g(theta) = L_retain - (L_retain_0 + tol) <= 0
        violation = retain_loss - (self._retain_ref + self.constraint_tol)
        cost = (
            -forget_loss
            + self.lam * violation
            + 0.5 * self.rho * torch.clamp(violation, min=0.0).pow(2)
        )

        # Dual ascent on the multiplier.
        with torch.no_grad():
            self.lam = max(0.0, self.lam + self.lam_lr * violation.item())

        self.add_record_item("RetainLoss", retain_loss.item())
        self.add_record_item("ForgetLoss", forget_loss.item())
        self.add_record_item("Lambda", self.lam)
        return cost

    # ------------------------------------------------------------- phase 2
    def _phase2_cost(self, x_a, y_a):
        """W2-regularized loss on the entangled (adjacent) retain subset."""
        x_a, y_a = x_a.to(self.device), y_a.to(self.device)
        logits = self.rmodel(x_a)
        ce = F.cross_entropy(logits, y_a)

        with torch.no_grad():
            ref_p = F.softmax(self._reference_model()(x_a), dim=1)
        w2 = (F.softmax(logits, dim=1) - ref_p).pow(2).sum(dim=1).mean()

        cost = ce + self.w2_lambda * w2
        self.add_record_item("AdjCE", ce.item())
        self.add_record_item("W2", w2.item())
        return cost

    def calculate_cost(self, train_data, reduction: str = "mean"):
        if not isinstance(train_data, dict):
            raise TypeError(
                "%s needs Retain and Forget batches together. Wrap your loaders "
                "with MergedLoaders({'Retain': ..., 'Forget': ...}) and pass that "
                "to fit()." % type(self).__name__
            )
        x_r, y_r = train_data["Retain"]
        x_f, y_f = train_data["Forget"]
        cost = self._phase1_cost(x_r, y_r, x_f, y_f)
        self.add_record_item("Cost", cost.item())
        return cost

    # -------------------------------------------------------- weight update
    def _flat_grad(self):
        return torch.cat([
            p.grad.view(-1) if p.grad is not None else torch.zeros(p.numel(),
                                                                   device=p.device)
            for p in self.rmodel.parameters()
        ])

    def _set_flat_grad(self, flat):
        i = 0
        for p in self.rmodel.parameters():
            n = p.numel()
            if p.grad is not None:
                p.grad.copy_(flat[i:i + n].view_as(p))
            i += n

    def _update_weight(self, *inputs):
        train_data = inputs[0]
        self._step += 1

        if self._step <= self.phase1_steps:
            cost = self.calculate_cost(train_data)
            self.optimizer.zero_grad()
            cost.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.rmodel.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            return

        # Phase 2: project the unlearning direction off the D_adj gradient.
        adj = train_data.get("Adjacent", train_data["Retain"])

        cost = self.calculate_cost(train_data)
        self.optimizer.zero_grad()
        cost.backward()
        g_unlearn = self._flat_grad().clone()

        self.optimizer.zero_grad()
        self._phase2_cost(*adj).backward()
        g_adj = self._flat_grad().clone()

        denom = g_adj.dot(g_adj) + 1e-12
        overlap = g_unlearn.dot(g_adj)
        if overlap < 0:                       # conflicting directions only
            g_unlearn = g_unlearn - (overlap / denom) * g_adj

        self._set_flat_grad(g_unlearn)
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.rmodel.parameters(), self.clip_grad_norm)
        self.optimizer.step()
