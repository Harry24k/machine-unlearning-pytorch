"""SFR-on: Saliency Forgetting in the Remain-preserving manifold, online.

Reference:
    Huang, Cheng, Zheng, Wang, He, Li & Huang,
    "Unified Gradient-Based Machine Unlearning with Remain Geometry
    Enhancement", NeurIPS 2024 (Spotlight).  arXiv:2409.19732
    Official code: github.com/K1nght/Unified-Unlearning-w-Remain-Geometry

Algorithm overview:
    The paper decomposes the steepest-descent direction for approximate MU
    (minimizing output KL to exact MU in a parameter neighborhood) into
    three parts:

        weighted forget gradient ASCENT
      + retain gradient DESCENT
      * a weight SALIENCY matrix

    Euclidean steepest descent ignores the geometry of the output
    probability space, so SFR-on embeds the update in a manifold shaped by
    the remaining data, implicitly picking up second-order (Hessian)
    information from the retain set. Rather than forming a Hessian, it uses
    a fast/slow two-timescale parameter update: fast weights take k
    unlearning steps, then the slow weights are pulled a fraction alpha of
    the way toward them (a Lookahead-style outer step). The saliency mask
    is recomputed online from the current forget gradient instead of being
    frozen up front, which is what "on" (online) refers to.

Notes on this implementation:
    Reimplemented from the paper description, not a port of the official
    code. The online saliency mask and the fast/slow outer step follow the
    paper; exact hyperparameter schedules may differ.
"""

import torch

from .unlearner import Unlearner
from ...core.losses import classification_loss


class SFRon(Unlearner):
    r"""Saliency Forgetting in the Remain-preserving manifold, online.

    Arguments:
        rmodel (RobModel): model to unlearn.
        forget_lambda (float): weight on the forget ascent term.
        retain_lambda (float): weight on the retain descent term.
        saliency_ratio (float): fraction of weights kept in the online
            saliency mask (top-k by |forget gradient|). Default 0.5.
        slow_alpha (float): outer-step interpolation toward the fast
            weights. 1.0 disables the slow update. Default 0.5.
        slow_every (int): number of fast steps k between outer steps.
    """

    def __init__(
        self,
        rmodel,
        forget_lambda: float = 1.0,
        retain_lambda: float = 1.0,
        saliency_ratio: float = 0.5,
        slow_alpha: float = 0.5,
        slow_every: int = 5,
    ):
        super().__init__(rmodel)
        self.forget_lambda = forget_lambda
        self.retain_lambda = retain_lambda
        self.saliency_ratio = saliency_ratio
        self.slow_alpha = slow_alpha
        self.slow_every = slow_every
        self._slow_weights = None
        self._step = 0

    def state_dict(self):
        """Extra state so ``fit(refit=True)`` resumes mid-schedule."""
        return {"step": self._step, "slow_weights": self._slow_weights}

    def load_state_dict(self, state):
        self._step = state.get("step", 0)
        self._slow_weights = state.get("slow_weights")

    # ------------------------------------------------------------ saliency
    @torch.no_grad()
    def _online_saliency_mask(self):
        """Top-k mask over |grad|, recomputed from the gradients now in .grad."""
        grads = [p.grad.abs().view(-1) for p in self.rmodel.parameters()
                 if p.grad is not None]
        if not grads:
            return None
        flat = torch.cat(grads)
        k = max(1, int(self.saliency_ratio * flat.numel()))
        thresh = flat.topk(k).values.min()
        return thresh

    def calculate_cost(self, train_data, reduction: str = "mean"):
        """Retain descent minus weighted forget ascent."""
        if not isinstance(train_data, dict):
            raise TypeError(
                "%s needs Retain and Forget batches together. Wrap your loaders "
                "with MergedLoaders({'Retain': ..., 'Forget': ...}) and pass that "
                "to fit()." % type(self).__name__
            )
        x_r, y_r = train_data["Retain"]
        x_f, y_f = train_data["Forget"]

        retain_loss = classification_loss(self.rmodel, x_r, y_r, self.device)
        forget_loss = classification_loss(self.rmodel, x_f, y_f, self.device)

        # Ascent on forget = descent on its negation.
        cost = self.retain_lambda * retain_loss - self.forget_lambda * forget_loss

        self.add_record_item("RetainLoss", retain_loss.item())
        self.add_record_item("ForgetLoss", forget_loss.item())
        self.add_record_item("Cost", cost.item())
        return cost

    # -------------------------------------------------------- weight update
    def _update_weight(self, *inputs):
        """Masked fast step, plus a periodic slow (remain-preserving) step."""
        if self._slow_weights is None:
            self._slow_weights = [p.detach().clone()
                                  for p in self.rmodel.parameters()]

        cost = self.calculate_cost(*inputs)
        self.optimizer.zero_grad()
        cost.backward()

        # Online saliency: zero the gradient of non-salient weights.
        thresh = self._online_saliency_mask()
        if thresh is not None:
            with torch.no_grad():
                for p in self.rmodel.parameters():
                    if p.grad is not None:
                        p.grad.mul_((p.grad.abs() >= thresh).to(p.grad.dtype))

        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.rmodel.parameters(), self.clip_grad_norm
            )
        self.optimizer.step()
        self._step += 1

        # Slow outer step: pull slow weights toward fast weights, then reset.
        if self.slow_alpha < 1.0 and self._step % self.slow_every == 0:
            with torch.no_grad():
                for slow, p in zip(self._slow_weights, self.rmodel.parameters()):
                    slow.add_(self.slow_alpha * (p.detach() - slow))
                    p.copy_(slow)
