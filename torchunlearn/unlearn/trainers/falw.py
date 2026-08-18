"""FaLW: Forgetting-aware Loss Reweighting for Long-tailed Unlearning.

Reference:
    Yu, Zhao, Wang, Wang, Cao, Wang & Wang,
    "FaLW: A Forgetting-aware Loss Reweighting for Long-tailed Unlearning",
    arXiv:2601.18650, 2026.

Algorithm overview:
    FaLW targets forget sets whose class distribution is long-tailed, where
    holistic methods under-forget head classes and over-forget tail classes
    (Heterogeneous / Skewed Unlearning Deviation). It is a plug-and-play,
    instance-wise loss reweighting on top of random-label unlearning
    (Algorithm 1 of the paper is the RL-based version, reproduced here).

    Per optimization step:
    1. TARGET DISTRIBUTION ESTIMATION:
       Run the current model on a held-out validation set ("unseen" data)
       and estimate a per-class Gaussian N(mu_c, sigma_c^2) over the
       predictive probability of the true class. The validation set proxies
       how the model behaves on data it was never trained on -- the desired
       endpoint of unlearning ("seen" -> "unseen").
    2. FORGETTING-AWARE WEIGHT (Eq. 5/7):
       For each forget sample with true class c and current true-class
       probability p_i, compute the z-score z_i = (p_i - mu_c) / sigma_c and

           w_i = 1 + sign(z_i) * tanh(|z_i|) ** (1 / B_i)

       Under-forgotten samples (p_i above mu_c) get w_i -> 2 (push harder);
       over-forgotten samples (p_i below mu_c) get w_i -> 0 (stop pushing).
    3. BALANCE FACTOR (Eq. 6):
       B_i = (N_f / (C * N_{f,k})) ** tau, precomputed per class from the
       forget-set label histogram. Tail classes get a larger B_i, making
       w_i react more sharply to their deviation (counters the skew).
    4. LOSS: cross-entropy of forget samples against i.i.d. random labels,
       weighted by w_i (detached), plus plain cross-entropy on the retain
       batch; averaged jointly over all samples in the merged batch.

Notes on this implementation:
    Ported against both the paper (Eq. 3, 5-7; Algorithm 1, Appendix C.1)
    and the authors' supplementary code. The supplementary code re-estimates
    (mu_c, sigma_c) on the full validation set at EVERY step, exactly as in
    Algorithm 1; `estimate_every` (default 1) keeps that behavior but lets
    you amortize the cost, which is a deviation from the paper when > 1.
    In the supplementary code the random labels are drawn once when the
    relabeled forget set is built; here they are redrawn per batch, matching
    this repository's RandomLabel / SalUn convention.
"""

import torch
import torch.nn as nn

from .unlearner import Unlearner


class FaLW(Unlearner):
    r"""Forgetting-aware loss reweighting on random-label unlearning.

    Arguments:
        rmodel (RobModel): model to unlearn (must expose ``n_classes``).
        tau (float): exponent of the balance factor B_c (Eq. 6). The paper
            tunes it in [0.1, 0.2]; larger values react more aggressively
            to tail-class deviation. Default 0.15.
        forget_lambda (float): weight on the (reweighted) forget loss
            (alpha in Eq. 3). Default 1.0.
        retain_lambda (float): weight on the retain loss (beta in Eq. 3).
            Default 1.0.
        estimate_every (int): re-estimate (mu_c, sigma_c) on the validation
            set every this many optimization steps. 1 reproduces Algorithm 1
            exactly (and the supplementary code); larger values trade
            fidelity for speed. Default 1.
        sigma_min (float): lower clamp on sigma_c to avoid division by zero
            for classes whose validation predictions are (near-)constant.
            Default 1e-8.

    Usage::

        trainer = FaLW(rmodel, tau=0.15)
        trainer.prepare(forget_loader, val_loader)
        trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9)")
        trainer.fit(MergedLoaders({"Retain": retain_loader,
                                   "Forget": forget_loader}), n_epochs=10)
    """

    def __init__(
        self,
        rmodel,
        tau: float = 0.15,
        forget_lambda: float = 1.0,
        retain_lambda: float = 1.0,
        estimate_every: int = 1,
        sigma_min: float = 1e-8,
    ):
        super().__init__(rmodel)
        if estimate_every < 1:
            raise ValueError("`estimate_every` must be >= 1.")
        self.tau = tau
        self.forget_lambda = forget_lambda
        self.retain_lambda = retain_lambda
        self.estimate_every = estimate_every
        self.sigma_min = sigma_min

        self._val_loader = None
        self._balance = None      # (C,) tensor, B_c per class (Eq. 6)
        self._mu = None           # (C,) tensor, mu_c
        self._sigma = None        # (C,) tensor, sigma_c
        self._valid_class = None  # (C,) bool, class seen in val set
        self._step = 0

    # ------------------------------------------------------------------ API
    def prepare(self, forget_loader, val_loader) -> None:
        """Precompute the balance factor and register the validation loader.

        Call this ONCE before ``fit()``.

        Arguments:
            forget_loader: DataLoader over the forget set. Only the labels
                are read, to build the class histogram N_{f,k} for Eq. 6.
            val_loader: DataLoader over held-out validation data (data the
                model has never trained on), used to estimate the per-class
                target distribution (mu_c, sigma_c) during unlearning.
        """
        n_classes = int(self.rmodel.n_classes)
        counts = torch.zeros(n_classes)
        for _, y in forget_loader:
            counts += torch.bincount(y.view(-1).cpu(), minlength=n_classes).float()

        n_forget = counts.sum()
        if n_forget == 0:
            raise ValueError("`forget_loader` yielded no samples.")

        # Eq. 6: B_c = (N_f / (C * N_{f,k})) ** tau. Classes absent from the
        # forget set never index into B_c, but clamp anyway to stay finite.
        self._balance = (n_forget / (counts.clamp(min=1.0) * n_classes)) ** self.tau
        self._balance = self._balance.to(self.device)
        self._val_loader = val_loader
        self._mu = None
        self._sigma = None
        self._step = 0

    # ----------------------------------------------------------- estimation
    @torch.no_grad()
    def _estimate_val_stats(self):
        """Per-class mean/std of the true-class probability on the val set."""
        n_classes = int(self.rmodel.n_classes)
        sums = torch.zeros(n_classes, device=self.device)
        sumsq = torch.zeros(n_classes, device=self.device)
        counts = torch.zeros(n_classes, device=self.device)

        was_training = self.rmodel.training
        self.rmodel.eval()
        for x, y in self._val_loader:
            x, y = x.to(self.device), y.to(self.device)
            probs = self.rmodel(x).softmax(dim=-1)
            p = probs.gather(1, y.unsqueeze(1)).squeeze(1)
            sums.index_add_(0, y, p)
            sumsq.index_add_(0, y, p * p)
            counts.index_add_(0, y, torch.ones_like(p))
        if was_training:
            self.rmodel.train()

        valid = counts > 0
        mu = sums / counts.clamp(min=1.0)
        var = (sumsq / counts.clamp(min=1.0) - mu * mu).clamp(min=0.0)
        sigma = var.sqrt().clamp(min=self.sigma_min)
        return mu, sigma, valid

    # ----------------------------------------------------------------- loss
    def calculate_cost(self, train_data, reduction: str = "mean"):
        """Overridden. Reweighted random-label forget loss + retain loss."""
        if self._balance is None or self._val_loader is None:
            raise RuntimeError(
                "Call prepare(forget_loader, val_loader) before fit()."
            )
        if not isinstance(train_data, dict):
            raise TypeError(
                "%s needs Retain and Forget batches together. Wrap your loaders "
                "with MergedLoaders({'Retain': ..., 'Forget': ...}) and pass that "
                "to fit()." % type(self).__name__
            )

        # Algorithm 1 estimates (mu_c, sigma_c) at every step.
        if self._mu is None or self._step % self.estimate_every == 0:
            self._mu, self._sigma, self._valid_class = self._estimate_val_stats()
        self._step += 1

        x_f, y_f = train_data["Forget"]
        x_r, y_r = train_data["Retain"]
        x_f, y_f = x_f.to(self.device), y_f.to(self.device)
        x_r, y_r = x_r.to(self.device), y_r.to(self.device)

        ce_none = nn.CrossEntropyLoss(reduction="none")

        # Forget branch: CE against random labels, weighted by w_i.
        y_rand = torch.randint_like(y_f, low=0, high=int(self.rmodel.n_classes))
        logits_f = self.rmodel(x_f)
        fg_loss = ce_none(logits_f, y_rand)
        weights = self._forgetting_aware_weights(logits_f, y_f)
        weighted_fg = weights * fg_loss

        # Retain branch: plain CE against true labels.
        rt_loss = ce_none(self.rmodel(x_r), y_r)

        self.add_record_item("FGLoss", fg_loss.mean().item())
        self.add_record_item("RTLoss", rt_loss.mean().item())
        self.add_record_item("WMean", weights.mean().item())
        self.add_record_item("WMin", weights.min().item())
        self.add_record_item("WMax", weights.max().item())

        if reduction == "mean":
            # Algorithm 1: mean over the merged batch (joint normalization).
            cost = (
                self.forget_lambda * weighted_fg.sum()
                + self.retain_lambda * rt_loss.sum()
            ) / (fg_loss.numel() + rt_loss.numel())
            self.add_record_item("Cost", cost.item())
            return cost

        # Per-sample: concatenate so cost[:n_forget] stays the forget block.
        cost = torch.cat(
            [self.forget_lambda * weighted_fg, self.retain_lambda * rt_loss]
        )
        self.add_record_item("Cost", cost.mean().item())
        return cost

    # -------------------------------------------------------------- weights
    @torch.no_grad()
    def _forgetting_aware_weights(self, logits_f, y_f):
        """Eq. 7 weights from the current true-class probabilities.

        The weight is a function of the TRUE class probability p_i even
        though the loss is computed against random labels; it is detached
        so no gradient flows through the weighting.
        """
        p = logits_f.softmax(dim=-1).gather(1, y_f.unsqueeze(1)).squeeze(1)
        z = (p - self._mu[y_f]) / self._sigma[y_f]
        b = self._balance[y_f]
        w = 1.0 + torch.sign(z) * torch.tanh(z.abs()) ** (1.0 / b)
        # Classes never seen in the validation set have no target
        # distribution; fall back to the unweighted (w = 1) loss.
        w = torch.where(self._valid_class[y_f], w, torch.ones_like(w))
        return w

    # ---------------------------------------------------------------- state
    def state_dict(self):
        """Extra state so ``fit(refit=True)`` resumes mid-schedule."""
        return {"step": self._step, "balance": self._balance}

    def load_state_dict(self, state):
        self._step = state.get("step", 0)
        self._balance = state.get("balance", self._balance)
