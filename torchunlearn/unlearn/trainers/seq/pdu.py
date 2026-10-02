"""PDU (Entesari et al., NeurIPS 2025; open-unlearning pdu.py): primal-dual unlearning.

min L_f s.t. L_r <= eps on the Lagrangian gamma * L_f + lambda * (L_r - eps), lambda = alpha:
    forget  L_f = (max logit - mean logit)^2 on answer positions
    retain  L_r = NLL on the retain batch
    dual    lambda <- max(0, lambda + dual_step_size * (L_r - eps))  per step or per epoch
"""
from typing import List

import torch
from torch.utils.data import DataLoader

from .base import SeqUnlearner


class PDU(SeqUnlearner):
    """Primal-dual unlearning with a logit-margin forget loss."""

    def __init__(self, rmodel, primal_dual=False, dual_step_size=1.0, retain_loss_eps=0.0, dual_update_upon="step",
                 dual_warmup_epochs=0, **kw):
        super().__init__(rmodel, **kw)
        if dual_update_upon not in ("step", "epoch"):
            raise ValueError("dual_update_upon must be 'step' or 'epoch'")
        self.primal_dual, self.dual_step_size, self.retain_loss_eps = bool(primal_dual), float(dual_step_size), float(retain_loss_eps)
        self.dual_update_upon, self.dual_warmup_epochs = dual_update_upon, int(dual_warmup_epochs)
        self.alpha_init = self.alpha
        self.dual_trace: List[dict] = []
        self._retain_src = None

    def forget_loss(self, batch):
        logits = self._logits(batch).float()
        m = (self._labels(batch, "Forget").to(logits.device) != -100).reshape(-1)
        flat = logits.reshape(-1, logits.shape[-1])
        f = (flat.max(-1)[0] - flat.mean(-1)) ** 2
        return (f * m).sum() / m.sum().clamp(min=1)

    def _can_update(self):
        return self.primal_dual and (self.accumulated_epoch - 1) >= self.dual_warmup_epochs

    def fit(self, train_loaders, *a, **kw):
        self._retain_src = getattr(train_loaders, "loaders", {}).get("Retain")
        return super().fit(train_loaders, *a, **kw)

    def calculate_cost(self, train_data, reduction="mean"):
        f_loss = self.forget_loss(train_data["Forget"])
        rb = train_data.get("Retain")
        r_raw = self._retain_loss(rb) if rb is not None else torch.zeros((), device=self.device)
        r_loss = r_raw - self.retain_loss_eps
        cost = self.gamma * f_loss + self.alpha * r_loss
        if self.dual_update_upon == "step" and self._can_update():
            self.alpha = max(0.0, self.alpha + self.dual_step_size * float(r_loss.detach()))
        self.add_record_item("alpha", self.alpha)
        self.add_record_item("FGLoss", float(f_loss.detach())); self.add_record_item("RTLoss", float(r_raw.detach()))
        self.add_record_item("Cost", float(cost.detach()))
        if self.eval_every and self.stop_lp is not None and self.accumulated_iter % self.eval_every == 0:
            self._check_stop()
        return cost

    @torch.no_grad()
    def _epoch_dual_update(self):
        src = self._retain_src
        if src is None:
            return
        self.rmodel.eval()
        dl = DataLoader(src.dataset, batch_size=src.batch_size, shuffle=False, collate_fn=src.collate_fn)
        tot, nb = 0.0, 0
        for b in dl:
            tot += float(self._retain_loss(b)); nb += 1
        self.alpha = max(0.0, self.alpha + self.dual_step_size * (tot / max(nb, 1) - self.retain_loss_eps))
        self.rmodel.train()

    def record_during_eval(self):
        if self.dual_update_upon == "epoch" and self.primal_dual and self.accumulated_epoch >= self.dual_warmup_epochs:
            self._epoch_dual_update()
        self.dual_trace.append({"epoch": self.accumulated_epoch, "alpha": self.alpha})
        super().record_during_eval()
