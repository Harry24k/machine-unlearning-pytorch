"""KL Minimization: -CE(forget) + alpha * KL(ref || model) on the retain answers.  Baseline of FIUBench (ICLR 2025),
MLLMU-Bench and MLUBench (ICML 2026); identical to GradDiff with ``retain_loss_type="KL"``."""
from .graddiff import GradDiff


class KLMin(GradDiff):
    """Gradient ascent on forget + KL-to-reference on retain."""

    def __init__(self, rmodel, **kw):
        kw.setdefault("retain_loss_type", "KL")
        super().__init__(rmodel, **kw)
