"""Optional base class. Algorithms are NOT required to inherit from this;
the registry contract (presence of `fit`, and `setup` for trainers) is
enforced by @register_unlearner. Use BaseUnlearner only if you want shared
device handling and no-op setup/record_rob for non-trainers."""

import torch


class BaseUnlearner:
    def __init__(self, rmodel, device=None):
        self.rmodel = rmodel
        self.device = device or next(rmodel.parameters()).device

    def setup(self, **kwargs):
        """No-op default. Trainer-type unlearners should override this."""
        return self

    def record_rob(self, *args, **kwargs):
        """No-op default. Override in trainers that need per-epoch eval hooks."""
        return self

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement fit().")