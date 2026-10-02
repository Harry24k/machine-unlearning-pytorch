"""SeqFinetune: plain next-token fine-tuning.

* with a single loader (``make_loader``): train on every batch -> use it to *build* the model to unlearn from
  (fine-tune a base checkpoint on the dataset) or to retrain-from-scratch on the retain set.
* with MergedLoaders({"Forget", "Retain"}): retain-only fine-tuning (the classic "Finetune" unlearning baseline).
"""
import torch

from .base import SeqUnlearner


class SeqFinetune(SeqUnlearner):
    """NLL fine-tuning (loss_on is decided by the collator: answer tokens or the full text)."""

    def __init__(self, rmodel, **kw):
        kw.setdefault("alpha", 1.0)
        kw.setdefault("gamma", 0.0)
        kw.setdefault("retain_loss_type", "NLL")
        super().__init__(rmodel, **kw)

    def forget_loss(self, batch):
        return torch.zeros((), device=self.device)

    def fit(self, train_loaders, n_epochs, n_iters=None, **kw):
        if not hasattr(train_loaders, "loaders"):
            n_iters = n_iters or len(train_loaders)
        return super().fit(train_loaders, n_epochs, n_iters=n_iters, **kw)

    def calculate_cost(self, train_data, reduction="mean"):
        if isinstance(train_data, dict) and ("Forget" in train_data or "Retain" in train_data):
            batch = train_data.get("Retain")
            if batch is None:
                raise ValueError("SeqFinetune on MergedLoaders needs a Retain loader")
        else:
            batch = train_data
        labels = batch["labels"]
        logits = self._logits(batch)
        nll, mask = self._token_nll(logits, labels)
        cost = nll.sum() / mask.sum().clamp(min=1)
        self.add_record_item("Cost", float(cost.detach()))
        return cost
