"""AltPO (Mekala et al., COLING 2025): DPO whose win answer is one of M plausible alternates per sample
(SeqSample.alt_answers, M >= 1).  One epoch = one pass over the forget set; the dataset rotates the alternate
per epoch so M epochs show every (alternate, original) pair once.  Reference NLL of all M alternates is cached."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dpo import DPO


class AltPO(DPO):
    """DPO with self-generated alternates as the win answer (no NLL term on the alternates)."""
    ref_needs = ("forget_seq_nll",)

    def fit(self, train_loaders, *a, **kw):
        ds = getattr(train_loaders, "loaders", train_loaders)["Forget"].dataset
        ds.epoch_fn = lambda: max(self.accumulated_epoch - 1, 0)
        return super().fit(train_loaders, *a, **kw)

    @torch.no_grad()
    def _prepare_reference(self, train_loaders):
        super()._prepare_reference(train_loaders)
        if self.ref_mode == "model":
            return
        src = getattr(train_loaders, "loaders", train_loaders)["Forget"]
        ds = src.dataset
        if getattr(ds, "n_alt", 0) == 0:
            raise ValueError("AltPO needs SeqSample.alt_answers (>= 1 alternate per forget sample)")
        if any("alt_seq_nll_0" in d for d in self._cache["Forget"].values()):
            return
        self.rmodel.eval()
        try:
            for k in range(ds.n_alt):
                ds.fixed_alt = k
                dl = DataLoader(ds, batch_size=src.batch_size, shuffle=False, collate_fn=src.collate_fn)
                for batch in dl:
                    nll, _ = self._token_nll(self._logits(batch, prefix="alt_"), batch["alt_labels"])
                    for j, i in enumerate(batch["idx"].tolist()):
                        self._cache["Forget"].setdefault(i, {})[f"alt_seq_nll_{k}"] = nll[j].sum().item()
        finally:
            ds.fixed_alt = None
        self.rmodel.train()

    def _refs(self, batch, lose):
        if self.ref_mode == "model":
            return super()._refs(batch, lose)
        lose_ref = torch.tensor(self._cached("Forget", batch, "seq_nll"), device=lose.device)
        win_ref = torch.tensor([self._cache["Forget"][int(i)][f"alt_seq_nll_{int(k)}"]
                                for i, k in zip(batch["idx"].tolist(), batch["alt_k"].tolist())], device=lose.device)
        return lose_ref, win_ref
