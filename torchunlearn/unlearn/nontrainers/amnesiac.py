r"""Amnesiac Machine Learning (Graves, Nagisetty & Ganesh, AAAI 2021).

Reference
---------
Graves, L., Nagisetty, V., & Ganesh, V. (2021).
*Amnesiac Machine Learning.* Proceedings of the AAAI Conference on Artificial
Intelligence, 35(13), 11516-11524.

The paper proposes two complementary mechanisms for removing the influence of
a designated *forget set* :math:`D_f` from an already-trained neural network
without retraining from scratch.

1. ``Delta`` mode (the canonical "Amnesiac" procedure).
   During the *original* training run, the parameter update applied by every
   mini-batch is cached, i.e. for batch index ``b`` we store

   .. math:: \Delta\theta_b \;=\; \theta_{after}^{(b)} - \theta_{before}^{(b)}.

   When a forgetting request arrives, the user supplies the set of batch
   indices :math:`\mathcal{S}_B` that *contained at least one forget sample*
   and the unlearned weights are obtained simply by *subtracting back* those
   contributions:

   .. math:: \theta_{unlearn} \;=\; \theta_{final} - \sum_{b\in\mathcal{S}_B} \Delta\theta_b.

   A short fine-tune on :math:`D_r` (the retain split) is then run as a
   ``repair`` step to restore any retain-side accuracy that incidentally
   shared the subtracted update directions.

   .. note::
       Subtracting :math:`\Delta\theta_b` undoes the *direct* effect of batch
       ``b`` on the parameters but does **not** undo its indirect effect via
       stateful optimisers (momentum, Adam's running moments propagate the
       gradient from batch ``b`` into subsequent steps' updates). The
       ``repair`` fine-tune is what cleans this residual up; do not expect
       raw subtraction alone to bit-perfectly erase influence.

   .. note::
       The recorder is not aware of mixed-precision wrappers, gradient
       accumulation, or optimisers whose ``step`` is conditionally skipped
       (e.g. ``torch.cuda.amp.GradScaler``). In such settings each recorded
       delta is the change effected by *one call to step*, which may
       correspond to several batches' worth of gradient or to none at all.
       Use ``record_predicate`` / ``record_only_indices`` to keep the cache
       aligned with the batches you actually care about.

2. ``Relabel`` mode (the second variant in the paper).
   When the per-batch deltas were *not* cached during training we approximate
   amnesia by continuing optimisation on batches that *replace* the true
   labels of the forget samples with uniformly random class labels. A few
   epochs of this push the model's predictions on :math:`D_f` away from the
   memorised targets without harming the retain split too much.

This implementation supports both modes through a single :class:`Amnesiac`
class and follows the conventions used by the other :mod:`...nontrainers`
utilities in this benchmark (e.g. :class:`FisherForget`, :class:`NegMerge`):

* takes a :class:`RobModel` at construction time,
* exposes ``setup`` / ``record_rob`` / ``fit`` entry-points,
* leaves the original model untouched and returns a *deep-copied*
  unlearned model so downstream evaluation can compare against the
  pre-unlearn checkpoint.

Example
-------
>>> amn = Amnesiac(rmodel, mode='delta')
>>> amn.attach_recorder(optimizer,
...                     record_only_indices={3, 17, 42})  # batches with forget samples
>>> # ... run training; amn now holds amn.deltas keyed by step index
>>> amn.fit(forget_batch_ids=[3, 17, 42],
...         retain_loader=retain_loader,
...         repair_epochs=2)
>>> unlearned_model = amn.rmodel
"""

import copy
import logging
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ... import RobModel


# Public type aliases.
DeltaDict = Dict[str, torch.Tensor]
RecordPredicate = Callable[[int], bool]


class Amnesiac:
    r"""Amnesiac Machine Learning unlearner.

    Arguments
    ---------
    rmodel : RobModel
        The trained model to unlearn from. A deep copy is taken internally so
        the caller's checkpoint is never mutated.
    mode : {'delta', 'relabel'}
        Which of the two variants from the paper to run.

        * ``'delta'`` -- selective subtraction of cached per-batch updates.
          Requires either :meth:`attach_recorder` to have been called before
          the original training run, or pre-computed deltas to be passed via
          :meth:`load_deltas`.
        * ``'relabel'`` -- random-label continuation training on the forget
          batches. Suitable when per-batch deltas are unavailable.
    device : torch.device or None
        Device on which to perform unlearning. Defaults to the device of the
        model's first parameter.
    """

    _VALID_MODES = ('delta', 'relabel')

    def __init__(self, rmodel, mode: str = 'delta', device=None):
        assert isinstance(rmodel, RobModel), \
            f"`rmodel` must be a RobModel, got {type(rmodel).__name__}."
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"`mode` must be one of {self._VALID_MODES}, got {mode!r}.")

        self.rmodel = rmodel
        if mode is None : 
            self.mode = 'relabel'
        else :
            self.mode = mode
            
        if device is None:
            self.device = next(rmodel.parameters()).device
        else:
            self.device = device

        # Per-batch parameter deltas keyed by the optimiser-step index of the
        # *original* training run. We use a dict (not a list) so that
        # selective recording leaves no gaps: caller's ``forget_batch_ids``
        # are the absolute step indices regardless of which were cached.
        self.deltas: Dict[int, DeltaDict] = {}

        # Internal bookkeeping for the optimiser hook used by ``attach_recorder``.
        self._recorder_handle = None
        self._step_counter = 0
        self._record_predicate: Optional[RecordPredicate] = None
        self._record_dtype: torch.dtype = torch.float32
        self._record_device: Union[torch.device, str] = 'cpu'

        # Optional record_rob hook (kept for API parity with FisherForget).
        self._record_loaders = None

    # ------------------------------------------------------------------ API
    def setup(self, **kwargs):
        """No-op hook kept for parity with other nontrainers."""
        return self

    def record_rob(self, loaders, n_limit=None):
        """Stash robustness-evaluation loaders for downstream logging."""
        self._record_loaders = loaders
        return self

    # ----------------------------------------------------- delta-mode helpers
    def _named_trainable(self):
        """Yield ``(name, parameter)`` for every trainable parameter."""
        for name, p in self.rmodel.named_parameters():
            if p.requires_grad:
                yield name, p

    def attach_recorder(
        self,
        optimizer: torch.optim.Optimizer,
        record_predicate: Optional[RecordPredicate] = None,
        record_only_indices: Optional[Iterable[int]] = None,
        store_dtype: torch.dtype = torch.float32,
        store_device: Union[torch.device, str] = 'cpu',
    ):
        """Wrap ``optimizer.step`` so that selected steps cache ``Delta theta_b``.

        The delta is computed *on the parameter's own device* and copied to
        ``store_device`` (default CPU) exactly once, in ``store_dtype``
        (default fp32; pass ``torch.float16`` to halve cache size).

        Parameters
        ----------
        optimizer
            The training optimiser to wrap. Must be called *before* the
            training loop starts.
        record_predicate : callable, optional
            ``step_idx -> bool``. Returning ``False`` skips caching for that
            step; the step still runs normally. Use this for fine-grained
            selection (e.g. "only steps that touch a forget sample").
        record_only_indices : iterable of int, optional
            Convenience equivalent of
            ``record_predicate=lambda i: i in set(record_only_indices)``.
            Mutually exclusive with ``record_predicate``.
        store_dtype : torch.dtype
            Dtype the cached deltas are cast to. fp16 cuts memory in half
            with negligible impact on subtraction quality.
        store_device : torch.device or str
            Where to keep the deltas. CPU by default; pass ``'cuda'`` only
            if you have headroom.

        Notes
        -----
        Re-attaching to the *same* optimiser is a no-op. Re-attaching to a
        *different* optimiser raises ``RuntimeError`` -- detach first.
        """
        if self._recorder_handle is not None:
            previous_opt, _ = self._recorder_handle
            if previous_opt is optimizer:
                return self
            raise RuntimeError(
                "A recorder is already attached to a different optimiser. "
                "Call `detach_recorder()` first.")

        if record_predicate is not None and record_only_indices is not None:
            raise ValueError(
                "Pass at most one of `record_predicate` / `record_only_indices`.")
        if record_only_indices is not None:
            allowed: Set[int] = {int(i) for i in record_only_indices}
            record_predicate = lambda step_idx, _allowed=allowed: step_idx in _allowed

        self._record_predicate = record_predicate
        self._record_dtype = store_dtype
        self._record_device = store_device
        self._step_counter = 0

        original_step = optimizer.step

        def _wrapped_step(*args, **kwargs):
            step_idx = self._step_counter
            self._step_counter += 1

            should_record = (
                self._record_predicate is None
                or self._record_predicate(step_idx)
            )
            if not should_record:
                return original_step(*args, **kwargs)

            # Snapshot before, run the real step, then compute the delta on
            # the parameter's own device. One copy per parameter, not two.
            before = {
                name: p.detach().clone()
                for name, p in self._named_trainable()
            }
            out = original_step(*args, **kwargs)
            delta: DeltaDict = {}
            with torch.no_grad():
                for name, p in self._named_trainable():
                    diff = (p.detach() - before[name]).to(
                        device=self._record_device,
                        dtype=self._record_dtype,
                        non_blocking=True,
                    )
                    delta[name] = diff
            self.deltas[step_idx] = delta
            return out

        optimizer.step = _wrapped_step  # type: ignore[assignment]
        self._recorder_handle = (optimizer, original_step)
        return self

    def detach_recorder(self):
        """Restore the original ``optimizer.step`` (call after training)."""
        if self._recorder_handle is None:
            return self
        optimizer, original_step = self._recorder_handle
        optimizer.step = original_step  # type: ignore[assignment]
        self._recorder_handle = None
        return self

    def load_deltas(self, deltas: Union[Sequence[DeltaDict], Dict[int, DeltaDict]]):
        """Inject externally computed per-batch deltas (e.g. loaded from disk).

        Accepts either a list (treated as dense step indices 0..N-1) or a
        dict ``{step_idx: {name: tensor}}`` for sparse recordings.
        """
        if isinstance(deltas, dict):
            iterator = deltas.items()
        else:
            iterator = enumerate(deltas)
        self.deltas = {
            int(b): {k: v.detach().clone().cpu() for k, v in d.items()}
            for b, d in iterator
        }
        return self

    def save_deltas(self, path: str, overwrite: bool = False):
        """Persist the cached deltas to disk via ``torch.save``.

        The cache can be large -- for an ``N``-parameter model trained for
        ``T`` recorded steps this writes roughly ``N * T * sizeof(dtype)``
        bytes. Prefer ``store_dtype=torch.float16`` and a record predicate
        in :meth:`attach_recorder` to keep this tractable.
        """
        if os.path.exists(path) and not overwrite:
            raise ValueError(f"[{path}] already exists.")
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.deltas, path)
        return self

    # --------------------------------------------------------- ergonomics
    @staticmethod
    def derive_forget_batch_ids(
        train_loader,
        is_forget_sample: Callable[[int], bool],
    ) -> List[int]:
        """Walk ``train_loader`` once and return the step indices whose batch
        contains at least one forget sample.

        ``is_forget_sample`` takes a *dataset-level* sample index and returns
        ``True`` if that sample belongs to the forget split. This requires
        the loader to yield ``(x, y, idx)`` tuples; if your loader yields
        ``(x, y)`` only, compute the ids yourself from the underlying
        sampler / dataset.
        """
        forget_ids: List[int] = []
        for step_idx, batch in enumerate(train_loader):
            if len(batch) < 3:
                raise ValueError(
                    "Loader must yield (x, y, idx) for automatic derivation; "
                    "got a batch of length {}.".format(len(batch)))
            idxs = batch[2]
            idx_iter = idxs.tolist() if torch.is_tensor(idxs) else list(idxs)
            if any(is_forget_sample(int(i)) for i in idx_iter):
                forget_ids.append(step_idx)
        return forget_ids

    # ------------------------------------------------------ subtraction core
    def _subtract_deltas(self, forget_batch_ids: Iterable[int]):
        r"""In-place :math:`\theta \leftarrow \theta - \sum_{b\in S} \Delta\theta_b`."""
        if len(self.deltas) == 0:
            raise RuntimeError(
                "No per-batch deltas were recorded. Call `attach_recorder` "
                "before training, supply them via `load_deltas`, or switch "
                "to mode='relabel'.")

        forget_batch_ids = sorted({int(b) for b in forget_batch_ids})
        missing = [b for b in forget_batch_ids if b not in self.deltas]
        if missing:
            raise KeyError(
                f"No cached delta for batch ids {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}. Either the recorder "
                "skipped them (check `record_predicate`) or they fall "
                "outside the recorded range.")

        # Aggregate first to minimise per-step memory traffic.
        with torch.no_grad():
            agg: DeltaDict = {}
            for b in forget_batch_ids:
                for name, dv in self.deltas[b].items():
                    if name not in agg:
                        agg[name] = dv.clone().float()
                    else:
                        agg[name].add_(dv.float())

            named_params = dict(self.rmodel.named_parameters())
            for name, dv in agg.items():
                if name not in named_params:
                    logging.warning(
                        "Parameter %r recorded in deltas but missing from "
                        "the current model; skipping.", name)
                    continue
                p = named_params[name]
                p.data.add_(dv.to(device=p.device, dtype=p.dtype), alpha=-1.0)
        return self

    # --------------------------------------------------------- relabel core
    def _relabel_step(self, x, y_true, optimizer, exclude_true_label: bool):
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        n_classes = self.rmodel.n_classes
        if exclude_true_label:
            # Sample a random offset in [1, n_classes - 1] and add modulo
            # n_classes to guarantee a different label without rejection
            # sampling.
            offsets = torch.randint(
                low=1, high=n_classes, size=y_true.shape, device=self.device,
            )
            y_rand = (y_true + offsets) % n_classes
        else:
            y_rand = torch.randint(
                low=0, high=n_classes, size=y_true.shape, device=self.device,
            )
        logits = self.rmodel(x)
        loss = F.cross_entropy(logits, y_rand)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item()

    # ----------------------------------------------------- repair fine-tune
    def _repair(self, retain_loader, epochs: int, lr: float):
        if epochs <= 0 or retain_loader is None:
            return
        optimizer = torch.optim.SGD(
            [p for p in self.rmodel.parameters() if p.requires_grad],
            lr=lr, momentum=0.9,
        )
        self.rmodel.train()
        for ep in range(epochs):
            pbar = tqdm(retain_loader, desc=f"[Amnesiac] repair ep {ep + 1}/{epochs}")
            running = 0.0
            n_batches = 0
            for batch in pbar:
                x, y = batch[0], batch[1]
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.rmodel(x)
                loss = F.cross_entropy(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{running / max(1, n_batches):.4f}")

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        forget_batch_ids: Optional[Sequence[int]] = None,
        forget_loader = None,
        retain_loader = None,
        repair_epochs: int = 0,
        repair_lr: float = 1e-3,
        relabel_epochs: int = 1,
        relabel_lr: float = 1e-3,
        relabel_exclude_true_label: bool = False,
        save_path: Optional[str] = None,
        overwrite: bool = False,
    ):
        r"""Run the amnesiac unlearning procedure.

        Parameters
        ----------
        forget_batch_ids : sequence of int, optional
            Step indices (into ``self.deltas``) of batches that touched the
            forget split. Required for ``mode='delta'``; ignored otherwise.
        forget_loader : DataLoader, optional
            Loader over :math:`D_f`. Required for ``mode='relabel'``.
        retain_loader : DataLoader, optional
            Loader over :math:`D_r`. Used for the post-subtraction repair
            fine-tune. Strongly recommended even for ``mode='relabel'``.
        repair_epochs : int
            Number of fine-tune epochs on the retain split *after* the
            forgetting step. Set to 0 to skip.
        repair_lr : float
            Learning rate for the repair optimiser (plain SGD + momentum).
        relabel_epochs : int
            Number of random-label passes over the forget loader
            (``mode='relabel'`` only). Set to 0 to skip the relabel pass
            entirely (e.g. if you only want the repair fine-tune).
        relabel_lr : float
            Learning rate for the relabel optimiser.
        relabel_exclude_true_label : bool
            If ``True``, the random replacement label is drawn uniformly
            from the ``n_classes - 1`` *other* classes. Defaults to
            ``False`` to match the paper exactly.
        save_path : str, optional
            If provided, the unlearned model's ``state_dict`` is written to
            this path on completion.
        overwrite : bool
            Whether ``save_path`` may overwrite an existing file.
        """
        if save_path is not None and os.path.exists(save_path) and not overwrite:
            raise ValueError(f"[{save_path}] already exists.")

        # Always work on a copy so the caller's checkpoint stays pristine.
        self.rmodel = copy.deepcopy(self.rmodel).to(self.device)
        
        if self.mode == 'delta':
            if forget_batch_ids is None:
                raise ValueError(
                    "mode='delta' requires `forget_batch_ids`.")
            logging.info(
                "[Amnesiac] subtracting %d cached batch updates.",
                len(forget_batch_ids))
            self._subtract_deltas(forget_batch_ids)

        elif self.mode == 'relabel':
            if forget_loader is None:
                raise ValueError(
                    "mode='relabel' requires `forget_loader`.")
            if relabel_epochs < 0:
                raise ValueError("`relabel_epochs` must be non-negative.")
            if relabel_epochs > 0:
                optimizer = torch.optim.SGD(
                    [p for p in self.rmodel.parameters() if p.requires_grad],
                    lr=relabel_lr, momentum=0.9,
                )
                self.rmodel.train()
                for ep in range(relabel_epochs):
                    pbar = tqdm(
                        forget_loader,
                        desc=f"[Amnesiac] relabel ep {ep + 1}/{relabel_epochs}",
                    )
                    running = 0.0
                    n_batches = 0
                    for batch in pbar:
                        x, y = batch[0], batch[1]
                        running += self._relabel_step(
                            x, y, optimizer, relabel_exclude_true_label,
                        )
                        n_batches += 1
                        pbar.set_postfix(
                            loss=f"{running / max(1, n_batches):.4f}")

        # Post-step repair fine-tune on the retain split.
        self._repair(retain_loader, repair_epochs, repair_lr)

        self.rmodel.eval()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save(self.rmodel.state_dict(), save_path)
        return self