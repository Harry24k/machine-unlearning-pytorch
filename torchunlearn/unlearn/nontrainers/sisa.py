"""SISA: Sharded, Isolated, Sliced, and Aggregated Training for Machine Unlearning.

Reference:
    Bourtoule et al., "Machine Unlearning",
    IEEE Symposium on Security and Privacy (S&P), 2021.

Algorithm overview:
    SISA is an ARCHITECTURAL approach to machine unlearning, not a fine-tuning
    method. It modifies the training protocol so that unlearning becomes cheap.

    Key ideas:
    1. SHARDING: Split the training dataset into S disjoint shards.
       Train S independent constituent models, one per shard.

    2. SLICING: Each shard is further divided into R ordered slices.
       Intermediate checkpoints are saved after each slice so that
       retraining only needs to resume from the slice containing the
       forget sample (not from scratch).

    3. AGGREGATION: At inference time, predictions from all S shard
       models are aggregated (majority vote or average softmax).

    Unlearning a sample x_f:
    - Identify which shard and slice contains x_f.
    - Reload the checkpoint BEFORE that slice.
    - Retrain only that shard from that checkpoint with x_f removed.
    - All other shards are unchanged.

Note:
    Unlike other methods in this library, SISA is a TRAINING-PROTOCOL wrapper
    rather than a post-hoc fine-tuning method.
"""

from __future__ import annotations

import os
from typing import Callable, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


def aggregate_predictions(
    shard_logits: List[torch.Tensor],
    mode: str = "soft_vote",
) -> torch.Tensor:
    """Combine logits from multiple shard models.

    Arguments:
        shard_logits: list of (B, C) tensors, one per shard.
        mode: soft_vote (average softmax, default) or hard_vote (majority).
    """
    stacked = torch.stack(shard_logits, dim=0)
    if mode == "soft_vote":
        avg_probs = torch.softmax(stacked, dim=-1).mean(dim=0)
        return avg_probs.argmax(dim=1)
    elif mode == "hard_vote":
        votes = stacked.argmax(dim=-1)
        return votes.mode(dim=0).values
    else:
        raise ValueError(f"Unknown aggregation mode: {mode!r}")


class SISATrainingProtocol:
    """Orchestrates sharded + sliced training for SISA.

    Arguments:
        model_factory: callable returning a fresh (untrained) model instance.
        n_shards (int): number of dataset shards S.
        n_slices (int): number of slices R per shard.
        checkpoint_dir (str): directory for saving slice checkpoints.
        train_fn: callable(model, DataLoader, shard_idx, slice_idx) -> model.
        device: torch.device.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        n_shards: int,
        n_slices: int,
        checkpoint_dir: str,
        train_fn: Callable,
        device=None,
    ):
        self.model_factory = model_factory
        self.n_shards = n_shards
        self.n_slices = n_slices
        self.checkpoint_dir = checkpoint_dir
        self.train_fn = train_fn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train(
        self,
        dataset: Dataset,
        batch_size: int = 128,
        shuffle: bool = True,
    ) -> List[nn.Module]:
        """Train all shards and save per-slice checkpoints."""
        indices = list(range(len(dataset)))
        shard_size = len(indices) // self.n_shards
        shard_indices = [
            indices[i * shard_size: (i + 1) * shard_size]
            for i in range(self.n_shards)
        ]
        trained_models = []
        for shard_idx, shard_idx_list in enumerate(shard_indices):
            model = self.model_factory().to(self.device)
            slice_size = max(1, len(shard_idx_list) // self.n_slices)
            for slice_idx in range(self.n_slices):
                slice_end = min((slice_idx + 1) * slice_size, len(shard_idx_list))
                cumulative_indices = shard_idx_list[:slice_end]
                subset = Subset(dataset, cumulative_indices)
                loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
                model = self.train_fn(model, loader, shard_idx, slice_idx)
                torch.save(model.state_dict(), self._checkpoint_path(shard_idx, slice_idx))
            trained_models.append(model)
        return trained_models

    def _checkpoint_path(self, shard_idx: int, slice_idx: int) -> str:
        return os.path.join(self.checkpoint_dir, f"shard{shard_idx}_slice{slice_idx}.pt")


class SISAUnlearner:
    """Handles forget requests for a SISA ensemble.

    Arguments:
        protocol: the SISATrainingProtocol used for original training.
        shard_models: list of trained shard models.
        shard_index_lists: list of index lists, one per shard.
        dataset: the original full training dataset.
        train_fn: same callable used during original training.
        batch_size (int): batch size for retraining.
    """

    def __init__(
        self,
        protocol: SISATrainingProtocol,
        shard_models: List[nn.Module],
        shard_index_lists: List[List[int]],
        dataset: Dataset,
        train_fn: Callable,
        batch_size: int = 128,
    ):
        self.protocol = protocol
        self.shard_models = shard_models
        self.shard_index_lists = shard_index_lists
        self.dataset = dataset
        self.train_fn = train_fn
        self.batch_size = batch_size

    def unlearn(self, forget_indices: Sequence[int]) -> List[nn.Module]:
        """Remove forget_indices from the ensemble via targeted retraining.

        For each affected shard:
          1. Find the earliest slice containing a forget sample.
          2. Load the checkpoint from the slice BEFORE that one.
          3. Retrain from that checkpoint on shard data minus forget samples.
        """
        forget_set = set(forget_indices)
        for shard_idx, shard_indices in enumerate(self.shard_index_lists):
            affected = [i for i in shard_indices if i in forget_set]
            if not affected:
                continue
            slice_size = max(1, len(shard_indices) // self.protocol.n_slices)
            first_affected_slice = self._find_first_affected_slice(
                shard_indices, forget_set, slice_size
            )
            resume_slice = max(0, first_affected_slice - 1)
            model = self.protocol.model_factory().to(self.protocol.device)
            if resume_slice > 0:
                ckpt_path = self.protocol._checkpoint_path(shard_idx, resume_slice - 1)
                if os.path.exists(ckpt_path):
                    model.load_state_dict(
                        torch.load(ckpt_path, map_location=self.protocol.device)
                    )
            clean_indices = [i for i in shard_indices if i not in forget_set]
            for slice_idx in range(resume_slice, self.protocol.n_slices):
                slice_end = min((slice_idx + 1) * slice_size, len(clean_indices))
                subset = Subset(self.dataset, clean_indices[:slice_end])
                loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
                model = self.train_fn(model, loader, shard_idx, slice_idx)
                torch.save(
                    model.state_dict(),
                    self.protocol._checkpoint_path(shard_idx, slice_idx),
                )
            self.shard_models[shard_idx] = model
        return self.shard_models

    @staticmethod
    def _find_first_affected_slice(
        shard_indices: List[int], forget_set: set, slice_size: int
    ) -> int:
        for i, idx in enumerate(shard_indices):
            if idx in forget_set:
                return i // slice_size
        return 0


class SISAAggregator:
    """Wraps a SISA ensemble for inference.

    Arguments:
        shard_models: list of trained/unlearned shard models.
        mode: aggregation mode (soft_vote or hard_vote).
    """

    def __init__(self, shard_models: List[nn.Module], mode: str = "soft_vote"):
        self.shard_models = shard_models
        self.mode = mode

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate predictions from all shards."""
        logits_list = [m.eval()(x) if not callable(m) else m.eval()(x)
                       for m in self.shard_models]
        logits_list = []
        for model in self.shard_models:
            model.eval()
            logits_list.append(model(x))
        return aggregate_predictions(logits_list, mode=self.mode)

    def eval_accuracy(self, data_loader: DataLoader, device=None) -> float:
        """Compute accuracy of the SISA ensemble on a data loader."""
        if device is None:
            device = next(self.shard_models[0].parameters()).device
        correct = total = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            preds = self.predict(x)
            correct += (preds == y).sum().item()
            total += len(y)
        return 100.0 * correct / total if total > 0 else 0.0
