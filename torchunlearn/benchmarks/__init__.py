"""torchunlearn.benchmarks -- side-by-side comparison of unlearned models.

This module deliberately does NOT run unlearning algorithms. You bring
pre-trained unlearned models (one per method you care about) and it
evaluates them with :class:`torchunlearn.metrics.UnlearningEvaluator`,
then formats the results as a comparison table or a pandas DataFrame.

Typical usage
-------------
    from torchunlearn.benchmarks import compare, BenchmarkSuite

    # Option A: one-shot
    results = compare(
        models={"Finetune": ft_model, "NegGrad": ng_model, "UAM": uam_model},
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        retrained_model=retrained,   # optional, enables UG
        n_limit=1000,
    )
    print_table(results)
    df = to_dataframe(results)

    # Option B: incremental (add more models later)
    suite = BenchmarkSuite(train_loaders, test_loaders,
                           retrained_model=retrained, n_limit=1000)
    suite.add("Finetune", ft_model)
    suite.add("NegGrad",  ng_model)
    suite.print_table()
    df = suite.to_dataframe()
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Mapping, Optional

from ..metrics import UnlearningEvaluator

# Order in which metrics appear in tables / DataFrames.
_METRIC_ORDER = ["RA", "FA", "TA", "UG", "MIA_Efficiency", "ZRF"]

# How each metric is displayed (suffix + decimals).
_METRIC_FMT = {
    "RA":             ("%", 2),
    "FA":             ("%", 2),
    "TA":             ("%", 2),
    "UG":             ("%", 2),
    "MIA_Efficiency": ("",  4),
    "ZRF":            ("",  4),
}


def _validate_loaders(train_loaders: Mapping[str, Any],
                      test_loaders: Mapping[str, Any]) -> None:
    for k in ("Retain", "Forget"):
        if k not in train_loaders:
            raise ValueError(f"train_loaders must contain '{k}' key")
    if "Test" not in test_loaders:
        raise ValueError("test_loaders must contain 'Test' key")


def _evaluate_one(model,
                  train_loaders: Mapping[str, Any],
                  test_loaders: Mapping[str, Any],
                  retrained_model=None,
                  n_limit: Optional[int] = 1000,
                  device=None) -> Dict[str, float]:
    evaluator = UnlearningEvaluator(
        unlearned_model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        retrained_model=retrained_model,
        n_limit=n_limit,
        device=device,
    )
    return evaluator.evaluate()


def compare(models: Mapping[str, Any],
            train_loaders: Mapping[str, Any],
            test_loaders: Mapping[str, Any],
            retrained_model=None,
            n_limit: Optional[int] = 1000,
            device=None,
            verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """Evaluate every model in ``models`` and return ``{name: metrics}``.

    Parameters
    ----------
    models : dict[str, nn.Module]
        Already-unlearned models, keyed by display name.
    train_loaders, test_loaders : dict
        Must contain 'Retain'/'Forget' and 'Test' respectively.
    retrained_model : nn.Module, optional
        Gold-standard retrained model; required for the UG metric.
    n_limit : int, optional
        Cap on samples per loader (speed knob). ``None`` disables.
    device : torch.device or str, optional
        Defaults to each model's own device.
    verbose : bool
        Print per-model progress.
    """
    _validate_loaders(train_loaders, test_loaders)
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        if verbose:
            print(f"[benchmarks] Evaluating: {name}")
        results[name] = _evaluate_one(
            model, train_loaders, test_loaders,
            retrained_model=retrained_model,
            n_limit=n_limit,
            device=device,
        )
    return results


def print_table(results: Mapping[str, Mapping[str, float]],
                metrics: Optional[list] = None,
                name_w: int = 18,
                col_w: int = 14) -> None:
    """Pretty-print a comparison table of ``compare()`` results."""
    keys = list(metrics) if metrics else _METRIC_ORDER
    header = f"{'Method':<{name_w}}" + "".join(f"{k:>{col_w}}" for k in keys)
    bar = "=" * len(header)
    print(bar)
    print(f"{'Unlearning Benchmark Results':^{len(header)}}")
    print(bar)
    print(header)
    print("-" * len(header))

    for name, metric_dict in results.items():
        row = f"{name:<{name_w}}"
        for k in keys:
            if k in metric_dict:
                suffix, ndp = _METRIC_FMT.get(k, ("", 4))
                row += f"{metric_dict[k]:>{col_w - len(suffix)}.{ndp}f}{suffix}"
            else:
                row += f"{'N/A':>{col_w}}"
        print(row)
    print(bar)


def to_dataframe(results: Mapping[str, Mapping[str, float]],
                 metrics: Optional[list] = None):
    """Return a pandas DataFrame (rows=methods, cols=metrics), or None if pandas missing."""
    try:
        import pandas as pd
    except ImportError:
        warnings.warn("pandas is not installed; returning None.")
        return None
    keys = list(metrics) if metrics else _METRIC_ORDER
    df = pd.DataFrame(results).T
    # Reorder columns: known metrics first (if present), then anything extra.
    ordered = [k for k in keys if k in df.columns] + [c for c in df.columns if c not in keys]
    return df[ordered]


class BenchmarkSuite:
    """Stateful counterpart to :func:`compare` for incremental use.

    Useful when you want to add unlearned models to the comparison as they
    finish training (e.g. across notebook cells) rather than all at once.
    """

    def __init__(self,
                 train_loaders: Mapping[str, Any],
                 test_loaders: Mapping[str, Any],
                 retrained_model=None,
                 n_limit: Optional[int] = 1000,
                 device=None):
        _validate_loaders(train_loaders, test_loaders)
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.retrained_model = retrained_model
        self.n_limit = n_limit
        self.device = device
        self.results: Dict[str, Dict[str, float]] = {}

    def add(self, name: str, model, verbose: bool = True) -> Dict[str, float]:
        """Evaluate ``model`` and store the result under ``name``."""
        if verbose:
            print(f"[benchmarks] Evaluating: {name}")
        metrics = _evaluate_one(
            model, self.train_loaders, self.test_loaders,
            retrained_model=self.retrained_model,
            n_limit=self.n_limit,
            device=self.device,
        )
        self.results[name] = metrics
        return metrics

    def add_many(self, models: Mapping[str, Any], verbose: bool = True) -> None:
        for name, model in models.items():
            self.add(name, model, verbose=verbose)

    def print_table(self, metrics: Optional[list] = None) -> None:
        print_table(self.results, metrics=metrics)

    def to_dataframe(self, metrics: Optional[list] = None):
        return to_dataframe(self.results, metrics=metrics)


__all__ = ["compare", "print_table", "to_dataframe", "BenchmarkSuite"]
