"""torchunlearn.metrics -- standardized evaluation suite for machine unlearning.

Provides UnlearningEvaluator which computes canonical metrics:
- RA (Retain Accuracy): should stay high
- FA (Forget Accuracy): should drop after unlearning
- TA (Test Accuracy): generalization should be maintained
- UG (Unlearning Gap): |RA_retrain - RA_unlearned| (smaller is better)
- MIA Score: membership-inference attack success on forget set (lower = better)
- ZRF Score: zero-retrain forgetting score (Chundawat et al., 2023)
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import torch
from ..utils.mia import MIA_SVC


class UnlearningEvaluator:
    """Compute standard machine-unlearning evaluation metrics.

    Arguments:
        unlearned_model: the model after unlearning (a RobModel).
        train_loaders: dict with Retain and Forget keys.
        test_loaders: dict with Test key.
        retrained_model: gold-standard retrained model (optional, needed for UG).
        n_limit (int): cap samples per loader for speed. None disables cap.
        device: target device. Defaults to the model's current device.
    """

    def __init__(
        self,
        unlearned_model,
        train_loaders: Dict[str, Any],
        test_loaders: Dict[str, Any],
        retrained_model=None,
        n_limit: Optional[int] = 1000,
        device=None,
    ):
        self.unlearned_model = unlearned_model
        self.retrained_model = retrained_model
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.n_limit = n_limit
        self.device = device or next(unlearned_model.parameters()).device

    def evaluate(self) -> Dict[str, float]:
        """Run all metrics and return a result dict with RA, FA, TA, UG, MIA_Efficiency, ZRF."""
        results: Dict[str, float] = {}
        results["RA"] = self._eval_accuracy(self.unlearned_model, self.train_loaders["Retain"])
        results["FA"] = self._eval_accuracy(self.unlearned_model, self.train_loaders["Forget"])
        results["TA"] = self._eval_accuracy(self.unlearned_model, self.test_loaders["Test"])
        if self.retrained_model is not None:
            retrain_ra = self._eval_accuracy(self.retrained_model, self.train_loaders["Retain"])
            results["UG"] = abs(retrain_ra - results["RA"])
        try:
            logits = self._collect_logits()
            mia_res = MIA_SVC(logits, prob_only=True)
            results["MIA_Efficiency"] = mia_res["MIA Efficiency"]
        except Exception:
            pass
        try:
            results["ZRF"] = self._compute_zrf()
        except Exception:
            pass
        return results

    def print_report(self, results: Optional[Dict[str, float]] = None) -> None:
        """Pretty-print the evaluation report."""
        if results is None:
            results = self.evaluate()
        width = 50
        print("=" * width)
        print(f"{'Unlearning Evaluation Report':^{width}}")
        print("=" * width)
        labels = {
            "RA": "Retain Accuracy      (high is good)",
            "FA": "Forget Accuracy      (low  is good)",
            "TA": "Test Accuracy        (high is good)",
            "UG": "Unlearning Gap       (low  is good)",
            "MIA_Efficiency": "MIA Efficiency  (low  is good)",
            "ZRF": "ZRF Score            (high is good)",
        }
        for key, label in labels.items():
            if key in results:
                print(f"  {label}: {results[key]:.4f}")
        print("=" * width)

    @torch.no_grad()
    def _eval_accuracy(self, model, loader) -> float:
        model.eval()
        correct = total = 0
        for x, y in loader:
            if self.n_limit and total >= self.n_limit:
                break
            x, y = x.to(self.device), y.to(self.device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
        return 100.0 * correct / total if total > 0 else 0.0

    @torch.no_grad()
    def _collect_logits(self) -> Dict[str, tuple]:
        out = {}
        for flag, loader in [("(R)", self.train_loaders["Retain"]),
                              ("(Te)", self.test_loaders["Test"]),
                              ("(F)", self.train_loaders["Forget"])]:
            all_x, all_y = [], []
            for x, y in loader:
                if self.n_limit and sum(len(a) for a in all_y) >= self.n_limit:
                    break
                all_x.append(self.unlearned_model(x.to(self.device)).cpu())
                all_y.append(y)
            out[flag] = (torch.cat(all_x), torch.cat(all_y))
        return out

    @torch.no_grad()
    def _compute_zrf(self) -> float:
        """Zero-Retrain Forgetting score (Chundawat et al., 2023)."""
        import copy
        ref_model = copy.deepcopy(self.unlearned_model)
        for layer in ref_model.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        ref_model.eval()
        self.unlearned_model.eval()
        cosine = torch.nn.CosineSimilarity(dim=1)
        sims, n = [], 0
        for x, _ in self.train_loaders["Forget"]:
            if self.n_limit and n >= self.n_limit:
                break
            x = x.to(self.device)
            sim = cosine(self.unlearned_model(x), ref_model(x)).mean().item()
            sims.append((sim + 1.0) / 2.0)
            n += len(x)
        del ref_model
        return sum(sims) / len(sims) if sims else 0.0


__all__ = ["UnlearningEvaluator"]
