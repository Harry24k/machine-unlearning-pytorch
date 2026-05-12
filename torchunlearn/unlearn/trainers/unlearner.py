import torch
from collections import OrderedDict

from ..trainer import Trainer
from ...utils import get_subloader


class Unlearner(Trainer):
    r"""Base class for unlearning trainers.

    Extends :class:`Trainer` with evaluation helpers that record clean and
    adversarial accuracy on Retain / Forget / Test loaders during unlearning.

    Functions:
        record_rob: register loaders + perturbation specs to evaluate at
            each record step.
        record_during_eval: actually compute the metrics. Called by the
            engine during ``fit``.
        get_grad: compute the flat gradient vector and per-layer gradients
            of the cross-entropy loss for a given batch (used for the
            retain/forget gradient cosine similarity).
    """

    def __init__(self, rmodel, device=None):
        super().__init__(rmodel, device)
        self.unlearn_dict = None
        self.record_cosine = False

    # ------------------------------------------------------------------ setup
    def record_rob(self, loaders, eps=None, alpha=None, steps=None, std=None,
                   record_cosine=None, n_limit=None):
        self._validate_pgd_args(alpha, steps)
        self.unlearn_dict = self._build_loader_dict(loaders, n_limit)
        self.unlearn_dict.update({"eps": eps, "alpha": alpha,
                                  "steps": steps, "std": std})
        self.record_cosine = bool(record_cosine)

    @staticmethod
    def _validate_pgd_args(alpha, steps):
        if (alpha is None) != (steps is None):
            raise ValueError("Both `alpha` and `steps` must be given for PGD.")

    @staticmethod
    def _build_loader_dict(loaders, n_limit):
        out = OrderedDict()
        out["loaders"] = {k: get_subloader(v, n_limit) for k, v in loaders.items()}
        return out

    # ------------------------------------------------------------ evaluation
    def record_during_eval(self):
        if self.unlearn_dict is None:
            return
        self._record_clean_and_adv_accuracy()
        if self.record_cosine:
            self._record_gradient_cosine()

    def _record_clean_and_adv_accuracy(self):
        eps = self.unlearn_dict.get("eps")
        alpha = self.unlearn_dict.get("alpha")
        steps = self.unlearn_dict.get("steps")
        std = self.unlearn_dict.get("std")
        for flag, loader in self.unlearn_dict["loaders"].items():
            self.dict_record["Clean" + flag] = self.rmodel.eval_accuracy(loader)
            if eps is not None:
                self.dict_record["FGSM" + flag] = self.rmodel.eval_rob_accuracy_fgsm(
                    loader, eps=eps, verbose=False)
                if steps is not None:
                    self.dict_record["PGD" + flag] = self.rmodel.eval_rob_accuracy_pgd(
                        loader, eps=eps, alpha=alpha, steps=steps, verbose=False)
            if std is not None:
                self.dict_record["GN" + flag] = self.rmodel.eval_rob_accuracy_gn(
                    loader, std=std, verbose=False)

    def _record_gradient_cosine(self):
        retain_grad, retain_layers = self._first_batch_grad("(R)")
        forget_grad, forget_layers = self._first_batch_grad("(F)")

        cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
        self.dict_record["Cosine"] = cosine(retain_grad, forget_grad).item()

        layer_cosines = [cosine(r, f).item()
                         for r, f in zip(retain_layers, forget_layers)]
        if layer_cosines:
            self.dict_record["Cosine(Layer)"] = sum(layer_cosines) / len(layer_cosines)

    def _first_batch_grad(self, flag):
        loader = self.unlearn_dict["loaders"][flag]
        for x, y in loader:
            return self.get_grad(x, y)
        raise RuntimeError(f"Loader for flag {flag!r} is empty.")

    # ---------------------------------------------------------- gradient util
    def get_grad(self, x, y):
        self.rmodel.eval()
        x, y = x.to(self.device), y.to(self.device)
        logits = self.rmodel(x)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        grads = torch.autograd.grad(loss, self.rmodel.parameters(),
                                    retain_graph=False, create_graph=False)
        per_layer = [g.contiguous().view(-1) for g in grads if g is not None]
        grad_vector = torch.cat(per_layer)
        return grad_vector, per_layer
