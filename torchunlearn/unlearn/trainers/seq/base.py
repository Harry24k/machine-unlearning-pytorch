"""SeqUnlearner -- base class for token-level unlearning of autoregressive models (text or image+text).

Generalised from ``trainers/llm_pii.py::LLMUnlearner`` (2026-09-07).  Loss formulas are identical; what changed:

* the batch is a dict produced by :class:`~torchunlearn.unlearn.seq_data.SeqCollator`; every non-reserved key
  is forwarded to the model, so ``pixel_values`` / ``image_grid_thw`` / ``pixel_attention_mask`` ... pass through
  untouched.  Alternate-answer keys carry the ``alt_`` prefix on the *text* side only (the image is shared).
* ``retain_loss_on`` is "answer" | "full" (old names span / window still accepted).
* the evaluator is any object with ``evaluate(model, quick, roles) -> {role: {"lp_mean": ...}}`` and
  ``table(results, title)``; ``stop_lp`` watches ``stop_role`` (default "Forget").
* ``trainable`` accepts the SeqRobModel spec ("lm", "lora:...", "re:<regex>", ...) or a bare regex.
"""
from __future__ import annotations

import copy
import re
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...trainer import Trainer
from ....nn.seqmodel import SeqRobModel

RESERVED = {"idx", "alt_k", "labels", "answer_labels", "full_labels", "alt_labels", "samples"}


class SeqUnlearner(Trainer):
    r"""Base: forward helpers, reference cache, retain loss, evaluation hooks, early stop.

    Common hparams
        gamma, alpha            weights of forget / retain terms
        retain_loss_type        "NLL" | "KL" | "JSD" | "none" (RMU: "EMBED_DIFF")
        retain_loss_on          "answer" (labels) | "full" (full_labels)
        ref_mode                "cache" (reference statistics computed once from the initial model) | "model" (frozen copy)
        ref_device              device of the frozen copy (ref_mode="model")
        trainable               SeqRobModel.apply_trainable spec or regex (None = keep the model's current setting)
        grad_ckpt               HF gradient checkpointing
        stop_lp / stop_role     stop when mean log-prob/token of ``stop_role`` <= stop_lp
        eval_every              check stop_lp every k iterations (0 = only at record steps)
        traj_every              evaluate the trajectory every k epochs (<=0: never; final eval only)
    """
    needs_alt = False
    ref_needs: tuple = ()   # subset of {"forget_seq_nll","alt_seq_nll","forget_tok_logits","retain_tok_logprobs","retain_hidden"}

    def __init__(self, rmodel, gamma=1.0, alpha=1.0, retain_loss_type="NLL", retain_loss_on="answer",
                 ref_mode="cache", ref_device=None, trainable=None, grad_ckpt=False, stop_lp=None, stop_role="Forget",
                 eval_every=0, traj_every=1, device=None):
        super().__init__(rmodel, device)
        if not isinstance(rmodel, SeqRobModel):
            raise TypeError("wrap the HF model with torchunlearn.nn.SeqRobModel")
        self.gamma, self.alpha = float(gamma), float(alpha)
        self.retain_loss_type = retain_loss_type
        self.retain_loss_on = {"span": "answer", "window": "full"}.get(retain_loss_on, retain_loss_on)
        assert self.retain_loss_on in ("answer", "full")
        self.ref_mode, self.ref_device = ref_mode, ref_device
        self.trainable, self.grad_ckpt = trainable, grad_ckpt
        self.stop_lp, self.stop_role, self.eval_every, self.traj_every = stop_lp, stop_role, int(eval_every), traj_every
        self.evaluator = None
        self.results: Dict[str, dict] = {}
        self.history: List[dict] = []
        self._stop = False
        self._ref_model = None
        self._last_traj = None
        self._cache: Dict[str, Dict[int, dict]] = {"Forget": {}, "Retain": {}}
        if self.retain_loss_type in ("KL", "JSD") and "retain_tok_logprobs" not in self.ref_needs:
            self.ref_needs = tuple(self.ref_needs) + ("retain_tok_logprobs",)
            if self.retain_loss_on == "full" and self.ref_mode == "cache":
                print(f"[{self.__class__.__name__}] retain_loss_on='full' + {self.retain_loss_type}: using ref_mode='model'")
                self.ref_mode = "model"

    # ------------------------------------------------------------------ setup
    def set_evaluator(self, evaluator):
        self.evaluator = evaluator
        return self

    def _apply_trainable(self):
        if self.trainable is not None:
            spec = self.trainable
            if not (spec.startswith(("lora", "re:")) or spec in ("all", "none") or set(spec.split("+")) <= {"lm", "vision", "projector"}):
                spec = "re:" + spec     # bare regex (LLM module convention)
            self.rmodel.apply_trainable(spec)
        if not any(p.requires_grad for p in self.rmodel.model.parameters()):
            raise ValueError("no trainable parameter (trainable='none'?)")
        if self.grad_ckpt:
            # non-reentrant checkpointing: gradients flow to frozen-input blocks (RMU trains 3 matrices deep inside a
            # model whose embeddings do not require grad) and no enable_input_require_grads() hook is needed
            self.rmodel.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(self.rmodel.model, "config"):
            self.rmodel.model.config.use_cache = False

    def setup(self, optimizer="AdamW(lr=1e-5)", scheduler=None, scheduler_type=None, n_epochs=None, n_iters=None,
              minimizer=None, clip_grad_norm=1.0):
        self._apply_trainable()
        params = [p for p in self.rmodel.model.parameters() if p.requires_grad]
        self.optimizer = self.generate_optimizer(optimizer, params)
        self.scheduler, self.scheduler_type = self.generate_scheduler(scheduler, self.optimizer, scheduler_type,
                                                                      n_epochs, n_iters)
        self.minimizer = self.generate_minimizer(minimizer, self.rmodel, self.optimizer)
        self.clip_grad_norm = clip_grad_norm
        self.n_epochs, self.n_iters = n_epochs, n_iters
        return self

    # ------------------------------------------------------------------ fit
    def fit(self, train_loaders, n_epochs, n_iters=None, record_type="Epoch", save_path=None, save_type=None,
            save_best=None, save_overwrite=False, eval_before=True, eval_after=True, **kw):
        """One epoch = one pass over the Forget loader (retain batches are cycled)."""
        self._stop = False
        loaders = getattr(train_loaders, "loaders", None)
        if n_iters is None and loaders and "Forget" in loaders:
            n_iters = len(loaders["Forget"])
        self._prepare_reference(train_loaders)
        if self.evaluator is not None and eval_before and "before" not in self.results:
            self.rmodel.eval()
            self.results["before"] = self.evaluator.evaluate(self.rmodel, quick=False)
            print(self.evaluator.table(self.results["before"], f"[{self.__class__.__name__}] before"))
        wrapped = _Stoppable(train_loaders, self)
        super().fit(wrapped, n_epochs, n_iters=n_iters, record_type=record_type, save_path=save_path,
                    save_type=save_type, save_best=save_best, save_overwrite=save_overwrite, **kw)
        if self.evaluator is not None and eval_after:
            self.rmodel.eval()
            self.results["after"] = self.evaluator.evaluate(self.rmodel, quick=False)
            print(self.evaluator.table(self.results["after"], f"[{self.__class__.__name__}] after"))
        self.results["stopped_early"] = self._stop
        self.results["iters"] = self.accumulated_iter
        return self

    # ------------------------------------------------------------------ reference
    def _ref(self):
        if self._ref_model is None:
            dev = self.ref_device or self.device
            self._ref_model = copy.deepcopy(self.rmodel.model).to(dev).eval()
            for p in self._ref_model.parameters():
                p.requires_grad_(False)
        return self._ref_model

    @torch.no_grad()
    def _prepare_reference(self, train_loaders):
        needs = set(self.ref_needs)
        if not needs:
            return
        if self.ref_mode == "model":
            self._ref()
            return
        if self.retain_loss_on == "full" and "retain_tok_logprobs" in needs:
            raise ValueError("retain_loss_on='full' with a KL/JSD retain loss needs ref_mode='model'")
        if self._cache["Forget"] or self._cache["Retain"]:
            return
        loaders = getattr(train_loaders, "loaders", train_loaders)
        self.rmodel.eval()
        for key in ("Forget", "Retain"):
            if key not in loaders:
                continue
            want = [n for n in needs if n.startswith("forget" if key == "Forget" else "retain")
                    or (key == "Forget" and n == "alt_seq_nll")]
            if not want:
                continue
            src = loaders[key]
            dl = DataLoader(src.dataset, batch_size=src.batch_size, shuffle=False, collate_fn=src.collate_fn)
            for batch in dl:
                idxs = batch["idx"].tolist()
                if any(n in want for n in ("forget_seq_nll", "forget_tok_logits", "retain_tok_logprobs")):
                    logits = self._logits(batch)
                    labels = self._labels(batch, key)
                    nll, mask = self._token_nll(logits, labels)
                    for j, i in enumerate(idxs):
                        d = self._cache[key].setdefault(i, {})
                        if "forget_seq_nll" in want:
                            d["seq_nll"] = nll[j].sum().item()
                        if "forget_tok_logits" in want:
                            d["tok_logits"] = logits[j, :-1][mask[j]].detach().to(torch.float16).cpu()
                        if "retain_tok_logprobs" in want:
                            d["tok_logprobs"] = F.log_softmax(logits[j, :-1][mask[j]].float(), -1).to(torch.float16).cpu()
                if "alt_seq_nll" in want:
                    if "alt_input_ids" not in batch:
                        raise ValueError(f"{self.__class__.__name__} needs alternate answers: give SeqCollator(alt_text=...) "
                                         "or SeqSample.alt_answers")
                    logits = self._logits(batch, prefix="alt_")
                    nll, mask = self._token_nll(logits, batch["alt_labels"])
                    for j, i in enumerate(idxs):
                        self._cache[key].setdefault(i, {})["alt_seq_nll"] = nll[j].sum().item()
                if "retain_hidden" in want:
                    h = self._hidden(batch, no_grad=True)
                    m = (self._labels(batch, key) != -100)
                    for j, i in enumerate(idxs):
                        self._cache[key].setdefault(i, {})["hidden"] = h[j][m[j]].detach().to(torch.float16).cpu()
        self.rmodel.train()

    def _cached(self, key, batch, field):
        return [self._cache[key][int(i)][field] for i in batch["idx"].tolist()]

    # ------------------------------------------------------------------ forward helpers
    def model_inputs(self, batch, prefix=""):
        """Everything the model should see: text keys with ``prefix`` + all shared (image) keys."""
        text_keys = set(self.rmodel.text_input_keys)
        out = {}
        for k, v in batch.items():
            if k in RESERVED or k == "offset_mapping" or k.startswith("alt_") or k.startswith("_"):
                continue
            if k in text_keys:
                continue
            out[k] = v
        for k in text_keys:
            if k == "labels":
                continue
            if prefix + k in batch:
                out[k] = batch[prefix + k]
        return out

    def _logits(self, batch, prefix="", model=None):
        model = model or self.rmodel.model
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in self.model_inputs(batch, prefix).items()}
        return model(**inputs, use_cache=False).logits

    def _labels(self, batch, key):
        if key == "Retain" and self.retain_loss_on == "full":
            return batch["full_labels"]
        return batch["answer_labels"] if "answer_labels" in batch else batch["labels"]

    def _token_nll(self, logits, labels):
        """per-token NLL on shifted positions (B, L-1), zero outside labels, plus the bool mask."""
        sl = logits[:, :-1].float()
        lab = labels[:, 1:].to(sl.device)
        mask = lab != -100
        nll = F.cross_entropy(sl.transpose(1, 2), lab.clamp(min=0), reduction="none") * mask
        return nll, mask

    def _hidden(self, batch, no_grad=False, prefix="", model=None, module=None):
        """Output of ``module`` (default self.act_module, set by RMU) at every position."""
        module = module or self.act_module
        cache = []
        hd = module.register_forward_hook(lambda m, i, o: cache.append(o[0] if isinstance(o, tuple) else o))
        try:
            with torch.set_grad_enabled(not no_grad):
                self._logits(batch, prefix=prefix, model=model)
        finally:
            hd.remove()
        return cache[0]

    # ------------------------------------------------------------------ losses
    def forget_nll(self, batch):
        """mean CE over the labelled forget tokens (and logits, per-token nll, mask)."""
        logits = self._logits(batch)
        nll, mask = self._token_nll(logits, self._labels(batch, "Forget"))
        return nll.sum() / mask.sum().clamp(min=1), logits, nll, mask

    def retain_loss(self, batch):
        if self.retain_loss_type == "none" or self.alpha == 0 or batch is None:
            return torch.zeros((), device=self.device)
        return self._retain_loss(batch)

    def _retain_loss(self, batch):
        labels = self._labels(batch, "Retain")
        logits = self._logits(batch)
        nll, mask = self._token_nll(logits, labels)
        if self.retain_loss_type == "NLL":
            return nll.sum() / mask.sum().clamp(min=1)
        logq = F.log_softmax(logits[:, :-1].float(), -1)
        if self.ref_mode == "model":
            with torch.no_grad():
                logp = F.log_softmax(self._logits(batch, model=self._ref())[:, :-1].float(), -1).to(logq.device)
            logq_m, logp_m = logq[mask], logp[mask]
        else:
            logq_m = logq[mask]
            logp_m = torch.cat(self._cached("Retain", batch, "tok_logprobs")).to(logq.device).float()
        if self.retain_loss_type == "KL":
            return F.kl_div(logq_m, logp_m, reduction="none", log_target=True).sum(-1).mean()
        if self.retain_loss_type == "JSD":
            p, q = logp_m.exp(), logq_m.exp()
            log_m = (0.5 * (p + q)).clamp(min=1e-10).log()
            return (0.5 * ((p * (logp_m - log_m)).sum(-1) + (q * (logq_m - log_m)).sum(-1))).mean()
        raise NotImplementedError(self.retain_loss_type)

    def forget_loss(self, batch):
        raise NotImplementedError

    def calculate_cost(self, train_data, reduction="mean"):
        fb = train_data["Forget"]
        rb = train_data.get("Retain")
        f_loss = self.forget_loss(fb)
        r_loss = self.retain_loss(rb) if self.alpha != 0 else torch.zeros((), device=self.device)
        cost = self.gamma * f_loss + self.alpha * r_loss
        self.add_record_item("FGLoss", float(f_loss.detach()))
        self.add_record_item("RTLoss", float(r_loss.detach()))
        self.add_record_item("Cost", float(cost.detach()))
        if self.eval_every and self.stop_lp is not None and self.accumulated_iter % self.eval_every == 0:
            self._check_stop()
        return cost

    # ------------------------------------------------------------------ evaluation hooks
    @torch.no_grad()
    def _check_stop(self):
        if self.evaluator is None or self.stop_lp is None:
            return
        r = self.evaluator.evaluate(self.rmodel, quick=True, roles=[self.stop_role])
        self.rmodel.train()
        if self.stop_role in r and r[self.stop_role]["lp_mean"] <= self.stop_lp:
            self._stop = True
            print(f"[{self.__class__.__name__}] stop: {self.stop_role} lp/token {r[self.stop_role]['lp_mean']:.3f} <= {self.stop_lp}")

    def record_during_eval(self):
        if self.evaluator is None:
            return
        te = self.traj_every
        if te is not None and te <= 0:
            for role, v in (self._last_traj or {}).items():
                self.dict_record[f"{role}_lp"] = v
            return
        te = te or 1
        last = getattr(self, "n_epochs", None)
        skip = te > 1 and self.accumulated_epoch % te != 0 and self.accumulated_epoch != last
        if skip and self._last_traj is not None:
            for role, v in self._last_traj.items():
                self.dict_record[f"{role}_lp"] = v
            return
        r = self.evaluator.evaluate(self.rmodel, quick=True)
        self._last_traj = {role: v["lp_mean"] for role, v in r.items() if "lp_mean" in v}
        row = {"epoch": self.accumulated_epoch, "iter": self.accumulated_iter}
        for role, v in self._last_traj.items():
            self.dict_record[f"{role}_lp"] = v
            row[f"{role}_lp"] = v
        self.history.append(row)
        if self.stop_lp is not None and self.stop_role in self._last_traj and self._last_traj[self.stop_role] <= self.stop_lp:
            self._stop = True


class _Stoppable:
    def __init__(self, loaders, trainer):
        self.loaders, self.trainer = loaders, trainer

    def __len__(self):
        return len(self.loaders)

    def __iter__(self):
        for b in self.loaders:
            if self.trainer._stop:
                return
            yield b
