# -*- coding: utf-8 -*-
r"""Span-level LLM unlearning trainers for the scope-aware PII benchmark (T/S/C/G evaluation).

Every trainer here consumes the batches produced by :mod:`torchunlearn.unlearn.llm_data`
(``SpanDataset`` + ``MergedLoaders({"Forget": ..., "Retain": ...})``):

    input_ids / attention_mask      window of the source document around one PII mention
    labels                          = input_ids on the mention tokens, -100 elsewhere  (loss only on the span)
    full_labels                     = input_ids on every real token                   (window-level retain loss)
    alt_input_ids/alt_labels        the same window with the mention replaced by an alternate string
                                    (DPO / FLAT "idk" answer; default "[REDACTED]")
    idx                             dataset index (used by the reference cache)

Loss formulas are ported verbatim from the reference implementations (file paths refer to the
upstream repos):

    GradAscent, GradDiff, NPO, SimNPO, DPO, WGA, SatImp, CEU, UNDIAL, PDU, RMU
        open-unlearning  src/trainer/unlearn/*.py, src/trainer/utils.py
    JensUn   JensUn-Unlearning src/trainer/utils.py (jensun_multitok_loss / jensun_retain_loss)
    FLAT     FLAT dataloader.py (ProbLossStable + get_contrastive_loss)

Reference model
---------------
Most methods need a frozen copy of the *pre-unlearning* model (NPO/DPO log-ratios, KL/JSD retain
terms, UNDIAL teacher, RMU retain activations).  Because unlearning always starts from that model,
``ref_mode="cache"`` (default) computes the needed reference quantities **once, at the start of
``fit``, with the not-yet-updated model** and stores only the masked positions
(seq-level NLL, per-position vocab distributions, or hidden states).  This is mathematically
identical to keeping a frozen copy but costs no extra GPU memory.  ``ref_mode="model"`` keeps a
deep copy instead (needed when ``retain_loss_on="window"`` with a KL/JSD retain loss).

T/S/C/G evaluation
------------------
``set_evaluator(TSCGEvaluator)`` records mean target log-prob per role at every record step
(``record_type="Epoch"``), stops early when the T log-prob/token drops below ``stop_lp``
(matched-forgetting rule of the pilot), and runs the full evaluation (log-prob + greedy extraction)
before and after ``fit`` into ``self.results``.
"""
from __future__ import annotations

import copy
import math
import re
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..trainer import Trainer
from ...nn.robmodel import RobModel


# ============================================================================ model adapter
class LLMRobModel(RobModel):
    """RobModel wrapper for a HF causal LM (forward passes **kwargs through)."""

    def __init__(self, model, tokenizer=None, device=None):
        nn.Module.__init__(self)
        if device is None:
            device = next(model.parameters()).device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.register_buffer("n_classes", torch.tensor(int(model.config.vocab_size)))
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def hf(self):
        return self.model

    def eval_accuracy(self, *a, **k):  # vision-only API
        raise NotImplementedError("use TSCGEvaluator for LLM evaluation")


def _find_module(model, regex):
    hits = {n: m for n, m in model.named_modules() if re.fullmatch(regex, n)}
    if len(hits) != 1:
        raise ValueError(f"module regex {regex!r} matched {list(hits)[:5]} (need exactly one)")
    return next(iter(hits.values()))


# ============================================================================ base trainer
class LLMUnlearner(Trainer):
    r"""Base class: loss helpers, reference cache, retain loss, T/S/C/G recording, early stop.

    Common hparams
        gamma, alpha            weights of forget / retain terms (subclasses)
        retain_loss_type        "NLL" | "KL" | "JSD" | "none"
        retain_loss_on          "span" (labels) | "window" (full_labels)
        ref_mode                "cache" | "model"
        ref_device              device for the deep-copied reference (ref_mode="model")
        trainable               regex over parameter names; others are frozen (None = all)
        grad_ckpt               enable HF gradient checkpointing
        stop_lp                 stop when mean T log-prob/token <= stop_lp (None = off)
        eval_every              check stop_lp every k iterations (0 = only at record steps)
    """
    needs_alt = False              # subclass needs alt_* fields (DPO, FLAT)
    ref_needs: tuple = ()          # subset of {"forget_seq_nll","alt_seq_nll","forget_tok_logits",
                                   #            "retain_tok_logprobs","retain_hidden"}

    def __init__(self, rmodel, gamma=1.0, alpha=1.0, retain_loss_type="NLL", retain_loss_on="span",
                 ref_mode="cache", ref_device=None, trainable=None, grad_ckpt=False,
                 stop_lp=None, eval_every=0, device=None):
        super().__init__(rmodel, device)
        assert isinstance(rmodel, LLMRobModel), "wrap the HF model with LLMRobModel"
        self.gamma, self.alpha = float(gamma), float(alpha)
        self.retain_loss_type, self.retain_loss_on = retain_loss_type, retain_loss_on
        self.ref_mode, self.ref_device = ref_mode, ref_device
        self.trainable, self.grad_ckpt = trainable, grad_ckpt
        self.stop_lp, self.eval_every = stop_lp, int(eval_every)
        self.evaluator = None
        self.results: Dict[str, dict] = {}
        self.history: List[dict] = []
        self._stop = False
        self._ref_model = None
        self._cache: Dict[str, Dict[int, dict]] = {"Forget": {}, "Retain": {}}
        if self.retain_loss_type in ("KL", "JSD") and "retain_tok_logprobs" not in self.ref_needs:
            self.ref_needs = tuple(self.ref_needs) + ("retain_tok_logprobs",)
            if self.retain_loss_on == "window" and self.ref_mode == "cache":
                # a full-vocab distribution per window token is too large to cache -> keep a frozen copy
                print(f"[{self.__class__.__name__}] retain_loss_on='window' + {self.retain_loss_type}: using ref_mode='model'")
                self.ref_mode = "model"

    # ------------------------------------------------------------------ setup
    def set_evaluator(self, evaluator):
        self.evaluator = evaluator
        return self

    def _apply_trainable(self):
        if self.trainable is None:
            for p in self.rmodel.model.parameters():
                p.requires_grad_(True)
        else:
            n_on = 0
            for n, p in self.rmodel.model.named_parameters():
                on = re.fullmatch(self.trainable, n) is not None
                p.requires_grad_(on)
                n_on += int(on)
            if n_on == 0:
                raise ValueError(f"trainable regex {self.trainable!r} matched no parameter")
        if self.grad_ckpt:
            self.rmodel.model.gradient_checkpointing_enable()
            if hasattr(self.rmodel.model, "enable_input_require_grads"):
                self.rmodel.model.enable_input_require_grads()

    def setup(self, optimizer="AdamW(lr=1e-5)", scheduler=None, scheduler_type=None, n_epochs=None,
              n_iters=None, minimizer=None, clip_grad_norm=1.0):
        self._apply_trainable()
        params = [p for p in self.rmodel.parameters() if p.requires_grad]
        self.optimizer = self.generate_optimizer(optimizer, params)
        self.scheduler, self.scheduler_type = self.generate_scheduler(
            scheduler, self.optimizer, scheduler_type, n_epochs, n_iters)
        self.minimizer = self.generate_minimizer(minimizer, self.rmodel, self.optimizer)
        self.clip_grad_norm = clip_grad_norm
        self.n_epochs, self.n_iters = n_epochs, n_iters

    # ------------------------------------------------------------------ fit
    def fit(self, train_loaders, n_epochs, n_iters=None, record_type="Epoch", save_path=None, save_type=None,
            save_best=None, save_overwrite=False, eval_before=True, eval_after=True, **kw):
        """One epoch = one pass over the Forget loader (retain batches are cycled), as in open-unlearning.
        MergedLoaders would otherwise define the epoch by the *longest* loader."""
        self._stop = False
        if n_iters is None and "Forget" in getattr(train_loaders, "loaders", {}):
            n_iters = len(train_loaders.loaders["Forget"])
        self._prepare_reference(train_loaders)
        if self.evaluator is not None and eval_before and "before" not in self.results:
            self.results["before"] = self.evaluator.evaluate(self.rmodel.model, quick=False)
            print(self.evaluator.table(self.results["before"], f"[{self.__class__.__name__}] before"))
        wrapped = _Stoppable(train_loaders, self)
        super().fit(wrapped, n_epochs, n_iters=n_iters, record_type=record_type, save_path=save_path,
                    save_type=save_type, save_best=save_best, save_overwrite=save_overwrite, **kw)
        if self.evaluator is not None and eval_after:
            self.rmodel.eval()
            self.results["after"] = self.evaluator.evaluate(self.rmodel.model, quick=False)
            print(self.evaluator.table(self.results["after"], f"[{self.__class__.__name__}] after"))
        self.results["stopped_early"] = self._stop
        self.results["iters"] = self.accumulated_iter
        return self

    # ------------------------------------------------------------------ reference
    def _dev(self):
        return self.device

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
        if self.retain_loss_on == "window" and ("retain_tok_logprobs" in needs):
            raise ValueError("retain_loss_on='window' with a KL/JSD retain loss needs ref_mode='model'")
        if self._cache["Forget"] or self._cache["Retain"]:
            return  # already built (second fit call)
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
    def _logits(self, batch, prefix="", model=None):
        model = model or self.rmodel.model
        dev = next(model.parameters()).device
        out = model(input_ids=batch[prefix + "input_ids"].to(dev),
                    attention_mask=batch[prefix + "attention_mask"].to(dev))
        return out.logits

    def _labels(self, batch, key):
        if key == "Retain" and self.retain_loss_on == "window":
            return batch["full_labels"]
        return batch["labels"]

    def _token_nll(self, logits, labels):
        """per-token NLL on shifted positions (B, L-1) zeroed outside labels, and the bool mask."""
        sl = logits[:, :-1].float()
        lab = labels[:, 1:].to(sl.device)
        mask = lab != -100
        nll = F.cross_entropy(sl.transpose(1, 2), lab.clamp(min=0), reduction="none") * mask
        return nll, mask

    def _hidden(self, batch, no_grad=False, prefix=""):
        """Output of self.act_module (set by RMU) at every position."""
        cache = []
        hd = self.act_module.register_forward_hook(
            lambda m, i, o: cache.append(o[0] if isinstance(o, tuple) else o))
        try:
            with torch.set_grad_enabled(not no_grad):
                self._logits(batch, prefix=prefix)
        finally:
            hd.remove()
        return cache[0]

    # ------------------------------------------------------------------ losses
    def forget_nll(self, batch):
        """HF-style mean CE over span tokens of the forget batch (and logits, nll, mask)."""
        logits = self._logits(batch)
        nll, mask = self._token_nll(logits, batch["labels"])
        return nll.sum() / mask.sum().clamp(min=1), logits, nll, mask

    def retain_loss(self, batch):
        if self.retain_loss_type == "none" or self.alpha == 0 or batch is None:
            return torch.zeros((), device=self.device)
        return self._retain_loss(batch)

    def _retain_loss(self, batch):
        """retain loss regardless of alpha (PDU's dual variable can reach 0 and must still see it)."""
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
        if self.retain_loss_type == "KL":       # KL(p_ref || q), open-unlearning compute_kl_divergence
            return F.kl_div(logq_m, logp_m, reduction="none", log_target=True).sum(-1).mean()
        if self.retain_loss_type == "JSD":      # jensun_retain_loss
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
        self.add_record_item("FGLoss", float(f_loss))
        self.add_record_item("RTLoss", float(r_loss))
        self.add_record_item("Cost", float(cost))
        if self.eval_every and self.stop_lp is not None and self.accumulated_iter % self.eval_every == 0:
            self._check_stop()
        return cost

    # ------------------------------------------------------------------ evaluation hooks
    @torch.no_grad()
    def _check_stop(self):
        if self.evaluator is None or self.stop_lp is None:
            return
        r = self.evaluator.evaluate(self.rmodel.model, quick=True, roles=["T"])
        self.rmodel.train()
        if r["T"]["lp_mean"] <= self.stop_lp:
            self._stop = True
            print(f"[{self.__class__.__name__}] stop: T lp/token {r['T']['lp_mean']:.3f} <= {self.stop_lp}")

    def record_during_eval(self):
        if self.evaluator is None:
            return
        # 궤적 평가 빈도: traj_every epoch 마다(기본 1). 마지막 epoch 은 항상 남긴다.
        # traj_every <= 0 이면 궤적 평가를 아예 하지 않는다 — 학습만 하고 최종 평가는 fit() 의
        # results["after"] 한 번으로 끝낸다. 평가가 run 시간의 90~100% 라 이게 가장 큰 절감이다.
        te = getattr(self, "traj_every", 1)
        if te is not None and te <= 0:
            for role, v in (getattr(self, "_last_traj", None) or {}).items():
                self.dict_record[f"{role}_lp"] = v
            return
        te = te or 1
        last = getattr(self, "n_epochs", None)
        skip = te > 1 and self.accumulated_epoch % te != 0 and self.accumulated_epoch != last
        if skip and getattr(self, "_last_traj", None) is not None:
            for role, v in self._last_traj.items():
                self.dict_record[f"{role}_lp"] = v
            return
        r = self.evaluator.evaluate(self.rmodel.model, quick=True)
        self._last_traj = {role: r[role]["lp_mean"] for role in ("T", "S", "C", "G") if role in r}
        row = {"epoch": self.accumulated_epoch, "iter": self.accumulated_iter}
        for role in ("T", "S", "C", "G"):
            if role in r:
                self.dict_record[f"{role}_lp"] = r[role]["lp_mean"]
                row[f"{role}_lp"] = r[role]["lp_mean"]
        self.history.append(row)
        if self.stop_lp is not None and "T" in r and r["T"]["lp_mean"] <= self.stop_lp:
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


# ============================================================================ classic
class GradAscent(LLMUnlearner):
    """Gradient ascent on the span tokens: loss = -CE (open-unlearning GradAscent)."""

    def __init__(self, rmodel, **kw):
        kw.setdefault("alpha", 0.0)
        kw.setdefault("retain_loss_type", "none")
        super().__init__(rmodel, **kw)

    def forget_loss(self, batch):
        ce, *_ = self.forget_nll(batch)
        return -ce


class GradDiff(LLMUnlearner):
    """gamma * (-CE_forget) + alpha * retain (NLL or KL vs reference)  (open-unlearning GradDiff)."""

    def forget_loss(self, batch):
        ce, *_ = self.forget_nll(batch)
        return -ce


class NPO(LLMUnlearner):
    """Negative Preference Optimization (Zhang et al., 2024).

    loss = -2/beta * logsigmoid(beta * (NLL_seq - NLL_seq_ref)).mean() + alpha * retain
    (open-unlearning compute_dpo_loss with win=None, lose=forget).
    """
    ref_needs = ("forget_seq_nll",)

    def __init__(self, rmodel, beta=0.1, **kw):
        super().__init__(rmodel, **kw)
        self.beta = float(beta)

    def forget_loss(self, batch):
        _, logits, nll, mask = self.forget_nll(batch)
        seq = nll.sum(-1)
        if self.ref_mode == "model":
            with torch.no_grad():
                rn, _ = self._token_nll(self._logits(batch, model=self._ref()), batch["labels"])
                ref = rn.sum(-1).to(seq.device)
        else:
            ref = torch.tensor(self._cached("Forget", batch, "seq_nll"), device=seq.device)
        lose_log_ratio = -(seq - ref)
        return -2 / self.beta * F.logsigmoid(self.beta * (0.0 - lose_log_ratio)).mean()


class SimNPO(LLMUnlearner):
    """SimNPO (Fan et al., 2024): reference-free NPO on length-normalised NLL.

    loss = -2/beta * logsigmoid(beta * (NLL_seq / n_tok - delta)).mean();  open-unlearning defaults
    beta=4.5, delta=0, gamma=0.125, alpha=1.
    """

    def __init__(self, rmodel, beta=4.5, delta=0.0, **kw):
        kw.setdefault("gamma", 0.125)
        super().__init__(rmodel, **kw)
        self.beta, self.delta = float(beta), float(delta)

    def forget_loss(self, batch):
        _, logits, nll, mask = self.forget_nll(batch)
        x = nll.sum(-1) / mask.sum(-1).clamp(min=1) - self.delta
        return -F.logsigmoid(self.beta * x).mean() * 2 / self.beta


class DPO(LLMUnlearner):
    """DPO unlearning with an alternate (win) answer replacing the span (default "[REDACTED]").

    loss = -2/beta * logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean() + alpha * retain
    (open-unlearning DPO / IdkDPO with the alternate answer given by ``SpanDataset(alt_text=...)``).
    """
    needs_alt = True
    ref_needs = ("forget_seq_nll", "alt_seq_nll")

    def __init__(self, rmodel, beta=0.1, **kw):
        super().__init__(rmodel, **kw)
        self.beta = float(beta)

    def _seq(self, batch, prefix, model=None):
        logits = self._logits(batch, prefix=prefix, model=model)
        nll, _ = self._token_nll(logits, batch[prefix + "labels"])
        return nll.sum(-1)

    def forget_loss(self, batch):
        lose = self._seq(batch, "")
        win = self._seq(batch, "alt_")
        if self.ref_mode == "model":
            with torch.no_grad():
                lose_ref = self._seq(batch, "", self._ref()).to(lose.device)
                win_ref = self._seq(batch, "alt_", self._ref()).to(lose.device)
        else:
            lose_ref = torch.tensor(self._cached("Forget", batch, "seq_nll"), device=lose.device)
            win_ref = torch.tensor(self._cached("Forget", batch, "alt_seq_nll"), device=lose.device)
        win_log_ratio = -(win - win_ref)
        lose_log_ratio = -(lose - lose_ref)
        return -2 / self.beta * F.logsigmoid(self.beta * (win_log_ratio - lose_log_ratio)).mean()


class AltPO(DPO):
    """Alternate Preference Optimization (Mekala et al., COLING 2025, arXiv 2409.13474).

    L = E_{y_a}[ L_DPO(y_a, y_f | x_f) ] + alpha * NLL(retain)   (no NLL term on the alternates)
    i.e. the DPO loss with *self-generated plausible alternates* as the win answer instead of a fixed
    string; open-unlearning runs it as ``trainer=DPO`` + ``QAwithAlternateDataset`` (beta 0.1, alpha 1).

    Needs a :class:`~torchunlearn.unlearn.llm_data.AltSpanDataset` (M alternates per span).  Paper
    Appendix B shows every (alternate, original) pair once per epoch over an M-times dataset; here one
    epoch stays one pass over the forget spans and the dataset rotates the alternate per epoch, so
    M epochs show every pair exactly once at the same compute as the other methods.  The reference
    NLL of all M alternates is cached up front (``alt_seq_nll_{k}``).
    """
    ref_needs = ("forget_seq_nll",)

    def fit(self, train_loaders, *a, **kw):
        ds = getattr(train_loaders, "loaders", train_loaders)["Forget"].dataset
        # batches are fetched lazily after Trainer increments accumulated_epoch (num_workers=0)
        ds.epoch_fn = lambda: max(self.accumulated_epoch - 1, 0)
        return super().fit(train_loaders, *a, **kw)

    @torch.no_grad()
    def _prepare_reference(self, train_loaders):
        super()._prepare_reference(train_loaders)
        if self.ref_mode == "model":
            return
        src = getattr(train_loaders, "loaders", train_loaders)["Forget"]
        ds = src.dataset
        if not hasattr(ds, "n_alt"):
            raise TypeError("AltPO needs an AltSpanDataset (per-span alternates)")
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

    def forget_loss(self, batch):
        lose = self._seq(batch, "")
        win = self._seq(batch, "alt_")
        if self.ref_mode == "model":
            with torch.no_grad():
                lose_ref = self._seq(batch, "", self._ref()).to(lose.device)
                win_ref = self._seq(batch, "alt_", self._ref()).to(lose.device)
        else:
            lose_ref = torch.tensor(self._cached("Forget", batch, "seq_nll"), device=lose.device)
            win_ref = torch.tensor([self._cache["Forget"][int(i)][f"alt_seq_nll_{int(k)}"]
                                    for i, k in zip(batch["idx"].tolist(), batch["alt_k"].tolist())], device=lose.device)
        win_log_ratio = -(win - win_ref)
        lose_log_ratio = -(lose - lose_ref)
        return -2 / self.beta * F.logsigmoid(self.beta * (win_log_ratio - lose_log_ratio)).mean()


# ============================================================================ recent (2025)
class WGA(LLMUnlearner):
    """Weighted Gradient Ascent (Wang et al., 2025): w = p^beta (detached); loss = -(w * CE).mean()."""

    def __init__(self, rmodel, beta=1.0, **kw):
        super().__init__(rmodel, **kw)
        self.beta = float(beta)

    def forget_loss(self, batch):
        _, logits, nll, mask = self.forget_nll(batch)
        w = ((-nll).exp().detach()) ** self.beta
        return -(w * nll)[mask].mean()


class SatImp(LLMUnlearner):
    """Saturation-Importance weighted GA (Wang et al., 2025).

    w = p^beta1 * (1-p)^beta2 (detached); loss = -(w * CE).mean();  open-unlearning defaults
    beta1=5, beta2=1, gamma=0.1, alpha=1.
    """

    def __init__(self, rmodel, beta1=5.0, beta2=1.0, **kw):
        kw.setdefault("gamma", 0.1)
        super().__init__(rmodel, **kw)
        self.beta1, self.beta2 = float(beta1), float(beta2)

    def forget_loss(self, batch):
        _, logits, nll, mask = self.forget_nll(batch)
        p = (-nll).exp().detach()
        w = (p ** self.beta1) * ((1 - p) ** self.beta2)
        return -(w * nll)[mask].mean()


class CEU(LLMUnlearner):
    """Cross-Entropy Unlearning (Wang, 2025): CE towards softmax(logits with the true token's logit
    set to -inf); the first ``ignore_first_n`` span tokens are skipped.  No retain term by default."""

    def __init__(self, rmodel, ignore_first_n=1, **kw):
        kw.setdefault("alpha", 0.0)
        kw.setdefault("retain_loss_type", "none")
        super().__init__(rmodel, **kw)
        self.ignore_first_n = int(ignore_first_n)

    def forget_loss(self, batch):
        logits = self._logits(batch)
        labels = batch["labels"].to(logits.device)
        valid = labels != -100
        drop = (valid.cumsum(-1) <= self.ignore_first_n) & valid
        labels = labels.masked_fill(drop, -100)
        sl, lab = logits[:, :-1].float(), labels[:, 1:]
        m = lab != -100
        vl, vlab = sl[m], lab[m]
        tgt = vl.detach().clone()
        tgt.scatter_(-1, vlab.unsqueeze(-1), float("-inf"))
        return F.cross_entropy(vl, F.softmax(tgt, -1))


class UNDIAL(LLMUnlearner):
    """UNDIAL (Dong et al., 2025): self-distillation with the teacher's logit on the true token
    reduced by beta; loss = CE(student, softmax(teacher - beta * onehot)) on span positions.
    open-unlearning defaults beta=10, alpha=0."""
    ref_needs = ("forget_tok_logits",)

    def __init__(self, rmodel, beta=10.0, **kw):
        kw.setdefault("alpha", 0.0)
        super().__init__(rmodel, **kw)
        self.beta = float(beta)

    def forget_loss(self, batch):
        logits = self._logits(batch)
        sl = logits[:, :-1].float()
        lab = batch["labels"][:, 1:].to(sl.device)
        m = lab != -100
        if self.ref_mode == "model":
            with torch.no_grad():
                t = self._logits(batch, model=self._ref())[:, :-1].float().to(sl.device)[m]
        else:
            t = torch.cat(self._cached("Forget", batch, "tok_logits")).to(sl.device).float()
        onehot = F.one_hot(lab[m], sl.shape[-1]).float()
        soft = F.softmax(t - onehot * self.beta, -1)
        return F.cross_entropy(sl[m], soft)


class PDU(LLMUnlearner):
    """Primal-Dual Unlearning (Entesari et al., NeurIPS 2025, arXiv 2506.05314; open-unlearning pdu.py).

    min L_f  s.t.  L_r <= eps, solved on the Lagrangian  gamma * L_f + lambda * (L_r - eps)  with
    lambda = ``alpha`` (initial value lambda_0):
        forget  L_f = (max logit - mean logit)^2 on span positions (unshifted mask, as upstream)
        retain  L_r = NLL on the retain batch
        dual    lambda <- max(0, lambda + dual_step_size * (L_r - eps))
                "step":  every batch, with that batch's pre-step retain loss (upstream compute_loss)
                "epoch": once per epoch, mean retain loss over the retain loader (eval mode, no grad)
        warm-up updates start after ``dual_warmup_epochs`` epochs (upstream DualOptimizationCallback)
    The retain loss is computed even when lambda reaches 0 (otherwise lambda could never recover).
    """

    def __init__(self, rmodel, primal_dual=False, dual_step_size=1.0, retain_loss_eps=0.0,
                 dual_update_upon="step", dual_warmup_epochs=0, **kw):
        super().__init__(rmodel, **kw)
        if dual_update_upon not in ("step", "epoch"):
            raise ValueError("dual_update_upon must be 'step' or 'epoch'")
        self.primal_dual, self.dual_step_size, self.retain_loss_eps = bool(primal_dual), float(dual_step_size), float(retain_loss_eps)
        self.dual_update_upon, self.dual_warmup_epochs = dual_update_upon, int(dual_warmup_epochs)
        self.alpha_init = self.alpha
        self.dual_trace: List[dict] = []
        self._retain_src = None

    def forget_loss(self, batch):
        logits = self._logits(batch).float()
        m = (batch["labels"].to(logits.device) != -100).reshape(-1)
        flat = logits.reshape(-1, logits.shape[-1])
        f = (flat.max(-1)[0] - flat.mean(-1)) ** 2
        return (f * m).sum() / m.sum().clamp(min=1)

    def _can_update(self):
        # upstream enables updates at the end of epoch `dual_warmup_epochs` -> epochs warmup+1, ...
        return self.primal_dual and (self.accumulated_epoch - 1) >= self.dual_warmup_epochs

    def fit(self, train_loaders, *a, **kw):
        self._retain_src = getattr(train_loaders, "loaders", {}).get("Retain")
        return super().fit(train_loaders, *a, **kw)

    def calculate_cost(self, train_data, reduction="mean"):
        f_loss = self.forget_loss(train_data["Forget"])
        rb = train_data.get("Retain")
        r_raw = self._retain_loss(rb) if rb is not None else torch.zeros((), device=self.device)
        r_loss = r_raw - self.retain_loss_eps
        cost = self.gamma * f_loss + self.alpha * r_loss
        if self.dual_update_upon == "step" and self._can_update():
            self.alpha = max(0.0, self.alpha + self.dual_step_size * float(r_loss))
        self.add_record_item("alpha", self.alpha)
        self.add_record_item("FGLoss", float(f_loss)); self.add_record_item("RTLoss", float(r_raw))
        self.add_record_item("Cost", float(cost))
        if self.eval_every and self.stop_lp is not None and self.accumulated_iter % self.eval_every == 0:
            self._check_stop()
        return cost

    @torch.no_grad()
    def _epoch_dual_update(self):
        src = self._retain_src
        if src is None:
            return
        self.rmodel.eval()
        dl = DataLoader(src.dataset, batch_size=src.batch_size, shuffle=False, collate_fn=src.collate_fn)
        tot, nb = 0.0, 0
        for b in dl:
            tot += float(self._retain_loss(b)); nb += 1
        self.alpha = max(0.0, self.alpha + self.dual_step_size * (tot / max(nb, 1) - self.retain_loss_eps))
        self.rmodel.train()

    def record_during_eval(self):
        # called once at the end of every epoch (record_type="Epoch")
        if self.dual_update_upon == "epoch" and self.primal_dual and self.accumulated_epoch >= self.dual_warmup_epochs:
            self._epoch_dual_update()
        self.dual_trace.append({"epoch": self.accumulated_epoch, "alpha": self.alpha})
        super().record_during_eval()


class JensUn(LLMUnlearner):
    """JensUn (2025): forget = JSD(model, one-hot cyclic target tokens) on span positions;
    retain = JSD(model, reference) on retain span positions (retain_loss_type="JSD").
    ``target_text`` is tokenised with the model's tokenizer (paper default "No way")."""

    def __init__(self, rmodel, target_text="No way", **kw):
        kw.setdefault("retain_loss_type", "JSD")
        super().__init__(rmodel, **kw)
        self.target_ids = rmodel.tokenizer(target_text, add_special_tokens=False)["input_ids"]

    def forget_loss(self, batch):
        logits = self._logits(batch)
        sl = logits[:, :-1].float()
        lab = batch["labels"][:, 1:].to(sl.device)
        mask = lab != -100
        eps = 1e-9
        pattern = torch.tensor(self.target_ids, device=sl.device)
        q = F.softmax(sl, -1).clamp(min=eps)
        total, n = 0.0, 0
        for i in range(sl.shape[0]):
            pos = torch.nonzero(mask[i], as_tuple=True)[0]
            if len(pos) == 0:
                continue
            cyc = pattern.repeat(math.ceil(len(pos) / len(pattern)))[:len(pos)]
            p = torch.full((len(pos), sl.shape[-1]), eps, device=sl.device)
            p.scatter_(1, cyc.unsqueeze(1), 1.0)
            qi = q[i, pos]
            m = 0.5 * (qi + p)
            jsd = 0.5 * ((qi * (qi.log() - m.log())).sum(-1) + (p * (p.log() - m.log())).sum(-1))
            total = total + jsd.sum(); n += len(pos)
        return total / max(n, 1)


class FLAT(LLMUnlearner):
    """FLAT (Wang et al., ICLR 2025): f-divergence contrastive loss between the probability of the
    alternate ("idk"/[REDACTED]) answer and of the forget answer (loss_type="cl" in the reference).
    loss = activation(-mean(p_good)) - conjugate(-mean(p_unlearn)); divergence default Total-Variation."""
    needs_alt = True
    DIVS = {
        "KL": (lambda x: -torch.mean(x), lambda x: -torch.mean(torch.exp(x - 1.0))),
        "Reverse-KL": (lambda x: -torch.mean(-torch.exp(x)), lambda x: -torch.mean(-1.0 - x)),
        "Jeffrey": (lambda x: -torch.mean(x), lambda x: -torch.mean(x + x * x / 4.0 + x * x * x / 16.0)),
        "Squared-Hellinger": (lambda x: -torch.mean(1.0 - torch.exp(x)), lambda x: -torch.mean((1.0 - torch.exp(x)) / torch.exp(x))),
        "Pearson": (lambda x: -torch.mean(x), lambda x: -torch.mean(x * x / 4.0 + x)),
        "Neyman": (lambda x: -torch.mean(1.0 - torch.exp(x)), lambda x: -torch.mean(2.0 - 2.0 * torch.sqrt(1.0 - x))),
        "Jenson-Shannon": (lambda x: -torch.mean(-torch.log(1.0 + torch.exp(-x))) - math.log(2.0),
                           lambda x: -torch.mean(x + torch.log(1.0 + torch.exp(-x))) + math.log(2.0)),
        "Total-Variation": (lambda x: -torch.mean(torch.tanh(x) / 2.0), lambda x: -torch.mean(torch.tanh(x) / 2.0)),
    }

    def __init__(self, rmodel, div="Total-Variation", **kw):
        kw.setdefault("alpha", 0.0)
        kw.setdefault("retain_loss_type", "none")
        super().__init__(rmodel, **kw)
        if div not in self.DIVS:
            raise ValueError(f"div must be one of {list(self.DIVS)}")
        self.div = div

    def _prob_loss(self, batch, prefix):
        """ProbLossStable: NLLLoss(ignore_index=-100) applied to softmax *probabilities* -> -p on
        labelled positions, 0 elsewhere; averaged over every (b, t) as in the reference loop."""
        logits = self._logits(batch, prefix=prefix)
        sl = logits[:, :-1].float()
        lab = batch[prefix + "labels"][:, 1:].to(sl.device)
        probs = F.softmax(sl, -1)
        return F.nll_loss(probs.transpose(1, 2), lab, ignore_index=-100, reduction="none").mean()

    def forget_loss(self, batch):
        loss_sum_unlearn = self._prob_loss(batch, "")
        loss_sum_good = self._prob_loss(batch, "alt_")
        activation, conjugate = self.DIVS[self.div]
        return activation(-loss_sum_good) - conjugate(-loss_sum_unlearn)


class RMU(LLMUnlearner):
    """Representation Misdirection for Unlearning (Li et al., 2024; open-unlearning port).

    forget: MSE(h_L(x), c * u) on span positions (u random unit vector, c = steering_coeff)
    retain: MSE(h_L(x), h_L^ref(x)) on retain span positions ("EMBED_DIFF")
    Only ``trainable`` parameters are updated (default: down_proj of layers L-2..L).
    """
    ref_needs = ("retain_hidden",)

    def __init__(self, rmodel, layer_id=7, steering_coeff=20.0, module_regex=None, trainable=None, **kw):
        kw.setdefault("retain_loss_type", "EMBED_DIFF")
        if trainable is None:
            ids = "|".join(str(i) for i in range(max(0, layer_id - 2), layer_id + 1))
            trainable = rf"model\.layers\.({ids})\.mlp\.down_proj\.weight"
        super().__init__(rmodel, trainable=trainable, **kw)
        self.layer_id, self.steering_coeff = int(layer_id), float(steering_coeff)
        self.act_module = _find_module(rmodel.model, module_regex or rf"model\.layers\.{layer_id}")
        self.control_vec = None

    def _control(self, dim, device, dtype):
        if self.control_vec is None:
            v = torch.rand(1, 1, dim)
            self.control_vec = v / v.norm() * self.steering_coeff
        return self.control_vec.to(device=device, dtype=dtype)

    @staticmethod
    def _act_loss(a, b, mask):
        sq = F.mse_loss(a, b, reduction="none")
        m = mask.unsqueeze(-1).expand_as(sq)
        s = (sq * m).mean(2).sum(1)
        return (s / mask.sum(-1, keepdim=True).squeeze(-1).clamp(min=1)).mean()

    def forget_loss(self, batch):
        h = self._hidden(batch).float()
        mask = (batch["labels"].to(h.device) != -100)
        return self._act_loss(h, self._control(h.shape[-1], h.device, h.dtype).expand_as(h), mask)

    def retain_loss(self, batch):
        if self.alpha == 0 or batch is None:
            return torch.zeros((), device=self.device)
        if self.retain_loss_type != "EMBED_DIFF":
            return super().retain_loss(batch)
        h = self._hidden(batch).float()
        mask = (self._labels(batch, "Retain").to(h.device) != -100)
        if self.ref_mode == "model":
            with torch.no_grad():
                ref = self._ref_hidden(batch).to(h.device).float()
        else:
            ref = torch.zeros_like(h)
            for j, r in enumerate(self._cached("Retain", batch, "hidden")):
                ref[j][mask[j]] = r.to(h.device).float()
        return self._act_loss(h, ref, mask)

    def _ref_hidden(self, batch):
        ref = self._ref()
        mod = _find_module(ref, rf"model\.layers\.{self.layer_id}")
        cache = []
        hd = mod.register_forward_hook(lambda m, i, o: cache.append(o[0] if isinstance(o, tuple) else o))
        try:
            self._logits(batch, model=ref)
        finally:
            hd.remove()
        return cache[0]


LLM_TRAINERS = OrderedDict([
    ("GradAscent", GradAscent), ("GradDiff", GradDiff), ("NPO", NPO), ("SimNPO", SimNPO), ("DPO", DPO),
    ("AltPO", AltPO), ("RMU", RMU), ("WGA", WGA), ("SatImp", SatImp), ("CEU", CEU), ("UNDIAL", UNDIAL), ("PDU", PDU),
    ("JensUn", JensUn), ("FLAT", FLAT),
])
