"""LUMoE -- Lifelong Unlearning with a Mixture of (LoRA) Experts (MLUBench, ICML 2026, arXiv 2606.12809).

Every unlearning request trains one switchable LoRA adapter ("expert") with Preference Optimisation (refusal
answers) on its forget set + retain set; the base weights are never touched, which keeps the vision-language
alignment intact across requests.  At inference a router decides which expert (if any) to activate:

    1. extract the entity mentioned by the multimodal input
    2. match it against the forget entities of the previous requests

The paper's router is GLM-4V-Plus; here ``router`` is pluggable.  The default ``NameMatchRouter`` matches
forget-entity names (``SeqSample.group`` / ``meta["entity"]``) against the prompt text, and a VLM-based router
can be given as any ``callable(sample) -> task name | None``.

    lumoe = LUMoE(rmodel, collator, lora="r=16,alpha=32")
    lumoe.add_request("A", forget_A, retain_A, n_epochs=3, optimizer="AdamW(lr=1e-4)")
    lumoe.add_request("B", forget_B, retain_B)
    lumoe.generate(samples)         # routed generation
    lumoe.evaluator_model()         # object with .generate(...) for the evaluators
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence

import torch

from .po import PO


class NameMatchRouter:
    def __init__(self):
        self.entities: Dict[str, List[str]] = {}

    def register(self, task: str, names: Sequence[str]):
        self.entities[task] = [n.lower() for n in names if n]

    def __call__(self, sample) -> Optional[str]:
        text = " ".join([sample.prompt or "", str(sample.meta.get("entity", "")), str(sample.group or "")]).lower()
        for task, names in self.entities.items():
            if any(re.search(r"\b" + re.escape(n) + r"\b", text) for n in names):
                return task
        return None


class LUMoE:
    def __init__(self, rmodel, collator, lora: str = "r=16,alpha=32,dropout=0.05", router: Optional[Callable] = None,
                 refusal: str = "I'm sorry, I cannot provide information about this."):
        from peft import LoraConfig, get_peft_model
        self.rmodel, self.collator = rmodel, collator
        self.router = router or NameMatchRouter()
        self.refusal = refusal
        opts = {"r": 16, "alpha": 32, "dropout": 0.05}
        for kv in lora.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                opts[k.strip()] = v.strip()
        names = sorted({n for n, m in rmodel.model.named_modules()
                        if isinstance(m, torch.nn.Linear) and rmodel.component_of(n + ".weight") == "lm" and "lm_head" not in n})
        self.lora_cfg = LoraConfig(r=int(opts["r"]), lora_alpha=int(opts["alpha"]), lora_dropout=float(opts["dropout"]),
                                   target_modules=names)
        self.tasks: List[str] = []
        self.history: Dict[str, dict] = {}
        self._peft = None

    # ------------------------------------------------------------------ experts
    def _ensure_peft(self, task):
        from peft import get_peft_model
        if self._peft is None:
            for p in self.rmodel.model.parameters():
                p.requires_grad_(False)
            self._peft = get_peft_model(self.rmodel.model, self.lora_cfg, adapter_name=task)
            self.rmodel.model = self._peft
        else:
            self._peft.add_adapter(task, self.lora_cfg)
        self._peft.set_adapter(task)
        for n, p in self._peft.named_parameters():
            p.requires_grad_(f".{task}." in n or n.endswith(f".{task}"))

    def add_request(self, task: str, forget: Sequence, retain: Sequence, n_epochs: int = 3,
                    optimizer: str = "AdamW(lr=1e-4)", batch_size: int = 4, evaluator=None, **po_kw):
        from ...seq_data import build_unlearn_loaders
        if task in self.tasks:
            raise ValueError(f"task {task!r} already unlearned")
        self._ensure_peft(task)
        names = sorted({s.meta.get("entity") or s.group for s in forget if (s.meta.get("entity") or s.group)})
        if hasattr(self.router, "register"):
            self.router.register(task, names)
        col = self.collator
        old_alt = col.alt_text
        col.alt_text = col.alt_text or self.refusal
        try:
            loaders = build_unlearn_loaders(forget, retain, col, batch_size=batch_size)
            trainer = PO(self.rmodel, trainable=None, **po_kw)
            if evaluator is not None:
                trainer.set_evaluator(evaluator)
            trainer.setup(optimizer=optimizer, n_epochs=n_epochs)
            trainer.fit(loaders, n_epochs=n_epochs)
        finally:
            col.alt_text = old_alt
        self.tasks.append(task)
        self.history[task] = {"n_forget": len(forget), "n_retain": len(retain), "entities": names,
                              "results": getattr(trainer, "results", {})}
        return trainer

    # ------------------------------------------------------------------ routing
    def route(self, sample) -> Optional[str]:
        t = self.router(sample)
        return t if t in self.tasks else None

    def _activate(self, task: Optional[str]):
        if self._peft is None:
            return
        if task is None:
            self._peft.disable_adapter_layers()
        else:
            self._peft.enable_adapter_layers()
            self._peft.set_adapter(task)

    @torch.no_grad()
    def generate(self, samples: Sequence, max_new_tokens: int = 32, batch_size: int = 8) -> List[str]:
        tok = self.rmodel.tokenizer
        dev = next(self.rmodel.model.parameters()).device
        out = [None] * len(samples)
        by_task: Dict[Optional[str], List[int]] = {}
        for i, s in enumerate(samples):
            by_task.setdefault(self.route(s), []).append(i)
        self.rmodel.model.eval()
        for task, idxs in by_task.items():
            self._activate(task)
            for j in range(0, len(idxs), batch_size):
                chunk = [samples[i] for i in idxs[j:j + batch_size]]
                enc = {k: v.to(dev) for k, v in self.collator.prompts(chunk).items() if torch.is_tensor(v)}
                gen = self.rmodel.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                                 pad_token_id=tok.pad_token_id, use_cache=True)
                texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for i, t in zip(idxs[j:j + batch_size], texts):
                    out[i] = t
        self._activate(None)
        return out

    def save(self, path: str):
        if self._peft is not None:
            self._peft.save_pretrained(path)   # one sub-directory per adapter
