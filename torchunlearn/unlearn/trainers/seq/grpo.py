"""GRPO-style reinforcement unlearning (Group Relative Policy Optimization) for sequence models.

Used by ASRU (ICML 2026).  One training iteration on a batch of prompts:

    1. sample G rollouts per prompt from the current policy (greedy off, temperature T)
    2. score each rollout with ``reward_fn(sample, text) -> float``
    3. advantages  A_i = (R_i - mean_group) / (std_group + eps)      (plain mean-centring when ``norm_std=False``)
    4. loss = - mean_i [ A_i * log pi_theta(y_i | x) / |y_i| ]  +  kl_coef * KL(pi_theta || pi_ref)

The policy ratio is 1 on-policy (we take one gradient step per rollout batch), so the PPO clip is a no-op and is
omitted; the KL term uses the per-token estimator of GRPO (exp(r) - r - 1 with r = log pi_ref - log pi_theta).
The forget prompts come from the Forget loader and the boundary / retain prompts from the Retain loader, each
with its own reward (``reward_fn`` receives ``role``).
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from .base import SeqUnlearner


class GRPOUnlearner(SeqUnlearner):
    """Rule-based-reward GRPO.  Subclasses (or callers) provide ``reward_fn(sample, text, role) -> float``."""
    ref_needs = ()

    def __init__(self, rmodel, reward_fn: Optional[Callable] = None, group_size=4, max_new_tokens=32, temperature=1.0,
                 kl_coef=0.1, norm_std=True, **kw):
        kw["ref_mode"] = "model"
        kw.setdefault("retain_loss_type", "none")
        kw.setdefault("alpha", 1.0)
        super().__init__(rmodel, **kw)
        self.reward_fn = reward_fn or self.reward
        self.group_size, self.max_new_tokens, self.temperature = int(group_size), int(max_new_tokens), float(temperature)
        self.kl_coef, self.norm_std = float(kl_coef), bool(norm_std)
        self.collator = None

    def reward(self, sample, text: str, role: str) -> float:  # default: override in subclasses
        raise NotImplementedError

    def set_collator(self, collator):
        self.collator = collator
        return self

    # ------------------------------------------------------------------ rollouts
    @torch.no_grad()
    def _rollout(self, samples) -> List[List[str]]:
        tok = self.rmodel.tokenizer
        enc = self.collator.prompts(samples)
        dev = next(self.rmodel.model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items() if torch.is_tensor(v)}
        self.rmodel.model.eval()
        out = self.rmodel.model.generate(**enc, do_sample=True, temperature=self.temperature, top_p=1.0,
                                         max_new_tokens=self.max_new_tokens, num_return_sequences=self.group_size,
                                         pad_token_id=tok.pad_token_id, use_cache=True)
        self.rmodel.model.train()
        n_in = enc["input_ids"].shape[1]
        texts = tok.batch_decode(out[:, n_in:], skip_special_tokens=True)
        return [texts[i * self.group_size:(i + 1) * self.group_size] for i in range(len(samples))]

    def _policy_loss(self, samples, groups, role):
        """Build (prompt, rollout) training sequences through the collator, then the GRPO objective."""
        from ...seq_data import SeqDataset, SeqSample
        flat_samples, rewards, gid = [], [], []
        for i, (s, outs) in enumerate(zip(samples, groups)):
            for t in outs:
                flat_samples.append(SeqSample(prompt=s.prompt, answer=t if t.strip() else " ", images=list(s.images),
                                              group=s.group, id=s.id, meta=dict(s.meta)))
                rewards.append(float(self.reward_fn(s, t, role)))
                gid.append(i)
        rewards_t = torch.tensor(rewards)
        adv = torch.zeros_like(rewards_t)
        for i in range(len(samples)):
            m = torch.tensor(gid) == i
            r = rewards_t[m]
            a = r - r.mean()
            if self.norm_std:
                a = a / (r.std(unbiased=False) + 1e-6)
            adv[m] = a
        total, n = 0.0, 0
        bs = max(1, len(flat_samples) // max(1, len(samples)))      # one prompt's group per forward
        for i in range(0, len(flat_samples), bs):
            chunk = flat_samples[i:i + bs]
            batch = self.collator([SeqDataset(chunk)[j] for j in range(len(chunk))])
            logits = self._logits(batch)
            nll, mask = self._token_nll(logits, batch["answer_labels"])
            logp_seq = -(nll.sum(-1) / mask.sum(-1).clamp(min=1))             # mean log-prob per token
            a = adv[i:i + bs].to(logp_seq.device)
            pg = -(a * logp_seq).mean()
            with torch.no_grad():
                ref_logits = self._logits(batch, model=self._ref())
            ref_lp = -self._token_nll(ref_logits, batch["answer_labels"])[0].to(logits.device)
            lp = -nll
            r = ref_lp - lp
            kl = ((torch.exp(r) - r - 1.0) * mask).sum() / mask.sum().clamp(min=1)
            total = total + pg + self.kl_coef * kl
            n += 1
        self.add_record_item(f"R_{role}", float(rewards_t.mean()))
        return total / max(n, 1)

    def calculate_cost(self, train_data, reduction="mean"):
        if self.collator is None:
            raise ValueError("call set_collator(collator) before fit (GRPO needs to encode its own rollouts)")
        fb = train_data["Forget"]
        groups = self._rollout(fb["samples"])
        cost = self.gamma * self._policy_loss(fb["samples"], groups, "Forget")
        rb = train_data.get("Retain")
        if rb is not None and self.alpha != 0:
            groups_r = self._rollout(rb["samples"])
            cost = cost + self.alpha * self._policy_loss(rb["samples"], groups_r, "Retain")
        self.add_record_item("Cost", float(cost.detach()))
        return cost

    def forget_loss(self, batch):  # unused (calculate_cost is overridden)
        raise NotImplementedError
