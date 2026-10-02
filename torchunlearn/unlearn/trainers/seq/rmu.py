"""RMU (Li et al., 2024; open-unlearning port): representation misdirection.

forget: MSE(h_L(x), c * u) on answer positions (u random unit vector, c = steering_coeff)
retain: MSE(h_L(x), h_L^ref(x)) on retain positions ("EMBED_DIFF")
Trainable = the MLP output projection of decoder layers L-2..L (found generically through
``SeqRobModel.decoder_layers()``, so it works for LLaVA / Qwen-VL / plain LLMs alike).
"""
import re

import torch
import torch.nn.functional as F

from .base import SeqUnlearner


class RMU(SeqUnlearner):
    """Representation misdirection at layer L (language-model blocks only)."""
    ref_needs = ("retain_hidden",)

    def __init__(self, rmodel, layer_id=7, steering_coeff=20.0, module_regex=None, trainable=None, **kw):
        kw.setdefault("retain_loss_type", "EMBED_DIFF")
        prefix, layers = rmodel.decoder_layers()
        if layer_id >= len(layers):
            raise ValueError(f"layer_id {layer_id} >= n_layers {len(layers)}")
        if trainable is None:
            names = rmodel.mlp_out_proj_names(range(max(0, layer_id - 2), layer_id + 1))
            trainable = "re:" + "|".join(re.escape(n) for n in names)
        super().__init__(rmodel, trainable=trainable, **kw)
        self.layer_id, self.steering_coeff = int(layer_id), float(steering_coeff)
        if module_regex:
            hits = {n: m for n, m in rmodel.model.named_modules() if re.fullmatch(module_regex, n)}
            if len(hits) != 1:
                raise ValueError(f"module_regex {module_regex!r} matched {list(hits)[:5]} (need exactly one)")
            self.act_module = next(iter(hits.values()))
        else:
            self.act_module = layers[layer_id]
        self._layer_prefix = prefix
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
        mask = (self._labels(batch, "Forget").to(h.device) != -100)
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
        mod = dict(ref.named_modules())[f"{self._layer_prefix}.{self.layer_id}"]
        return self._hidden(batch, no_grad=True, model=ref, module=mod)
