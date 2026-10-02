"""SeqRobModel -- one wrapper for every autoregressive HF model (text-only LLM or image+text LVLM).

The unlearning trainers in ``torchunlearn.unlearn.trainers.seq`` never look at the model class.  They only
rely on this surface:

    rmodel.model            the HF module (``forward(**batch)`` -> ``.logits``)
    rmodel.processor        AutoProcessor (image+text) or the tokenizer (text)
    rmodel.tokenizer        the text tokenizer in both cases
    rmodel.modality         "text" | "image_text"
    rmodel.image_token_ids  token ids that are image/video placeholders (loss must never target them)
    rmodel.text_input_keys  batch keys that belong to the *text* side (prefixed ``alt_`` for alternates)
    rmodel.decoder_layers() (name prefix, ModuleList) of the language-model transformer blocks
    rmodel.apply_trainable(spec)   "all" | "none" | "lm" | "vision" | "projector" | "lm+projector"
                                   | "lora:r=16,alpha=32,target=lm" | "re:<regex over parameter names>"

The family table and the processor fallback / verification logic were ported from
``/home1/irteam/mmforge/src/mmforge/models/generic_hf.py`` (2026-10-01).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from .robmodel import RobModel

# model_type -> modality.  Anything not listed falls back to the ``architectures`` heuristic below.
MODEL_TYPE_MODALITY: Dict[str, str] = {
    "llava": "image_text", "llava_next": "image_text", "llava_onevision": "image_text",
    "qwen2_vl": "image_text", "qwen2_5_vl": "image_text", "qwen3_vl": "image_text", "qwen3_5": "image_text",
    "idefics2": "image_text", "idefics3": "image_text", "smolvlm": "image_text",
    "internvl": "image_text", "internvl_chat": "image_text",
    "gemma3": "image_text", "paligemma": "image_text", "mllama": "image_text",
    "phi4_multimodal": "image_text", "pixtral": "image_text", "mistral3": "image_text",
    "aya_vision": "image_text", "glm4v": "image_text", "kimi_vl": "image_text",
    "instructblip": "enc_dec_vqa", "blip-2": "enc_dec_vqa", "blip_2": "enc_dec_vqa", "blip": "enc_dec_vqa",
    "clip": "dual_encoder", "siglip": "dual_encoder",
}

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "auto": "auto"}

VISION_RE = re.compile(r"(^|\.)(vision_tower|vision_model|visual|vision_encoder|image_encoder|vit)(\.|$)")
PROJECTOR_RE = re.compile(r"(^|\.)(multi_modal_projector|mm_projector|merger|connector|projector|aligner|"
                          r"vision_projection|image_projection|resampler|abstractor)(\.|$)")
TEXT_INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids", "position_ids", "labels")


class CapabilityError(RuntimeError):
    """The model cannot do what the caller asked.  We fail loudly instead of guessing."""


def resolve_modality(model_type: Optional[str], architectures: Optional[List[str]]) -> str:
    if model_type and model_type in MODEL_TYPE_MODALITY:
        return MODEL_TYPE_MODALITY[model_type]
    arch = " ".join(architectures or []).lower()
    if "conditionalgeneration" in arch or "imagetexttotext" in arch or "vision2seq" in arch:
        return "image_text"
    if "clipmodel" in arch or "siglip" in arch:
        return "dual_encoder"
    if "causallm" in arch or "lmhead" in arch:
        return "text"
    return "unknown"


def _load_processor(model_id, processor_id, cfg, modality, *, local_files_only=False):
    """AutoProcessor (image_text) / AutoTokenizer (text).  Fine-tuned checkpoints often miss the
    preprocessor files, so fall back to the base model named in the config -- loudly."""
    import logging
    from transformers import AutoProcessor, AutoTokenizer

    log = logging.getLogger("torchunlearn.seqmodel")
    loader = AutoProcessor if modality == "image_text" else AutoTokenizer
    candidates = [c for c in (processor_id, model_id, getattr(cfg, "_name_or_path", None)) if c]
    seen, errors = set(), []
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            proc = loader.from_pretrained(cand, local_files_only=local_files_only, trust_remote_code=True)
            if cand != model_id:
                log.warning(f"{model_id} has no processor/tokenizer files; using the one from {cand}. "
                            "Token alignment is verified below.")
            return proc, cand
        except Exception as e:  # noqa: BLE001
            errors.append(f"  {cand}: {type(e).__name__}: {str(e)[:160]}")
    raise CapabilityError(f"no processor/tokenizer found for {model_id}. Pass processor_id=<base model>.\n"
                          + "\n".join(errors))


def _verify_processor_matches(model, tokenizer, model_id, proc_src):
    if proc_src == model_id or tokenizer is None:
        return
    try:
        emb = model.get_input_embeddings().weight.shape[0]
    except Exception:  # noqa: BLE001
        return
    n_tok = len(tokenizer)
    if n_tok > emb:
        raise CapabilityError(f"tokenizer from {proc_src} has {n_tok} tokens but {model_id} embeds {emb} rows; "
                              "pass the right processor_id.")
    if emb - n_tok > 64:
        raise CapabilityError(f"tokenizer/embedding mismatch suspected: {emb} rows vs {n_tok} tokens "
                              f"(diff {emb - n_tok}); the checkpoint probably added tokens. Pass processor_id.")


class SeqRobModel(RobModel):
    """RobModel wrapper for autoregressive HF models (text or image+text)."""

    def __init__(self, model, processor, modality: Optional[str] = None, device=None, name: Optional[str] = None,
                 family: Optional[str] = None):
        nn.Module.__init__(self)
        if device is not None and getattr(model, "hf_device_map", None) is None:
            model = model.to(device)
        self.model = model
        self.processor = processor
        self.name = name or getattr(getattr(model, "config", None), "_name_or_path", model.__class__.__name__)
        cfg = getattr(model, "config", None)
        if modality is None or modality == "auto":
            modality = resolve_modality(getattr(cfg, "model_type", None), getattr(cfg, "architectures", None))
            if modality == "unknown":
                modality = "image_text" if hasattr(processor, "image_processor") else "text"
        self.modality = modality
        self.family = family or getattr(cfg, "model_type", "unknown")
        self.device = device or next(model.parameters()).device
        self.processor_source = None
        # buffers kept for RobModel API compatibility (vision-side helpers read them)
        vocab = int(getattr(getattr(cfg, "text_config", cfg), "vocab_size", 0) or 0)
        self.register_buffer("n_classes", torch.tensor(vocab))
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        tok = self.tokenizer
        if tok is not None and tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.image_token_ids = self._find_image_token_ids()

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path: str, modality: str = "auto", dtype: str = "bf16", device=None, device_map=None,
                        trainable: str = "all", processor_id: Optional[str] = None, attn_impl: Optional[str] = None,
                        local_files_only: bool = False, merge_lora: bool = True, trust_remote_code: bool = True,
                        **kw) -> "SeqRobModel":
        """Load any HF checkpoint (base, fine-tuned, or a PEFT adapter directory).

        modality  "auto" (from config) | "text" | "image_text"
        dtype     bf16 | fp16 | fp32 | auto
        device    e.g. "cuda:0" (whole model on one device); device_map="auto" for sharded loading
        trainable see :meth:`apply_trainable`
        """
        from transformers import AutoConfig
        adapter_cfg = os.path.join(path, "adapter_config.json")
        base_path = path
        if os.path.exists(adapter_cfg):
            import json
            base_path = json.load(open(adapter_cfg))["base_model_name_or_path"]
        cfg = AutoConfig.from_pretrained(base_path, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
        if modality == "auto":
            modality = resolve_modality(getattr(cfg, "model_type", None), getattr(cfg, "architectures", None))
        if modality not in ("text", "image_text"):
            raise CapabilityError(f"{path}: model_type={getattr(cfg, 'model_type', None)} resolved to {modality!r}; "
                                  "SeqRobModel supports autoregressive text / image_text models only. "
                                  "Add the model_type to MODEL_TYPE_MODALITY or pass modality= explicitly.")
        load_kw: Dict[str, Any] = {"local_files_only": local_files_only, "trust_remote_code": trust_remote_code}
        if DTYPES[dtype] != "auto":
            load_kw["dtype"] = DTYPES[dtype]
        else:
            load_kw["dtype"] = "auto"
        if device_map is not None:
            load_kw["device_map"] = device_map
        if attn_impl:
            load_kw["attn_implementation"] = attn_impl
        load_kw.update(kw)
        if modality == "image_text":
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(base_path, **load_kw)
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(base_path, **load_kw)
        if base_path != path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, path)
            if merge_lora:
                model = model.merge_and_unload()
        processor, proc_src = _load_processor(path if os.path.exists(os.path.join(path, "tokenizer_config.json"))
                                              or not os.path.isdir(path) else base_path,
                                              processor_id, cfg, modality, local_files_only=local_files_only)
        if device is None and device_map is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj = cls(model, processor, modality=modality, device=device, name=path)
        obj.processor_source = proc_src
        _verify_processor_matches(model, obj.tokenizer, path, proc_src)
        obj.apply_trainable(trainable)
        return obj

    # ------------------------------------------------------------------ surface
    @property
    def tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    @property
    def hf(self):
        return self.model

    @property
    def config(self):
        return self.model.config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, **kw):
        return self.model.generate(**kw)

    def get_device(self):
        return next(self.model.parameters()).device

    @property
    def text_input_keys(self) -> Tuple[str, ...]:
        return TEXT_INPUT_KEYS

    def _find_image_token_ids(self) -> List[int]:
        ids = set()
        cfg = self.model.config
        for attr in ("image_token_id", "image_token_index", "video_token_id", "video_token_index",
                     "image_start_token_id", "image_end_token_id", "boi_token_index", "eoi_token_index"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v >= 0:
                ids.add(v)
        tok = self.tokenizer
        if tok is not None:
            for name in ("image_token", "video_token", "boi_token", "eoi_token"):
                s = getattr(self.processor, name, None)
                if isinstance(s, str):
                    i = tok.convert_tokens_to_ids(s)
                    if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
                        ids.add(i)
        return sorted(ids)

    def has_chat_template(self) -> bool:
        return bool(getattr(self.processor, "chat_template", None) or getattr(self.tokenizer, "chat_template", None))

    # ------------------------------------------------------------------ parameter groups
    def component_of(self, param_name: str) -> str:
        if VISION_RE.search(param_name):
            return "vision"
        if PROJECTOR_RE.search(param_name):
            return "projector"
        return "lm"

    def component_names(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {"vision": [], "projector": [], "lm": []}
        for n, _ in self.model.named_parameters():
            out[self.component_of(n)].append(n)
        return out

    def apply_trainable(self, spec: Optional[str]) -> Dict[str, int]:
        """Set requires_grad.  Returns {component: n_trainable_params}.

        "all" | "none" | "lm" | "vision" | "projector" | "lm+projector" (any '+' combination)
        "lora:r=16,alpha=32,dropout=0.05,target=lm"  (target: lm | vision | projector | <module names joined by '|'>)
        "re:<regex>"  full-match regex over parameter names (same as the LLM module's ``trainable`` hparam)
        None -> leave requires_grad as it is.
        """
        if spec is None:
            return self.trainable_summary()
        if spec.startswith("lora"):
            self._apply_lora(spec)
            return self.trainable_summary()
        if spec.startswith("re:"):
            rx = re.compile(spec[3:])
            n_on = 0
            for n, p in self.model.named_parameters():
                on = rx.fullmatch(n) is not None
                p.requires_grad_(on)
                n_on += int(on)
            if n_on == 0:
                raise ValueError(f"trainable regex {spec[3:]!r} matched no parameter")
            return self.trainable_summary()
        if spec == "all":
            for p in self.model.parameters():
                p.requires_grad_(True)
            return self.trainable_summary()
        if spec == "none":
            for p in self.model.parameters():
                p.requires_grad_(False)
            return self.trainable_summary()
        comps = set(spec.split("+"))
        unknown = comps - {"lm", "vision", "projector"}
        if unknown:
            raise ValueError(f"unknown trainable spec {spec!r} (parts {sorted(unknown)})")
        if self.modality == "text" and comps != {"lm"}:
            raise CapabilityError(f"{self.name} is text-only; trainable={spec!r} has no vision/projector parameters")
        n_on = 0
        for n, p in self.model.named_parameters():
            on = self.component_of(n) in comps
            p.requires_grad_(on)
            n_on += int(on)
        if n_on == 0:
            raise CapabilityError(f"trainable={spec!r} matched no parameter of {self.name}; "
                                  f"component sizes: { {k: len(v) for k, v in self.component_names().items()} }")
        return self.trainable_summary()

    def _apply_lora(self, spec: str):
        from peft import LoraConfig, get_peft_model
        opts: Dict[str, Any] = {"r": 16, "alpha": 32, "dropout": 0.05, "target": "lm"}
        if ":" in spec:
            for kv in spec.split(":", 1)[1].split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    opts[k.strip()] = v.strip()
        target = str(opts["target"])
        if target in ("lm", "vision", "projector"):
            # every nn.Linear of that component -> module names (unique leaf names, as peft expects)
            names = set()
            for n, m in self.model.named_modules():
                if isinstance(m, nn.Linear) and self.component_of(n + ".weight") == target and "lm_head" not in n:
                    names.add(n)
            if not names:
                raise CapabilityError(f"LoRA target {target!r}: no nn.Linear found in that component")
            target_modules = sorted(names)
        else:
            target_modules = [t.strip() for t in target.split("|")]
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model = get_peft_model(self.model, LoraConfig(
            r=int(opts["r"]), lora_alpha=int(opts["alpha"]), lora_dropout=float(opts["dropout"]),
            target_modules=target_modules))

    def trainable_summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {"vision": 0, "projector": 0, "lm": 0}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                out[self.component_of(n)] += p.numel()
        return out

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.model.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ structure helpers
    def decoder_layers(self) -> Tuple[str, nn.ModuleList]:
        """(prefix, ModuleList) of the language model's transformer blocks (never the vision encoder)."""
        cands = []
        for n, m in self.model.named_modules():
            if isinstance(m, nn.ModuleList) and n.endswith("layers") and len(m) > 0 \
                    and hasattr(m[0], "mlp") and not VISION_RE.search(n + ".") and not PROJECTOR_RE.search(n + "."):
                cands.append((n, m))
        if not cands:
            raise CapabilityError(f"{self.name}: could not find the decoder layer list (ModuleList named *.layers "
                                  "whose blocks have .mlp). Pass module_regex / trainable explicitly.")
        if len(cands) > 1:
            pref = [c for c in cands if "language_model" in c[0] or "text_model" in c[0]]
            if len(pref) == 1:
                cands = pref
            else:
                raise CapabilityError(f"{self.name}: ambiguous decoder layer lists {[c[0] for c in cands]}")
        return cands[0]

    def mlp_out_proj_names(self, layer_ids) -> List[str]:
        """Parameter names of the last nn.Linear inside ``layers[i].mlp`` (down_proj / fc2 / c_proj ...)."""
        prefix, layers = self.decoder_layers()
        names = []
        for i in layer_ids:
            lin = [n for n, m in layers[i].mlp.named_modules() if isinstance(m, nn.Linear)]
            if not lin:
                raise CapabilityError(f"layer {i}: no nn.Linear in .mlp")
            names.append(f"{prefix}.{i}.mlp.{lin[-1]}.weight")
        return names

    # ------------------------------------------------------------------ persistence
    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    def save_dict(self, save_path):  # RobModel API -> HF format (a .pth of a 7B model is pointless)
        self.save_pretrained(save_path)

    def load_dict(self, save_path):
        raise NotImplementedError("use SeqRobModel.from_pretrained(path)")

    # vision-only API that must not be called silently
    def eval_accuracy(self, *a, **k):
        raise NotImplementedError("use torchunlearn.metrics.seq.SeqUnlearningEvaluator")

    def __repr__(self):
        t = self.trainable_summary()
        return (f"SeqRobModel(name={self.name!r}, modality={self.modality}, family={self.family}, "
                f"trainable={t}, image_token_ids={self.image_token_ids})")
