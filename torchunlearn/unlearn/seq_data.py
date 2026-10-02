"""Data layer for sequence (text / image+text) unlearning.

Standard sample ---------------------------------------------------------------------------------
Every dataset recipe converts its native format into :class:`SeqSample`:

    SeqSample(prompt, answer, images=[PIL | path, ...], group="subject-7", alt_answers=[...], meta={})

``group`` is the forgetting unit (a person, a class, an image id ...); :func:`split_forget_retain` cuts by it.

Collation ---------------------------------------------------------------------------------------
:class:`SeqCollator` turns a list of samples into one model batch:

    input_ids, attention_mask (+ pixel_values, image_grid_thw, ... whatever the processor emits)
    labels        = input_ids on the *answer* tokens, -100 elsewhere   (loss_on="answer")
    full_labels   = input_ids on every real text token (never image tokens), -100 elsewhere
    alt_input_ids / alt_attention_mask / alt_labels   when an alternate answer exists (DPO/AltPO/FLAT)
    idx           dataset index (used by the reference cache)
    alt_k         which alternate was used (AltPO rotation)

The answer mask is computed with the chat template applied twice (prompt-only vs prompt+answer); the answer
tokens are the suffix.  Image tokens are all on the prompt side, so no model-specific image-token counting
is needed (ported from mmforge).  Without a chat template, ``template="plain"`` concatenates
``prompt + sep + answer`` and uses the tokenizer's offset mapping (token boundaries respected).
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

IGNORE = -100


# ================================================================================ sample schema
@dataclass
class SeqSample:
    prompt: str
    answer: Optional[str] = None
    images: List[Any] = field(default_factory=list)       # PIL.Image or file paths (loaded lazily)
    group: Optional[str] = None                            # forgetting unit (subject / class / image id)
    alt_answers: List[str] = field(default_factory=list)   # alternates for DPO / AltPO / FLAT
    id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_multimodal(self) -> bool:
        return len(self.images) > 0

    def load_images(self, image_root: Optional[str] = None):
        from PIL import Image
        out = []
        for im in self.images:
            if isinstance(im, str):
                p = im if (os.path.isabs(im) or image_root is None) else os.path.join(image_root, im)
                out.append(Image.open(p).convert("RGB"))
            else:
                out.append(im if im.mode == "RGB" else im.convert("RGB"))
        return out

    def without_images(self) -> "SeqSample":
        """Text-only copy (cross-modal probe: does the answer leak without the image?)."""
        return SeqSample(self.prompt, self.answer, [], self.group, list(self.alt_answers), self.id, dict(self.meta))


def strip_images(samples: Iterable[SeqSample]) -> List[SeqSample]:
    return [s.without_images() for s in samples]


# ================================================================================ recipes
def from_records(rows: Iterable[dict], fields: Optional[Dict[str, str]] = None, image_root: Optional[str] = None,
                 ) -> List[SeqSample]:
    """Generic mapping.  fields maps SeqSample attribute -> record key, e.g.
    {"prompt": "question", "answer": "answer", "images": "image", "group": "subject", "alt_answers": "alts"}."""
    f = {"prompt": "prompt", "answer": "answer", "images": "images", "group": "group", "alt_answers": "alt_answers",
         "id": "id"}
    f.update(fields or {})
    out = []
    for i, r in enumerate(rows):
        imgs = r.get(f["images"], [])
        if imgs is None:
            imgs = []
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        imgs = [os.path.join(image_root, im) if (isinstance(im, str) and image_root and not os.path.isabs(im)) else im
                for im in imgs]
        alts = r.get(f["alt_answers"], []) or []
        if isinstance(alts, str):
            alts = [alts]
        out.append(SeqSample(prompt=str(r[f["prompt"]]), answer=None if r.get(f["answer"]) is None else str(r[f["answer"]]),
                             images=list(imgs), group=None if r.get(f["group"]) is None else str(r[f["group"]]),
                             alt_answers=list(alts), id=str(r.get(f["id"], i)),
                             meta={k: v for k, v in r.items() if k not in set(f.values())}))
    return out


def from_jsonl(path: str, fields: Optional[Dict[str, str]] = None, image_root: Optional[str] = None) -> List[SeqSample]:
    rows = []
    with open(path) as fh:
        if path.endswith(".json"):
            data = json.load(fh)
            rows = data if isinstance(data, list) else data["data"]
        else:
            rows = [json.loads(l) for l in fh if l.strip()]
    return from_records(rows, fields, image_root)


def from_llava_json(path: str, image_root: Optional[str] = None, group_key: Optional[str] = None,
                    turn: str = "first") -> List[SeqSample]:
    """LLaVA-style ``conversations`` JSON (VLGuard / MLLMU exports):
    {"id", "image", "conversations": [{"from": "human", "value": "<image>\\nQ"}, {"from": "gpt", "value": "A"}, ...]}
    turn="first" -> one sample per record (first human/gpt pair); "all" -> one sample per pair (same image)."""
    data = json.load(open(path))
    out = []
    for r in data:
        conv = r["conversations"]
        pairs = [(conv[i]["value"], conv[i + 1]["value"]) for i in range(0, len(conv) - 1, 2)
                 if conv[i]["from"] in ("human", "user") and conv[i + 1]["from"] in ("gpt", "assistant")]
        if turn == "first":
            pairs = pairs[:1]
        img = r.get("image")
        imgs = [img] if img else []
        if image_root and imgs and not os.path.isabs(imgs[0]):
            imgs = [os.path.join(image_root, imgs[0])]
        for j, (q, a) in enumerate(pairs):
            q = q.replace("<image>", "").strip()
            out.append(SeqSample(prompt=q, answer=a, images=list(imgs),
                                 group=str(r[group_key]) if group_key and group_key in r else str(r.get("id", len(out))),
                                 id=f"{r.get('id', len(out))}#{j}", meta={k: v for k, v in r.items() if k != "conversations"}))
    return out


def from_hf(name: str, split: str = "train", fields: Optional[Dict[str, str]] = None, config: Optional[str] = None,
            n_limit: Optional[int] = None, **load_kw) -> List[SeqSample]:
    """HF datasets.  Image columns are already PIL objects.  fields as in :func:`from_records`."""
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, **load_kw)
    if n_limit:
        ds = ds.select(range(min(n_limit, len(ds))))
    f = fields or {}
    rows = []
    for r in ds:
        rows.append(dict(r))
    return from_records(rows, f)


def load_spec(spec: str, fields: Optional[Dict[str, str]] = None, image_root: Optional[str] = None, **kw) -> List[SeqSample]:
    """String spec used by the CLI:
        jsonl:<path>                 (.jsonl or .json list; fields via --fields)
        llava:<path.json>            (LLaVA conversations)
        hf:<name>[:<split>[:<config>]]
    """
    kind, _, rest = spec.partition(":")
    if kind == "jsonl":
        return from_jsonl(rest, fields, image_root)
    if kind == "llava":
        return from_llava_json(rest, image_root, **kw)
    if kind == "hf":
        parts = rest.split(":")
        return from_hf(parts[0], parts[1] if len(parts) > 1 else "train", fields,
                       parts[2] if len(parts) > 2 else None, **kw)
    raise ValueError(f"unknown data spec {spec!r} (jsonl: | llava: | hf:)")


# ================================================================================ splits
def split_forget_retain(samples: Sequence[SeqSample], by: str = "random", ratio: float = 0.1,
                        groups: Optional[Iterable[str]] = None, ids: Optional[Iterable[str]] = None,
                        seed: int = 0) -> Tuple[List[SeqSample], List[SeqSample]]:
    """by="random": ratio of samples;  by="group": ratio of groups (or the explicit ``groups``);
    by="ids": explicit sample ids.  Returns (forget, retain)."""
    rng = random.Random(seed)
    if by == "random":
        idx = list(range(len(samples)))
        rng.shuffle(idx)
        k = max(1, int(round(ratio * len(samples))))
        fset = set(idx[:k])
        return [s for i, s in enumerate(samples) if i in fset], [s for i, s in enumerate(samples) if i not in fset]
    if by == "group":
        all_groups = sorted({s.group for s in samples if s.group is not None})
        if not all_groups:
            raise ValueError("split by group needs SeqSample.group")
        if groups is None:
            rng.shuffle(all_groups)
            groups = all_groups[:max(1, int(round(ratio * len(all_groups))))]
        gset = set(map(str, groups))
        missing = gset - set(all_groups)
        if missing:
            raise ValueError(f"unknown groups {sorted(missing)[:5]}")
        return [s for s in samples if s.group in gset], [s for s in samples if s.group not in gset]
    if by == "ids":
        iset = set(map(str, ids or []))
        return [s for s in samples if s.id in iset], [s for s in samples if s.id not in iset]
    raise ValueError(by)


# ================================================================================ dataset / collator
class SeqDataset(Dataset):
    """Holds samples; the collator does the tokenisation.  ``alt_k`` rotates alternates per epoch (AltPO)."""

    def __init__(self, samples: Sequence[SeqSample]):
        self.samples = list(samples)
        self.epoch_fn: Optional[Callable[[], int]] = None
        self.fixed_alt: Optional[int] = None

    @property
    def n_alt(self) -> int:
        return min((len(s.alt_answers) for s in self.samples), default=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        n = len(s.alt_answers)
        if n == 0:
            k = 0
        elif self.fixed_alt is not None:
            k = self.fixed_alt % n
        else:
            k = ((self.epoch_fn() if self.epoch_fn else 0) + i) % n
        return {"idx": i, "sample": s, "alt_k": k}


class SeqCollator:
    """Tokenise a list of samples for a :class:`~torchunlearn.nn.seqmodel.SeqRobModel`.

    loss_on       "answer" (labels on answer tokens) | "full" (labels on all text tokens)
    template      "auto" (chat template if the processor has one, else plain) | "chat" | "plain"
    alt_text      fixed alternate answer for every sample (e.g. "[REDACTED]" / "I don't know") when the
                  sample has no ``alt_answers`` of its own
    system        optional system prompt for chat templates
    max_length    text-only truncation (never applied when images are present -- would cut image tokens)
    """

    def __init__(self, rmodel, loss_on: str = "answer", template: str = "auto", alt_text: Optional[str] = None,
                 system: Optional[str] = None, max_length: Optional[int] = None, sep: str = " ",
                 image_root: Optional[str] = None, add_eos: bool = True):
        self.rmodel = rmodel
        self.processor = rmodel.processor
        self.tok = rmodel.tokenizer
        self.modality = rmodel.modality
        self.loss_on = {"span": "answer", "window": "full"}.get(loss_on, loss_on)
        assert self.loss_on in ("answer", "full")
        if template == "auto":
            template = "chat" if rmodel.has_chat_template() else "plain"
        if template == "chat" and not rmodel.has_chat_template():
            raise ValueError(f"{rmodel.name} has no chat template; use template='plain' or set one on the tokenizer")
        if template == "plain" and self.modality == "image_text":
            raise ValueError("plain template is text-only; image+text models need a chat template "
                             "(the processor must expand image tokens)")
        self.template = template
        self.alt_text, self.system, self.max_length, self.sep = alt_text, system, max_length, sep
        self.image_root, self.add_eos = image_root, add_eos
        self.image_token_ids = torch.tensor(rmodel.image_token_ids, dtype=torch.long)

    # ---------------------------------------------------------------- text building
    def _messages(self, s: SeqSample, answer: Optional[str]):
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        if self.modality == "image_text":
            content = [{"type": "image"} for _ in s.images] + [{"type": "text", "text": s.prompt}]
        else:
            content = s.prompt
        msgs.append({"role": "user", "content": content})
        if answer is not None:
            msgs.append({"role": "assistant",
                         "content": [{"type": "text", "text": answer}] if self.modality == "image_text" else answer})
        return msgs

    def render(self, s: SeqSample, answer: Optional[str]) -> str:
        if self.template == "chat":
            src = self.processor if getattr(self.processor, "chat_template", None) else self.tok
            txt = src.apply_chat_template(self._messages(s, answer), add_generation_prompt=answer is None, tokenize=False)
            if answer is not None and self.add_eos and self.tok.eos_token and not txt.rstrip().endswith(self.tok.eos_token):
                pass  # chat templates emit their own end-of-turn token; do not double it
            return txt
        txt = s.prompt + self.sep
        if answer is not None:
            txt += answer + (self.tok.eos_token if (self.add_eos and self.tok.eos_token) else "")
        return txt

    # ---------------------------------------------------------------- processor calls
    def encode(self, samples: Sequence[SeqSample], answers: Sequence[Optional[str]], padding_side: str = "right"
               ) -> Dict[str, torch.Tensor]:
        texts = [self.render(s, a) for s, a in zip(samples, answers)]
        old_side = self.tok.padding_side
        self.tok.padding_side = padding_side
        try:
            if self.modality == "image_text":
                images = [im for s in samples for im in s.load_images(self.image_root)]
                kw: Dict[str, Any] = {"text": texts, "return_tensors": "pt", "padding": True}
                if images:
                    kw["images"] = images
                enc = self.processor(**kw)
            else:
                kw = {"return_tensors": "pt", "padding": True, "return_offsets_mapping": self.template == "plain"}
                if self.max_length:
                    kw.update(truncation=True, max_length=self.max_length)
                enc = self.tok(texts, add_special_tokens=self.template == "plain", **kw)
        finally:
            self.tok.padding_side = old_side
        return dict(enc)

    def _answer_mask(self, samples, answers, enc, padding_side="right") -> torch.Tensor:
        ids, attn = enc["input_ids"], enc["attention_mask"]
        B, L = ids.shape
        mask = torch.zeros((B, L), dtype=torch.bool)
        if self.template == "chat":
            # The answer tokens are the suffix of (prompt + answer) after the longest common token prefix with
            # the prompt-only rendering.  Comparing *tokens* (not lengths) is what makes this robust: a chat
            # template ending in "assistant: " tokenises its trailing space alone, but in the full text that
            # space merges into the first answer token ("\u2581a"); a plain length difference would drop that token.
            prompt_enc = self.encode(samples, [None] * len(samples), padding_side="right")
            p_ids, p_attn = prompt_enc["input_ids"], prompt_enc["attention_mask"]
            n_full = attn.sum(-1)
            for i, a in enumerate(answers):
                if a is None:
                    continue
                off = int(attn[i].nonzero()[0]) if padding_side == "left" else 0
                pr = p_ids[i][: int(p_attn[i].sum())]
                fu = ids[i][off: off + int(n_full[i])]
                n = min(len(pr), len(fu))
                eq = (pr[:n] == fu[:n])
                common = int(n) if bool(eq.all()) else int((~eq).nonzero()[0])
                if common < len(pr) - 2:
                    raise RuntimeError(f"answer_span failed (sample {i}): prompt-only tokens diverge from the full "
                                       f"rendering at position {common}/{len(pr)}; the chat template is not prefix-stable")
                if common >= len(fu):
                    raise RuntimeError(f"answer_span failed (sample {i}): the chat template did not append the answer")
                mask[i, off + common: off + int(n_full[i])] = True
        else:
            offsets = enc["offset_mapping"]
            for i, (s, a) in enumerate(zip(samples, answers)):
                if a is None:
                    continue
                start = len(s.prompt + self.sep)
                mask[i] = (offsets[i, :, 1] > start) & (attn[i] > 0)
        if len(self.image_token_ids):
            mask &= ~torch.isin(ids, self.image_token_ids)
        return mask

    def _labels(self, enc, mask):
        ids = enc["input_ids"]
        labels = ids.clone()
        labels[~mask] = IGNORE
        full = ids.clone()
        real = enc["attention_mask"] > 0
        if len(self.image_token_ids):
            real &= ~torch.isin(ids, self.image_token_ids)
        full[~real] = IGNORE
        return labels, full

    # ---------------------------------------------------------------- public
    def __call__(self, items: List[dict]) -> Dict[str, Any]:
        samples = [it["sample"] for it in items]
        answers = [s.answer for s in samples]
        if any(a is None for a in answers):
            raise ValueError("training batches need SeqSample.answer")
        enc = self.encode(samples, answers)
        mask = self._answer_mask(samples, answers, enc)
        labels, full = self._labels(enc, mask)
        batch: Dict[str, Any] = {k: v for k, v in enc.items() if k != "offset_mapping"}
        batch["labels"] = full if self.loss_on == "full" else labels
        batch["answer_labels"], batch["full_labels"] = labels, full
        batch["idx"] = torch.tensor([it["idx"] for it in items])
        batch["samples"] = samples          # per-sample metadata for methods that need it (never sent to the model)
        batch["alt_k"] = torch.tensor([it["alt_k"] for it in items])
        alts = [s.alt_answers[it["alt_k"]] if s.alt_answers else self.alt_text for s, it in zip(samples, items)]
        if all(a is not None for a in alts):
            aenc = self.encode(samples, alts)
            amask = self._answer_mask(samples, alts, aenc)
            alabels, _ = self._labels(aenc, amask)
            for k in self.rmodel.text_input_keys:
                if k in aenc:
                    batch["alt_" + k] = aenc[k]
            batch["alt_labels"] = alabels
        return batch

    def prompts(self, samples: Sequence[SeqSample]) -> Dict[str, Any]:
        """Prompt-only, left-padded inputs for batched generation."""
        enc = self.encode(samples, [None] * len(samples), padding_side="left")
        enc.pop("offset_mapping", None)
        return enc


def make_loader(samples: Sequence[SeqSample], collator: SeqCollator, batch_size: int = 4, shuffle: bool = True,
                seed: int = 0, num_workers: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(SeqDataset(samples), batch_size=batch_size, shuffle=shuffle, collate_fn=collator,
                      generator=g, num_workers=num_workers)


def build_unlearn_loaders(forget: Sequence[SeqSample], retain: Optional[Sequence[SeqSample]], collator: SeqCollator,
                          batch_size: int = 4, retain_batch_size: Optional[int] = None, seed: int = 0):
    """MergedLoaders({"Forget", "Retain"}) as every seq trainer expects (epoch = one pass over Forget)."""
    from ..utils.data import MergedLoaders
    loaders = {"Forget": make_loader(forget, collator, batch_size, True, seed)}
    if retain:
        loaders["Retain"] = make_loader(retain, collator, retain_batch_size or batch_size, True, seed + 1)
    return MergedLoaders(loaders)
