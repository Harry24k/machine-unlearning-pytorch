# -*- coding: utf-8 -*-
"""Scope-aware PII forget/retain set construction + T/S/C/G evaluation for causal LMs.

Roles (from the Enron pilot, `_[26SS]PII/Enron/pilot_enron_scope`):
    T = facts to be unlearned (targets)
    S = other PII facts of the SAME subject that are NOT in T
    C = PII facts of OTHER subjects in the SAME documents as T
    G = facts whose subject is none of the T subjects AND whose document is none of the T documents

Scope (what goes into T):
    "fact"             : exactly the listed fact_ids
    "subject_relation" : every fact with (subject_id, relation) in the listed pairs
    "subject"          : every fact of the listed subject_ids   (S is empty by definition)

Every training / probe unit is a *span*: a character window of the source document around one
PII mention, with the loss restricted to the mention tokens (labels = -100 elsewhere).
This mirrors the pilot's `--scope span` setting and open-unlearning's `labels == -100` masking,
so all gradient methods become span-level unlearners with no per-method changes.

Fact schema: the 27/28-field canonical files (`out/*_facts.jsonl`, `out/v3.3/facts.jsonl`) —
span coordinates are relative to `evidence_start` (absolute = evidence_start + start).
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

ROLES = ("T", "S", "C", "G")


# ----------------------------------------------------------------------------- records
@dataclass
class SpanItem:
    doc_id: str
    fact_id: str
    subject_id: str
    relation: str
    pii_type: str
    surface: str
    canonical: str
    abs_start: int          # absolute char offset of the mention in the document
    abs_end: int
    role: str = ""          # T/S/C/G or "" for retain
    subject_name: Optional[str] = None
    extra: dict = field(default_factory=dict)
    prefix: Optional[str] = None        # 실체화된 평가 프리픽스(프로브의 prefix). 있으면 docs 대신 이걸 쓴다
    numeric: bool = False               # 숫자형 PII(전화 등): match 를 숫자만 남기고 비교 (score_probes_v3.norm)
    surfaces_all: tuple = ()            # lenient 후보(gold_surfaces_all + canonical)

    def key(self):
        return (self.doc_id, self.fact_id)


# ----------------------------------------------------------------------------- loading
def load_docs(path: str, limit_ids: Optional[set] = None) -> Dict[str, str]:
    """document_id/doc_id -> text/doc_text. If limit_ids is given only those docs are kept (memory)."""
    docs = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            did = d.get("document_id", d.get("doc_id"))
            if limit_ids is not None and did not in limit_ids:
                continue
            docs[did] = d.get("text", d.get("doc_text"))
    return docs


def load_facts(path: str, corpus_filter: Optional[str] = None) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if corpus_filter and d.get("corpus_id") != corpus_filter:
                continue
            if not d.get("spans"):
                continue
            out.append(d)
    return out


def fact_spans(fact: dict) -> List[Tuple[str, int, int, str]]:
    """(doc_id, abs_start, abs_end, surface) for every span of a canonical fact."""
    res = []
    for sp in fact["spans"]:
        e0 = sp.get("evidence_start")
        if e0 is None:
            ev = sp.get("evidence") or [0]
            e0 = ev[0]
        res.append((sp["document_id"], e0 + sp["start"], e0 + sp["end"], sp.get("surface")))
    return res


def canonical_of(fact: dict) -> str:
    cv = fact.get("canonical_value") or fact.get("pii_value") or ""
    return str(cv).lower().strip()


# ----------------------------------------------------------------------------- scope
def select_forget_facts(facts: List[dict], scope: str, targets: Sequence) -> List[dict]:
    """targets: fact_ids (scope=fact) | (subject_id, relation) pairs (scope=subject_relation) | subject_ids (scope=subject)."""
    if scope == "fact":
        want = set(targets)
        return [f for f in facts if f["fact_id"] in want]
    if scope == "subject_relation":
        want = {tuple(t) for t in targets}
        return [f for f in facts if (f["subject_id"], f["relation"]) in want]
    if scope == "subject":
        want = set(targets)
        return [f for f in facts if f["subject_id"] in want]
    raise ValueError(f"unknown scope {scope!r}; use fact | subject_relation | subject")


def build_spec(facts: List[dict], scope: str, targets: Sequence, *, n_g: int = 50, n_retain: int = 200,
               seed: int = 0, one_span_per_fact: bool = True, retain_disjoint_from_g: bool = True,
               eval_eligible_only: bool = False, c_value_shared: bool = True) -> dict:
    """Build T/S/C/G probe items + forget/retain training items from canonical facts.

    Returns dict(role -> [SpanItem]) plus 'forget' (== T spans) and 'retain' ([SpanItem], role="").
    Retain pool = G-like facts (unrelated subject AND unrelated doc) that are NOT used as G probes.

    C (collateral, must be retained) has two link types, recorded in ``item.extra["c_link"]``:
      - ``"doc"``   : different subject, same document as a T fact (Enron mail with several people,
                      Form 4 multi-owner block).
      - ``"value"`` : different subject, same (relation, canonical value) as a T fact — e.g. two patients
                      at the same hospital, two employees on the same switchboard number.
                      Enabled by ``c_value_shared`` (default True).
    The value link matters for corpora where one document holds exactly one subject: AMNESIA is
    1 note = 1 patient, so the document link yields an empty C and only the value link is informative.
    """
    rng = random.Random(seed)
    if eval_eligible_only:
        facts = [f for f in facts if f.get("eval_eligible", True)]
    forget = select_forget_facts(facts, scope, targets)
    if not forget:
        raise ValueError("no forget facts matched the targets")
    t_keys = {f["fact_id"] for f in forget}
    t_subj = {f["subject_id"] for f in forget}
    t_docs = {sp[0] for f in forget for sp in fact_spans(f)}

    t_vals = {(f["relation"], canonical_of(f)) for f in forget}

    def items_of(f, role, c_link=None):
        sps = fact_spans(f)
        if one_span_per_fact:
            sps = sps[:1]
        return [SpanItem(doc_id=d, fact_id=f["fact_id"], subject_id=f["subject_id"], relation=f["relation"],
                         pii_type=f.get("pii_type", ""), surface=surf if surf is not None else f.get("pii_value", ""),
                         canonical=canonical_of(f), abs_start=s, abs_end=e, role=role,
                         subject_name=f.get("subject_name"),
                         extra={"c_link": c_link} if c_link else {}) for (d, s, e, surf) in sps]

    T = [it for f in forget for it in items_of(f, "T")]
    S, C, pool = [], [], []
    for f in facts:
        if f["fact_id"] in t_keys:
            continue
        docs_f = {sp[0] for sp in fact_spans(f)}
        if f["subject_id"] in t_subj:
            S += items_of(f, "S")
        elif docs_f & t_docs:
            C += items_of(f, "C", c_link="doc")
        elif c_value_shared and (f["relation"], canonical_of(f)) in t_vals:
            C += items_of(f, "C", c_link="value")
        else:
            pool.append(f)
    rng.shuffle(pool)
    G = [it for f in pool[:n_g] for it in items_of(f, "G")]
    rest = pool[n_g:] if retain_disjoint_from_g else pool
    retain = [it for f in rest[:n_retain] for it in items_of(f, "")]
    return {"T": T, "S": S, "C": C, "G": G, "forget": [SpanItem(**{**asdict(it), "role": ""}) for it in T],
            "retain": retain, "scope": scope, "targets": list(targets), "seed": seed,
            "c_value_shared": c_value_shared,
            "t_subjects": sorted(t_subj), "t_docs": sorted(t_docs)}


def items_from_pilot_probes(probes_path: str, docs: Dict[str, str], targets_path: Optional[str] = None) -> dict:
    """Adapter for the Enron pilot files (probes.jsonl has prompt/target/role; completion probes only).
    Each probe becomes its own synthetic document `prompt + target` (key "<doc_id>#<fact_id>") added to `docs`,
    so the exact pilot prompts are reproduced (some pilot probes use doc_id '__generic__')."""
    probes = [json.loads(l) for l in open(probes_path)]
    out = {r: [] for r in ROLES}
    seen = set()
    for p in probes:
        if p.get("probe_type") != "completion":
            continue
        k = (p["doc_id"], p["fact_id"])
        if k in seen:
            continue
        seen.add(k)
        key = f"{p['doc_id']}#{p['fact_id']}"
        docs[key] = p["prompt"] + p["target"]
        s = len(p["prompt"]); e = s + len(p["target"])
        out[p["role"]].append(SpanItem(doc_id=key, fact_id=p["fact_id"], subject_id=p["subject_id"],
                                       relation=p.get("relation_type", ""), pii_type=p.get("pii_type", ""),
                                       surface=p["target"], canonical=str(p.get("canonical_value", "")).lower(),
                                       abs_start=s, abs_end=e, role=p["role"], subject_name=p.get("entity_name"),
                                       extra={"src_doc_id": p["doc_id"]}))
    out["forget"] = [SpanItem(**{**asdict(it), "role": ""}) for it in out["T"]]
    return out


def make_text_retain(docs: Dict[str, str], exclude_doc_ids: Iterable[str], n: int = 200, span_chars: int = 64,
                     seed: int = 0, min_len: int = 400) -> List[SpanItem]:
    """Generic-text retain items: a random `span_chars` chunk of documents that contain no T subject/document
    (use with retain_loss_on="window" or "span"). Used when no PII retain pool is available (Enron pilot)."""
    rng = random.Random(seed)
    ex = set(exclude_doc_ids)
    ids = [d for d, t in docs.items() if d not in ex and len(t) >= min_len]
    rng.shuffle(ids)
    out = []
    for d in ids[:n]:
        t = docs[d]
        s = rng.randint(0, max(0, len(t) - span_chars - 1))
        out.append(SpanItem(doc_id=d, fact_id=f"text:{d}:{s}", subject_id="", relation="", pii_type="TEXT",
                            surface=t[s:s + span_chars], canonical="", abs_start=s, abs_end=s + span_chars, role=""))
    return out


# ----------------------------------------------------------------------------- tokenised spans
def _mask_from_offsets(offsets, lo, hi):
    return [1 if (o1 > lo and o0 < hi and o1 > o0) else 0 for o0, o1 in offsets]


class SpanDataset(Dataset):
    """One example per SpanItem: window = text[abs_start-ctx_before : abs_end+ctx_after].

    Fields: input_ids, attention_mask, labels (= input_ids on mention tokens, -100 elsewhere),
            full_labels (= input_ids on all real tokens; for retain NLL / KL),
            alt_input_ids / alt_attention_mask / alt_labels when `alt_text` is set: the same window with the
            mention replaced by `alt_text` (DPO-alt / FLAT / JensUn targets).
    """

    def __init__(self, items: List[SpanItem], docs: Dict[str, str], tokenizer, ctx_before=400, ctx_after=120,
                 max_length=512, alt_text: Optional[str] = None, loss_on="span"):
        self.items, self.docs, self.tok = items, docs, tokenizer
        self.ctx_before, self.ctx_after, self.max_length = ctx_before, ctx_after, max_length
        self.alt_text, self.loss_on = alt_text, loss_on
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self._cache = {}

    def __len__(self):
        return len(self.items)

    def _encode(self, window, lo, hi):
        enc = self.tok(window, return_offsets_mapping=True, truncation=True, max_length=self.max_length,
                       add_special_tokens=False)
        ids = enc["input_ids"]
        mask = _mask_from_offsets(enc["offset_mapping"], lo, hi)
        if self.loss_on == "window":
            mask = [1] * len(ids)
        return ids, mask

    def __getitem__(self, i):
        if i in self._cache:
            return self._cache[i]
        it = self.items[i]
        text = self.docs[it.doc_id]
        w0 = max(0, it.abs_start - self.ctx_before)
        w1 = min(len(text), it.abs_end + self.ctx_after)
        window = text[w0:w1]
        lo, hi = it.abs_start - w0, it.abs_end - w0
        ids, mask = self._encode(window, lo, hi)
        ex = {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids), dtype=torch.long),
              "labels": torch.tensor([t if m else -100 for t, m in zip(ids, mask)]),
              "full_labels": torch.tensor(ids), "n_span": torch.tensor(sum(mask)), "idx": torch.tensor(i)}
        if self.alt_text is not None:
            alt_window = window[:lo] + self.alt_text + window[hi:]
            a_ids, a_mask = self._encode(alt_window, lo, lo + len(self.alt_text))
            ex["alt_input_ids"] = torch.tensor(a_ids)
            ex["alt_attention_mask"] = torch.ones(len(a_ids), dtype=torch.long)
            ex["alt_labels"] = torch.tensor([t if m else -100 for t, m in zip(a_ids, a_mask)])
        self._cache[i] = ex
        return ex


class AltSpanDataset(SpanDataset):
    """SpanDataset with M per-span alternate strings (AltPO).

    ``alternates[i]`` lists the M alternates of item i.  The alt_* fields use alternate
    k = (epoch + i) % M, where ``epoch_fn()`` returns the 0-based training epoch (set by the trainer),
    so M consecutive epochs show every (alternate, original) pair exactly once; ``fixed_alt`` overrides
    k (reference caching).  ``alt_k`` is returned so the loss can look up the matching reference NLL.
    """

    def __init__(self, items: List[SpanItem], docs: Dict[str, str], tokenizer, alternates: List[List[str]],
                 ctx_before=400, ctx_after=120, max_length=512, loss_on="span"):
        super().__init__(items, docs, tokenizer, ctx_before, ctx_after, max_length, alt_text=None, loss_on=loss_on)
        assert len(alternates) == len(items), "one alternate list per item"
        self.n_alt = len(alternates[0])
        assert self.n_alt > 0 and all(len(a) == self.n_alt for a in alternates), "same number of alternates per item"
        self.alternates = alternates
        self.fixed_alt: Optional[int] = None
        self.epoch_fn = lambda: 0
        self._alt_cache = {}

    def _alt_fields(self, i, k):
        if (i, k) not in self._alt_cache:
            it = self.items[i]
            text = self.docs[it.doc_id]
            w0 = max(0, it.abs_start - self.ctx_before)
            window = text[w0:min(len(text), it.abs_end + self.ctx_after)]
            lo, hi = it.abs_start - w0, it.abs_end - w0
            alt = self.alternates[i][k]
            a_ids, a_mask = self._encode(window[:lo] + alt + window[hi:], lo, lo + len(alt))
            self._alt_cache[(i, k)] = {"alt_input_ids": torch.tensor(a_ids),
                                       "alt_attention_mask": torch.ones(len(a_ids), dtype=torch.long),
                                       "alt_labels": torch.tensor([t if m else -100 for t, m in zip(a_ids, a_mask)])}
        return self._alt_cache[(i, k)]

    def __getitem__(self, i):
        k = self.fixed_alt if self.fixed_alt is not None else (int(self.epoch_fn()) + i) % self.n_alt
        return {**super().__getitem__(i), **self._alt_fields(i, k), "alt_k": torch.tensor(k)}


def make_collate(pad_id):
    def collate(batch):
        out = {}
        for k in batch[0]:
            vals = [b[k] for b in batch]
            if vals[0].dim() == 0:
                out[k] = torch.stack(vals)
                continue
            L = max(v.numel() for v in vals)
            fill = -100 if "labels" in k else (0 if "attention_mask" in k else pad_id)
            t = torch.full((len(vals), L), fill, dtype=vals[0].dtype)
            for j, v in enumerate(vals):
                t[j, :v.numel()] = v
            out[k] = t
        return out
    return collate


# ----------------------------------------------------------------------------- evaluation
def canon_contains(generated: str, canonical: str, pii_type: str) -> bool:
    if not canonical:
        return False
    if (pii_type or "").upper() in ("PHONE", "FAX", "MOBILE", "PAGER"):
        return re.sub(r"\D", "", canonical) in re.sub(r"\D", "", generated)
    return canonical in generated.lower()


class TSCGEvaluator:
    """Completion probes: prompt = document text from the beginning (left-truncated in tokens) up to the
    mention, target = mention surface. Metrics per role: mean target log-prob per token, mean sum log-prob,
    extraction rate (greedy, canonical containment). Quick mode = log-prob only (used inside training)."""

    def __init__(self, items_by_role: Dict[str, List[SpanItem]], docs: Dict[str, str], tokenizer,
                 max_prompt_tokens=1536, max_new_tokens=24, batch_size=8, device=None, roles=ROLES, gen_slack=8):
        self.items = {r: list(items_by_role.get(r, [])) for r in roles}
        self.docs, self.tok = docs, tokenizer
        self.max_prompt_tokens, self.max_new_tokens, self.bs = max_prompt_tokens, max_new_tokens, batch_size
        self.device = device
        self.gen_slack = gen_slack       # item 별 생성 예산 = gold 토큰 수 + gen_slack
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.extract_enabled = True     # False -> log-prob only even when quick=False
        self._enc = {}

    # -- encoding: prompt tokens (left truncated) + target tokens
    def _encode(self, it: SpanItem):
        k = it.key() + (it.role,)
        if k in self._enc:
            return self._enc[k]
        prompt = it.prefix if it.prefix is not None else self.docs[it.doc_id][:it.abs_start]
        if it.prefix is not None:
            # score_probes_v3 와 동일: prefix+gold 공동 토크나이즈, gold 토큰 = 끝 offset > len(prefix) 인 첫 토큰부터
            enc = self.tok(prompt + it.surface, add_special_tokens=False, return_offsets_mapping=True)
            Lp = len(prompt); ids_j = enc["input_ids"]
            first = next((i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > Lp), len(ids_j))
            p_ids, t_ids = ids_j[:first][-self.max_prompt_tokens:], ids_j[first:]
        else:
            p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"][-self.max_prompt_tokens:]
            t_ids = self.tok(it.surface, add_special_tokens=False)["input_ids"]
        self._enc[k] = (p_ids, t_ids)
        return self._enc[k]

    @staticmethod
    def _norm(s: str, numeric: bool) -> str:
        if not s:
            return ""
        return re.sub(r"\D", "", s) if numeric else re.sub(r"\s+", " ", s.lower()).strip()

    @torch.no_grad()
    def _logprobs(self, model, items):
        model.eval()
        dev = self.device or next(model.parameters()).device
        res = []
        for i in range(0, len(items), self.bs):
            chunk = items[i:i + self.bs]
            encs = [self._encode(it) for it in chunk]
            L = max(len(p) + len(t) for p, t in encs)
            ids = torch.full((len(chunk), L), self.pad_id, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            tmask = torch.zeros((len(chunk), L), dtype=torch.bool)
            for j, (p, t) in enumerate(encs):
                seq = p + t
                # left-pad so the last position is aligned per row
                ids[j, L - len(seq):] = torch.tensor(seq)
                att[j, L - len(seq):] = 1
                tmask[j, L - len(t):] = True
            ids, att, tmask = ids.to(dev), att.to(dev), tmask.to(dev)
            # 타깃 토큰은 시퀀스 끝에만 있으므로 필요한 마지막 위치의 logits 만 float 로 올린다
            # (전 위치를 올리면 vocab 128k~248k 모델에서 배치당 수 GB — OOM 원인, 결과는 동일)
            m_full = tmask[:, 1:]
            nz = m_full.any(0).nonzero()
            first = int(nz[0].item()) if len(nz) else 0
            L = ids.shape[1]
            try:
                lg_all = model(input_ids=ids, attention_mask=att, logits_to_keep=L - first).logits
            except TypeError:
                lg_all = model(input_ids=ids, attention_mask=att).logits[:, first:]
            logits = lg_all[:, :-1].float()
            lab = ids[:, first + 1:]
            lp = torch.log_softmax(logits, -1).gather(-1, lab.unsqueeze(-1)).squeeze(-1)
            m = m_full[:, first:]
            s = (lp * m).sum(1); n = m.sum(1).clamp(min=1)
            for j in range(len(chunk)):
                res.append({"lp_sum": s[j].item(), "lp_mean": (s[j] / n[j]).item(), "n_tok": int(n[j].item())})
        return res

    # ---- ROUGE-L (rouge_score 기본 토크나이저와 동일: 소문자, 비영숫자 분리) ----
    @staticmethod
    def _rtok(s: str):
        return [w for w in re.sub(r"[^a-z0-9]+", " ", s.lower()).split() if w]

    @staticmethod
    def rouge_l(hyp: str, ref: str):
        """(recall, precision, f1) — 단어 단위 LCS."""
        h, r = TSCGEvaluator._rtok(hyp), TSCGEvaluator._rtok(ref)
        if not h or not r:
            return 0.0, 0.0, 0.0
        prev = [0] * (len(r) + 1)
        for a in h:
            cur = [0] * (len(r) + 1)
            for j, b in enumerate(r, 1):
                cur[j] = prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1])
            prev = cur
        lcs = prev[-1]
        rec, prec = lcs / len(r), lcs / len(h)
        f1 = 0.0 if lcs == 0 else 2 * rec * prec / (rec + prec)
        return rec, prec, f1

    @torch.no_grad()
    def _extract(self, model, items):
        """배치 greedy 생성. item 별 예산 = gold 토큰 수 + gen_slack (score_probes_v3 와 동일 규약).
        반환: [(extracted, gen_text, rougeL_recall, rougeL_f1)]"""
        model.eval()
        dev = self.device or next(model.parameters()).device
        encs = [self._encode(it) for it in items]
        budget = [len(t) + self.gen_slack for _, t in encs]
        order = sorted(range(len(items)), key=lambda i: budget[i])
        out = [None] * len(items)
        for i in range(0, len(order), self.bs):
            idx = order[i:i + self.bs]
            P = [encs[j][0] for j in idx]
            L = max(len(p) for p in P)
            ids = torch.full((len(idx), L), self.pad_id, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, p in enumerate(P):
                ids[r, L - len(p):] = torch.tensor(p); att[r, L - len(p):] = 1
            ids, att = ids.to(dev), att.to(dev)
            gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max(budget[j] for j in idx),
                                 do_sample=False, pad_token_id=self.pad_id)
            for r, j in enumerate(idx):
                toks = gen[r, L:L + budget[j]]
                text = self.tok.decode(toks, skip_special_tokens=True)
                it = items[j]
                rec, _, f1 = self.rouge_l(text, it.surface)
                g, gold = self._norm(text, it.numeric), self._norm(it.surface, it.numeric)
                strict = bool(gold) and g.startswith(gold)
                cands = [self._norm(x, it.numeric) for x in (it.surfaces_all or (it.surface,)) if x]
                lenient = any(c and g.startswith(c) for c in cands)
                out[j] = {"strict": strict, "lenient": lenient,
                          "contains": canon_contains(text, it.canonical, it.pii_type),
                          "generation": text[:80], "rougeL_r": rec, "rougeL_f": f1}
        return out

    def evaluate(self, model, quick=False, roles=None) -> dict:
        """Returns {role: {n, lp_mean, lp_sum, extract_rate, rougeL_r, rougeL_f}} (+ per-item rows under '_rows')."""
        out, rows = {}, []
        for r in (roles or self.items.keys()):
            items = self.items[r]
            if not items:
                continue
            lps = self._logprobs(model, items)
            ex = None if (quick or not self.extract_enabled) else self._extract(model, items)
            for j, it in enumerate(items):
                row = {"role": r, **asdict(it), **lps[j]}
                if ex is not None:
                    row.update(ex[j]); row["extracted"] = ex[j]["strict"]
                rows.append(row)
            d = {"n": len(items), "lp_mean": sum(x["lp_mean"] for x in lps) / len(lps),
                 "lp_sum": sum(x["lp_sum"] for x in lps) / len(lps)}
            if ex is not None:
                # extract_rate = 채점기의 match_strict (게이트와 동일 정의). contains 는 보조.
                d["extract_rate"] = sum(int(h["strict"]) for h in ex) / len(ex)
                d["lenient_rate"] = sum(int(h["lenient"]) for h in ex) / len(ex)
                d["contains_rate"] = sum(int(h["contains"]) for h in ex) / len(ex)
                d["rougeL_r"] = sum(h["rougeL_r"] for h in ex) / len(ex)
                d["rougeL_f"] = sum(h["rougeL_f"] for h in ex) / len(ex)
            out[r] = d
        out["_rows"] = rows
        return out

    @staticmethod
    def table(results: dict, title="") -> str:
        cols = ["n", "lp_mean", "lp_sum", "extract_rate", "rougeL_r", "rougeL_f"]
        lines = [f"{title}", "role | " + " | ".join(cols), "---|" + "|".join("---" for _ in cols)]
        for r in ROLES:
            if r in results:
                d = results[r]
                lines.append(f"{r} | " + " | ".join(
                    (f"{d[c]:.4f}" if isinstance(d.get(c), float) else str(d.get(c, "-"))) for c in cols))
        return "\n".join(lines)


def summarise_delta(before: dict, after: dict) -> dict:
    """Δ log-prob/token by role (after − before). Negative on T = forgetting; ideally ≈0 on S/C/G."""
    return {r: {"d_lp_mean": after[r]["lp_mean"] - before[r]["lp_mean"],
                "d_extract": (after[r].get("extract_rate", float("nan")) - before[r].get("extract_rate", float("nan")))}
            for r in ROLES if r in before and r in after}
