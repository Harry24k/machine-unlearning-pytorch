#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run one span-level unlearning method with scope-aware forget set and T/S/C/G evaluation.

Data sources
  --pilot          Enron pilot (M_T Qwen2-1.5B checkpoint, probes.jsonl, pilot_docs.jsonl); retain = random
                   text windows from pilot docs that are not T/G documents.
  --facts/--docs   canonical fact/doc files (27/28-field schema) + --scope {fact,subject_relation,subject}
                   + --targets (fact_ids | "subject_id:relation" pairs | subject_ids); S/C/G built by rule.

Outputs  <out>/<method>[_tag]/results.json  {method, hparams, setup, before, after, delta, history, time}
         (+ model weights if --save-model)

Examples
  python scripts/run_pii_unlearn.py --pilot --method LLM-NPO --epochs 5 --lr 1e-5 --hp beta=0.1
  python scripts/run_pii_unlearn.py --model out/ft_runs/llama31_8b/epoch2 --facts out/v3.3/facts.jsonl \
      --docs out/v3.3/docs.jsonl --scope subject --targets "N:allen-p/_sent_mail/586.#jim murnan" --method LLM-RMU --hp layer_id=7
"""
import argparse, json, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from torchunlearn.unlearn.llm_data import (load_docs, load_facts, build_spec, items_from_pilot_probes, make_text_retain,
                                           SpanDataset, make_collate, TSCGEvaluator, summarise_delta, ROLES)
from torchunlearn.unlearn.trainers.llm_pii import LLMRobModel, LLM_TRAINERS
from torchunlearn.unlearn.nontrainers.revs import REVS
from torchunlearn.utils.data import MergedLoaders

PII = "/home1/irteam/_[chaewon]/_[26SS]PII"
PILOT = dict(model=f"{PII}/Enron/pilot_enron_scope/out/M_T",
             probes=f"{PII}/Enron/pilot_enron_scope/out/probes.jsonl",
             docs=f"{PII}/Enron/pilot_enron_annotation/data/pilot_docs.jsonl")


def parse_hp(kvs):
    out = {}
    for kv in kvs or []:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except Exception:
            pass
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="registry name, e.g. LLM-NPO (or short: NPO)")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--model", default=None, help="HF model dir (default: pilot M_T)")
    ap.add_argument("--facts"); ap.add_argument("--docs"); ap.add_argument("--corpus", default=None)
    ap.add_argument("--scope", default="fact", choices=["fact", "subject_relation", "subject"])
    ap.add_argument("--targets", nargs="*", default=[], help="fact_ids | subject_id:RELATION (last colon splits) | subject_ids | @file.json")
    ap.add_argument("--n-g", type=int, default=50); ap.add_argument("--n-retain", type=int, default=200)
    ap.add_argument("--retain-kind", default="pii", choices=["pii", "text"])
    ap.add_argument("--max-s", type=int, default=100); ap.add_argument("--max-c", type=int, default=100)
    ap.add_argument("--hp", nargs="*", default=[], help="method hparams key=value (JSON values)")
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optimizer", default=None, help='e.g. "AdamW(lr=1e-5)" / "SGD(lr=1e-3)"')
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--retain-bs", type=int, default=4)
    ap.add_argument("--ctx-before", type=int, default=400); ap.add_argument("--ctx-after", type=int, default=120)
    ap.add_argument("--max-length", type=int, default=512); ap.add_argument("--alt", default="[REDACTED]")
    ap.add_argument("--stop-lp", type=float, default=None); ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--max-prompt-tokens", type=int, default=1536); ap.add_argument("--eval-bs", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/pii_unlearn"); ap.add_argument("--tag", default="")
    ap.add_argument("--save-model", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-extract", action="store_true", help="skip greedy extraction (log-prob only)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    name = args.method if args.method.startswith("LLM-") else "LLM-" + args.method
    short = name[4:]
    if short != "REVS" and short not in LLM_TRAINERS:
        sys.exit(f"unknown method {name}; choose from {list(LLM_TRAINERS)} + REVS")

    # ---------------------------------------------------------------- data
    if args.pilot:
        model_path = args.model or PILOT["model"]
        docs = load_docs(PILOT["docs"])
        spec = items_from_pilot_probes(PILOT["probes"], docs)
        excl = {it.extra.get("src_doc_id", it.doc_id) for r in ROLES for it in spec[r]} | {k for k in docs if "#" in k}
        spec["retain"] = make_text_retain(docs, excl, n=args.n_retain, seed=args.seed)
        retain_loss_on_default = "window"
    else:
        if not (args.model and args.facts and args.docs and args.targets):
            sys.exit("--model --facts --docs --targets required without --pilot")
        model_path = args.model
        targets = args.targets
        if len(targets) == 1 and targets[0].startswith("@"):
            targets = json.load(open(targets[0][1:]))
        if args.scope == "subject_relation":
            targets = [tuple(t.rsplit(":", 1)) if isinstance(t, str) else tuple(t) for t in targets]  # subject ids contain ":"
        facts = load_facts(args.facts, corpus_filter=args.corpus)
        spec = build_spec(facts, args.scope, targets, n_g=args.n_g, n_retain=args.n_retain, seed=args.seed)
        rng = random.Random(args.seed)
        for r, cap in (("S", args.max_s), ("C", args.max_c)):
            if len(spec[r]) > cap:
                rng.shuffle(spec[r]); spec[r] = spec[r][:cap]
        need = {it.doc_id for r in ROLES for it in spec[r]} | {it.doc_id for it in spec["retain"]}
        docs = load_docs(args.docs, limit_ids=need)
        if args.retain_kind == "text":
            excl = {it.doc_id for r in ROLES for it in spec[r]}
            spec["retain"] = make_text_retain(docs, excl, n=args.n_retain, seed=args.seed)
        retain_loss_on_default = "window" if args.retain_kind == "text" else "span"
    counts = {r: len(spec[r]) for r in ROLES}; counts["retain"] = len(spec["retain"])
    print("[data]", counts, "scope:", spec.get("scope", "pilot"))
    if args.dry_run:
        return

    # ---------------------------------------------------------------- model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=getattr(torch, args.dtype)).to(args.device)
    rmodel = LLMRobModel(model, tok, device=args.device)
    evaluator = TSCGEvaluator({r: spec[r] for r in ROLES}, docs, tok, max_prompt_tokens=args.max_prompt_tokens,
                              batch_size=args.eval_bs, device=args.device)
    if args.no_extract:
        evaluator.extract_enabled = False

    hp = parse_hp(args.hp)
    out_dir = os.path.join(args.out, short + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ---------------------------------------------------------------- run
    if short == "REVS":
        un = REVS(rmodel, **hp).set_evaluator(evaluator)
        un.fit(spec["forget"], docs)
        setup = {}
        history = []
    else:
        cls = LLM_TRAINERS[short]
        hp.setdefault("retain_loss_on", retain_loss_on_default)
        if args.stop_lp is not None:
            hp["stop_lp"] = args.stop_lp; hp["eval_every"] = args.eval_every
        un = cls(rmodel, **hp).set_evaluator(evaluator)
        alt = args.alt if cls.needs_alt else None
        f_ds = SpanDataset(spec["forget"], docs, tok, args.ctx_before, args.ctx_after, args.max_length, alt_text=alt)
        r_ds = SpanDataset(spec["retain"], docs, tok, args.ctx_before, args.ctx_after, args.max_length,
                           loss_on="span")
        col = make_collate(tok.pad_token_id)
        loaders = {"Forget": DataLoader(f_ds, batch_size=args.bs, shuffle=True, collate_fn=col)}
        if un.alpha != 0 and len(r_ds):
            loaders["Retain"] = DataLoader(r_ds, batch_size=args.retain_bs, shuffle=True, collate_fn=col)
        opt = args.optimizer or f"AdamW(lr={args.lr})"
        un.setup(optimizer=opt, n_epochs=args.epochs)
        setup = {"optimizer": opt, "n_epochs": args.epochs, "bs": args.bs, "retain_bs": args.retain_bs}
        un.fit(MergedLoaders(loaders), n_epochs=args.epochs, record_type="Epoch")
        history = un.history

    res = un.results
    delta = summarise_delta(res["before"], res["after"])
    summary = {"method": name, "model": model_path, "hparams": hp, "setup": setup, "data": counts,
               "scope": spec.get("scope", "pilot"), "targets": spec.get("targets", None),
               "before": {r: res["before"][r] for r in ROLES if r in res["before"]},
               "after": {r: res["after"][r] for r in ROLES if r in res["after"]},
               "delta": delta, "history": history, "stopped_early": res.get("stopped_early"),
               "n_edits": res.get("n_edits"), "iters": res.get("iters"), "time_s": round(time.time() - t0, 1)}
    json.dump(summary, open(os.path.join(out_dir, "results.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "rows_after.jsonl"), "w") as f:
        for row in res["after"]["_rows"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[done] {name}  Δlp/token: " + "  ".join(f"{r}={delta[r]['d_lp_mean']:+.3f}" for r in delta)
          + f"  ({summary['time_s']}s) -> {out_dir}/results.json")
    if args.save_model:
        rmodel.model.save_pretrained(os.path.join(out_dir, "model")); tok.save_pretrained(os.path.join(out_dir, "model"))


if __name__ == "__main__":
    main()
