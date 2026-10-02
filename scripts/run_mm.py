#!/usr/bin/env python
"""Sequence / multimodal unlearning CLI (text LLMs and image+text LVLMs share every code path).

  finetune   fine-tune a base checkpoint on a dataset -> the model to unlearn from (or retrain on retain)
  unlearn    run one registered MM-* method on (forget, retain) and evaluate before/after
  eval       evaluate a checkpoint on any number of role=spec sets

Data specs (see torchunlearn.unlearn.seq_data.load_spec):
  jsonl:<path>[.json|.jsonl]      records; map fields with --fields prompt=question,answer=answer,images=image,group=subject
  llava:<path.json>               LLaVA "conversations" JSON (VLGuard / MLLMU exports); --image-root for relative paths
  hf:<name>[:<split>[:<config>]]  HF datasets (image columns are PIL already)

Examples
  python scripts/run_mm.py finetune --model llava-hf/llava-1.5-7b-hf --data llava:train.json --image-root imgs \\
      --trainable lm --epochs 3 --lr 2e-5 --batch-size 4 --out runs/ft_llava
  python scripts/run_mm.py unlearn --model runs/ft_llava/model --data llava:train.json --image-root imgs \\
      --split group:0.1 --method MM-NPO --hp beta=0.1 alpha=1.0 --trainable lm --epochs 5 --lr 1e-5 --out runs/npo
  python scripts/run_mm.py unlearn --model <ckpt> --forget jsonl:forget.jsonl --retain jsonl:retain.jsonl \\
      --test jsonl:test.jsonl --method MM-RMU --hp layer_id=7 --out runs/rmu --save-model
  python scripts/run_mm.py eval --model <ckpt> --data Forget=jsonl:forget.jsonl Retain=jsonl:retain.jsonl --out runs/eval
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from torchunlearn import SeqRobModel, SeqUnlearningEvaluator  # noqa: E402
from torchunlearn.api.registry import build_unlearner_split, get_kind, list_algorithms, short_spec  # noqa: E402
from torchunlearn.metrics.seq import summarise_delta  # noqa: E402
from torchunlearn.unlearn.seq_data import (SeqCollator, build_unlearn_loaders, load_spec, make_loader,  # noqa: E402
                                           split_forget_retain, strip_images)
from torchunlearn.unlearn.trainers.seq import SeqFinetune  # noqa: E402


def parse_kv(kvs):
    out = {}
    for kv in kvs or []:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except Exception:  # noqa: BLE001
            pass
        out[k] = v
    return out


def add_model_args(ap):
    ap.add_argument("--model", required=True, help="HF id / local dir / PEFT adapter dir (base, fine-tuned ...)")
    ap.add_argument("--modality", default="auto", choices=["auto", "text", "image_text"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    ap.add_argument("--device", default=None, help="e.g. cuda:0 (default: cuda if available)")
    ap.add_argument("--device-map", default=None, help="'auto' for sharded loading")
    ap.add_argument("--processor", default=None, help="processor/tokenizer repo when the checkpoint lacks one")
    ap.add_argument("--attn", default=None, help="attn_implementation (sdpa | eager | flash_attention_2)")
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="torch.cuda.set_per_process_memory_fraction (shared GPUs: always set this)")
    ap.add_argument("--local-files-only", action="store_true")


def add_data_args(ap):
    ap.add_argument("--fields", nargs="*", default=[], help="SeqSample field=record key (jsonl/hf specs)")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--n-limit", type=int, default=None, help="cap per set (smoke runs)")
    ap.add_argument("--template", default="auto", choices=["auto", "chat", "plain"])
    ap.add_argument("--system", default=None, help="system prompt for chat templates")
    ap.add_argument("--loss-on", default="answer", choices=["answer", "full"])
    ap.add_argument("--alt-text", default="[REDACTED]", help="fixed alternate answer (DPO / FLAT) when samples have none")
    ap.add_argument("--max-length", type=int, default=None, help="text-only truncation")


def add_train_args(ap):
    ap.add_argument("--trainable", default=None,
                    help="all | lm | vision | projector | lm+projector | lora:r=16,alpha=32,target=lm | re:<regex>")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--optimizer", default=None, help='full spec, e.g. "AdamW(lr=1e-5, weight_decay=0.01)"')
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--retain-batch-size", type=int, default=None)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)


def add_eval_args(ap):
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--no-gen", action="store_true", help="log-prob metrics only")
    ap.add_argument("--no-noimg-probe", action="store_true", help="skip the text-only (image stripped) forget probe")
    ap.add_argument("--n-eval", type=int, default=None, help="cap samples per role at evaluation")


def load_model(a, trainable="none"):
    if a.mem_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    model = SeqRobModel.from_pretrained(a.model, modality=a.modality, dtype=a.dtype, device=a.device, device_map=a.device_map,
                                    trainable=trainable, processor_id=a.processor, attn_impl=a.attn,
                                    local_files_only=a.local_files_only)
    print(model)
    return model


def load_set(spec, a, **kw):
    if spec is None:
        return None
    ss = load_spec(spec, fields=parse_kv(a.fields) or None, image_root=a.image_root, **kw)
    if a.n_limit:
        ss = ss[: a.n_limit]
    return ss


def make_collator(model, a):
    return SeqCollator(model, loss_on=a.loss_on, template=a.template, alt_text=a.alt_text, system=a.system,
                       max_length=a.max_length, image_root=a.image_root)


def make_evaluator(roles, col, a):
    roles = {k: v for k, v in roles.items() if v}
    if "Forget" in roles and not a.no_noimg_probe and any(s.is_multimodal for s in roles["Forget"]):
        roles["Forget_noimg"] = strip_images(roles["Forget"])
    return SeqUnlearningEvaluator(roles, col, batch_size=a.eval_batch_size, gen=not a.no_gen,
                                  max_new_tokens=a.max_new_tokens, n_limit=a.n_eval)


def dump(out, obj, name="results.json"):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name), "w") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)
    print(f"[run_mm] wrote {os.path.join(out, name)}")


def split_sets(a, data):
    """--split random:<ratio> | group:<ratio> | groups:<g1,g2,...> | ids:<file>"""
    kind, _, arg = a.split.partition(":")
    if kind == "random":
        return split_forget_retain(data, by="random", ratio=float(arg), seed=a.seed)
    if kind == "group":
        return split_forget_retain(data, by="group", ratio=float(arg), seed=a.seed)
    if kind == "groups":
        return split_forget_retain(data, by="group", groups=arg.split(","))
    if kind == "ids":
        ids = [l.strip() for l in open(arg) if l.strip()]
        return split_forget_retain(data, by="ids", ids=ids)
    raise ValueError(a.split)


# --------------------------------------------------------------------------------------- commands
def cmd_finetune(a):
    model = load_model(a, trainable=a.trainable or "all")
    data = load_set(a.data, a)
    col = make_collator(model, a)
    loader = make_loader(data, col, batch_size=a.batch_size, shuffle=True, seed=a.seed)
    roles = {"Train": data[: (a.n_eval or 200)]}
    if a.eval_data:
        roles["Eval"] = load_set(a.eval_data, a)
    ev = make_evaluator(roles, col, a)
    tr = SeqFinetune(model, grad_ckpt=a.grad_ckpt, traj_every=a.traj_every).set_evaluator(ev)
    opt = a.optimizer or f"AdamW(lr={a.lr or 2e-5})"
    tr.setup(optimizer=opt, n_epochs=a.epochs or 3, clip_grad_norm=1.0)
    t0 = time.time()
    tr.fit(loader, n_epochs=a.epochs or 3, eval_before=True, eval_after=True)
    model.save_pretrained(os.path.join(a.out, "model"))
    dump(a.out, {"command": "finetune", "model": a.model, "data": a.data, "n": len(data), "trainable": model.trainable_summary(),
                 "optimizer": opt, "epochs": a.epochs or 3, "before": tr.results.get("before"), "after": tr.results.get("after"),
                 "history": tr.history, "time_s": time.time() - t0})


def cmd_unlearn(a):
    if a.method not in list_algorithms(modality="seq"):
        raise SystemExit(f"--method must be one of {list_algorithms(modality='seq')}")
    print(short_spec(a.method))
    # the model is loaded with the requested trainable set (default: language model); the method's own `trainable`
    # hparam is only passed when --trainable was given explicitly, so RMU keeps its 3-matrix default
    model = load_model(a, trainable=a.trainable or "lm")
    if a.data:
        forget, retain = split_sets(a, load_set(a.data, a))
    else:
        forget, retain = load_set(a.forget, a), load_set(a.retain, a)
    test = load_set(a.test, a)
    if a.n_limit:
        forget, retain = forget[: a.n_limit], (retain or [])[: a.n_limit]
    print(f"[run_mm] forget {len(forget)}  retain {len(retain or [])}  test {len(test or [])}")
    col = make_collator(model, a)
    loaders = build_unlearn_loaders(forget, retain, col, batch_size=a.batch_size, retain_batch_size=a.retain_batch_size, seed=a.seed)
    ev = make_evaluator({"Forget": forget, "Retain": retain, "Test": test}, col, a)
    hp = parse_kv(a.hp)
    if a.trainable:
        hp.setdefault("trainable", a.trainable)
    if a.grad_ckpt:
        hp["grad_ckpt"] = True
    if a.optimizer:
        hp["optimizer"] = a.optimizer
    elif a.lr is not None:
        hp["optimizer"] = f"AdamW(lr={a.lr})"
    if a.epochs is not None:
        hp["n_epochs"] = a.epochs
    u, setup_kw = build_unlearner_split(a.method, model, hparams=hp)
    t0 = time.time()
    if get_kind(a.method) == "nontrainer":
        # closed-form methods (MM-ASRUSteer): fit(train_loaders, target_samples, collator)
        before = ev.evaluate(model, quick=False)
        print(ev.table(before, f"[{a.method}] before"))
        target = load_set(a.target, a) if a.target else None
        if a.method == "MM-ASRUSteer":
            if not target:
                raise SystemExit("MM-ASRUSteer needs --target <spec> (prompts with unseen images = knowledge-absence anchor)")
            u.fit(loaders, target, col)
        else:
            u.fit(loaders)
        after = ev.evaluate(model, quick=False)
        print(ev.table(after, f"[{a.method}] after"))
        u.results = {"before": before, "after": after, **getattr(u, "results", {})}
        u.history, u._resolved_hparams = [], u._resolved_hparams
        n_epochs = 0
    else:
        if hasattr(u, "set_collator"):
            u.set_collator(col)                      # GRPO methods generate their own rollouts
        u.set_evaluator(ev)
        n_epochs = setup_kw.pop("n_epochs")
        u.setup(**setup_kw)
        u.fit(loaders, n_epochs=n_epochs)
    if a.save_model:
        model.save_pretrained(os.path.join(a.out, "model"))
    res = {"command": "unlearn", "method": a.method, "model": a.model, "hparams": u._resolved_hparams, "setup": setup_kw,
           "n_epochs": n_epochs, "n_forget": len(forget), "n_retain": len(retain or []), "trainable": model.trainable_summary(),
           "before": u.results.get("before"), "after": u.results.get("after"),
           "delta": summarise_delta(u.results.get("before", {}), u.results.get("after", {})),
           "history": u.history, "stopped_early": u.results.get("stopped_early"), "iters": u.results.get("iters"),
           "nontrainer": {k: v for k, v in u.results.items() if k not in ("before", "after")} if get_kind(a.method) == "nontrainer" else None,
           "time_s": time.time() - t0}
    dump(a.out, res)
    print(json.dumps(res["delta"], indent=1))

def cmd_eval(a):
    model = load_model(a, trainable="none")
    roles = {}
    for item in a.data:
        role, _, spec = item.partition("=")
        roles[role] = load_set(spec, a)
    col = make_collator(model, a)
    ev = make_evaluator(roles, col, a)
    r = ev.evaluate(model, quick=a.no_gen)
    print(ev.table(r, f"[eval] {a.model}"))
    dump(a.out, {"command": "eval", "model": a.model, "roles": {k: len(v) for k, v in roles.items()}, "results": r})

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("finetune", help="fine-tune a checkpoint on a dataset (builds the model to unlearn from)")
    add_model_args(f); add_data_args(f); add_train_args(f); add_eval_args(f)
    f.add_argument("--data", required=True); f.add_argument("--eval-data", default=None)
    f.add_argument("--traj-every", type=int, default=1); f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_finetune)

    u = sub.add_parser("unlearn", help="run one MM-* method")
    add_model_args(u); add_data_args(u); add_train_args(u); add_eval_args(u)
    u.add_argument("--method", required=True)
    u.add_argument("--hp", nargs="*", default=[], help="method hparams key=value (JSON values); describe('MM-NPO') lists them")
    u.add_argument("--data", default=None, help="single set + --split"); u.add_argument("--split", default="group:0.1")
    u.add_argument("--forget", default=None); u.add_argument("--retain", default=None); u.add_argument("--test", default=None)
    u.add_argument("--target", default=None, help="MM-ASRUSteer: prompts with unseen images (knowledge-absence anchor)")
    u.add_argument("--save-model", action="store_true"); u.add_argument("--out", required=True)
    u.set_defaults(fn=cmd_unlearn)

    e = sub.add_parser("eval", help="evaluate a checkpoint on role=spec sets")
    add_model_args(e); add_data_args(e); add_eval_args(e)
    e.add_argument("--data", nargs="+", required=True, help="Role=spec ..."); e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_eval)

    a = ap.parse_args()
    if a.cmd == "unlearn" and not (a.data or (a.forget and a.retain is not None)):
        ap.error("unlearn needs --data (+--split) or --forget/--retain")
    torch.manual_seed(getattr(a, "seed", 0))
    a.fn(a)


if __name__ == "__main__":
    main()
