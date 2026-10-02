#!/usr/bin/env python
"""CLIP-track unlearning CLI (dual-encoder models): SLUG (ICML 2025) and ADU (NeurIPS 2025).

  slug   python scripts/run_clip.py slug --model openai/clip-vit-large-patch14-336 --forget pairs_forget.jsonl \\
             --retain pairs_retain.jsonl --eval-forget forget_eval.jsonl --eval-test test_eval.jsonl --classes classes.txt \\
             --out runs/slug [--vlm llava-hf/llava-1.5-7b-hf --vlm-out runs/slug/llava]
         pairs jsonl: {"image": path, "text": caption}; eval jsonl: {"image": path, "label": class_idx}
  adu    python scripts/run_clip.py adu --model openai/clip-vit-base-patch16 --root /data/office_home --forget-domains Clipart,Art \\
             --shots 8 --epochs 50 --out runs/adu
         root/<domain>/<class>/*.jpg
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from torchunlearn import CLIPRobModel  # noqa: E402


def read_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def dump(out, obj):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    print("[run_clip] wrote", os.path.join(out, "results.json"))


def cmd_slug(a):
    from torch.utils.data import DataLoader
    from torchunlearn.unlearn.nontrainers.slug import SLUG, make_clip_loader
    from torchunlearn.unlearn.clip import make_domain_loader
    if a.mem_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    model = CLIPRobModel.from_pretrained(a.model, dtype=a.dtype, local_files_only=a.local_files_only)
    pairs = lambda p: [(r["image"], r["text"]) for r in read_jsonl(p)]
    f = make_clip_loader(pairs(a.forget), model, batch_size=a.batch_size, image_root=a.image_root)
    r = make_clip_loader(pairs(a.retain), model, batch_size=a.batch_size, image_root=a.image_root)
    classes = [l.strip() for l in open(a.classes) if l.strip()]
    texts = [a.template.format(x) for x in classes]
    items = lambda p: [(x["image"], int(x["label"]), 0) for x in read_jsonl(p)]
    ef = make_domain_loader(items(a.eval_forget), model, batch_size=a.batch_size, shuffle=False, image_root=a.image_root)
    et = make_domain_loader(items(a.eval_test), model, batch_size=a.batch_size, shuffle=False, image_root=a.image_root)

    def eval_fn(model):
        return model.zero_shot_accuracy(ef, texts), model.zero_shot_accuracy(et, texts)
    slug = SLUG(model, forget_loss=a.forget_loss, layer_regex=a.layer_regex)
    t0 = time.time()
    slug.compute_gradients(f, r)
    slug.fit(layer=a.layer, lam=a.lam, eval_fn=None if a.lam is not None else eval_fn, forget_target=a.forget_target,
             test_drop_tol=a.test_drop_tol, lam_init=a.lam_init, n_search=a.n_search)
    res = {"command": "slug", "model": a.model, "layer_table": slug.table, "pareto": slug.pareto_front(), **slug.results, "time_s": time.time() - t0}
    if a.save_model:
        model.save_pretrained(os.path.join(a.out, "clip"))
    if a.vlm:
        from torchunlearn import SeqRobModel
        v = SeqRobModel.from_pretrained(a.vlm, dtype="bf16", device_map="auto" if torch.cuda.is_available() else None,
                                        trainable="none", local_files_only=a.local_files_only)
        res["vlm_updated_params"] = slug.apply_to_vlm(v)
        v.save_pretrained(a.vlm_out or os.path.join(a.out, "vlm"))
    dump(a.out, res)


def cmd_adu(a):
    from torchunlearn.unlearn.clip import ADU, make_domain_loader
    from torchunlearn.unlearn.recipes.domains import few_shot, from_domain_folders, train_test_split
    if a.mem_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    model = CLIPRobModel.from_pretrained(a.model, dtype=a.dtype, local_files_only=a.local_files_only)
    items, classes, domains = from_domain_folders(a.root)
    fd = [domains.index(d) for d in a.forget_domains.split(",")]
    train, test = train_test_split(items, a.test_ratio, a.seed)
    train = few_shot(train, a.shots, a.seed)
    print(f"[adu] classes {len(classes)} domains {domains} forget {fd} train {len(train)} test {len(test)}")
    adu = ADU(model, classes, n_domains=len(domains), template=a.template, n_ctx=a.n_ctx, depth=a.depth, insta_layer=a.insta_layer,
              gamma=a.gamma, lam=a.lam)
    te = make_domain_loader(test, model, batch_size=a.batch_size, shuffle=False)
    before = adu.evaluate(te, fd)
    print("[adu] before", before)
    t0 = time.time()
    adu.fit(make_domain_loader(train, model, batch_size=a.batch_size, shuffle=True), fd, n_epochs=a.epochs, lr=a.lr)
    after = adu.evaluate(te, fd)
    print("[adu] after", after)
    adu.save(os.path.join(a.out, "adu.pt"))
    dump(a.out, {"command": "adu", "model": a.model, "classes": classes, "domains": domains, "forget_domains": fd,
                 "before": before, "after": after, "history": adu.history, "time_s": time.time() - t0})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("slug", "adu"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True); p.add_argument("--dtype", default="fp32")
        p.add_argument("--batch-size", type=int, default=32); p.add_argument("--mem-fraction", type=float, default=None)
        p.add_argument("--local-files-only", action="store_true"); p.add_argument("--out", required=True)
        p.add_argument("--template", default="a photo of a {}."); p.add_argument("--seed", type=int, default=0)
        if name == "slug":
            p.add_argument("--forget", required=True); p.add_argument("--retain", required=True); p.add_argument("--image-root", default=None)
            p.add_argument("--eval-forget", required=True); p.add_argument("--eval-test", required=True); p.add_argument("--classes", required=True)
            p.add_argument("--forget-loss", default="cosine", choices=["cosine", "contrastive"])
            p.add_argument("--layer-regex", default=r"vision_model\.encoder\.layers\.\d+\.(self_attn|mlp)\..*weight")
            p.add_argument("--layer", default=None); p.add_argument("--lam", type=float, default=None)
            p.add_argument("--forget-target", type=float, default=0.05); p.add_argument("--test-drop-tol", type=float, default=0.02)
            p.add_argument("--lam-init", type=float, default=1.0); p.add_argument("--n-search", type=int, default=10)
            p.add_argument("--save-model", action="store_true"); p.add_argument("--vlm", default=None); p.add_argument("--vlm-out", default=None)
            p.set_defaults(fn=cmd_slug)
        else:
            p.add_argument("--root", required=True); p.add_argument("--forget-domains", required=True)
            p.add_argument("--shots", type=int, default=8); p.add_argument("--test-ratio", type=float, default=0.2)
            p.add_argument("--epochs", type=int, default=50); p.add_argument("--lr", type=float, default=0.0025)
            p.add_argument("--n-ctx", type=int, default=8); p.add_argument("--depth", type=int, default=9); p.add_argument("--insta-layer", type=int, default=1)
            p.add_argument("--gamma", type=float, default=30.0); p.add_argument("--lam", type=float, default=10.0)
            p.set_defaults(fn=cmd_adu)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    a.fn(a)


if __name__ == "__main__":
    main()
