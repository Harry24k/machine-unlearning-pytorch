"""CPU smoke for the sequence / multimodal unlearning stack (tiny random-init LLaVA + tiny Llama).

    python -m pytest tests/test_seq_smoke.py -q

What is pinned down here
  * answer-only labels never touch image tokens or the prompt, and start at the first answer token
  * changing the answer changes the loss (the mask is alive)
  * every registered MM-* method runs setup -> fit(1 epoch) -> evaluate on an image+text model and on a text model,
    the cost is finite, and only the parameters it declares trainable move
  * MM-Finetune on a single loader memorises (forget lp goes up), GradAscent then pushes it down
  * save_pretrained -> from_pretrained round-trips the logits
  * trainable specs select the right component
"""
from __future__ import annotations

import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tiny_models import make_tiny_llava, make_tiny_llm, random_image  # noqa: E402

from torchunlearn import SeqRobModel, SeqUnlearningEvaluator  # noqa: E402
from torchunlearn.api.registry import build_unlearner_split, list_algorithms  # noqa: E402
from torchunlearn.unlearn.seq_data import (SeqCollator, SeqDataset, SeqSample, build_unlearn_loaders,  # noqa: E402
                                           make_loader, split_forget_retain, strip_images)

ROOT = os.environ.get("TU_TINY_DIR", os.path.join(os.path.dirname(__file__), "_tiny"))


@pytest.fixture(scope="module")
def paths():
    return {"llava": make_tiny_llava(os.path.join(ROOT, "llava")), "llm": make_tiny_llm(os.path.join(ROOT, "llm"))}


def _samples(multimodal: bool, n: int = 8):
    out = []
    for i in range(n):
        out.append(SeqSample(prompt=f"Question number {i}: who owns item {i}?", answer=f"Person {i} Example owns it",
                             images=[random_image(i)] if multimodal else [], group=f"g{i % 4}",
                             alt_answers=[f"Nobody {i}", f"Someone else {i}"], id=f"s{i}"))
    return out


def _load(path):
    return SeqRobModel.from_pretrained(path, dtype="fp32", device="cpu")


# ----------------------------------------------------------------------------- masks
@pytest.mark.parametrize("which", ["llava", "llm"])
def test_answer_mask(paths, which):
    model = _load(paths[which])
    ss = _samples(which == "llava", 3)
    col = SeqCollator(model, alt_text="[REDACTED]")
    b = col([SeqDataset(ss)[i] for i in range(3)])
    tok = model.tokenizer
    for i, s in enumerate(ss):
        ids, lab = b["input_ids"][i], b["answer_labels"][i]
        if model.image_token_ids:
            assert int((ids == model.image_token_ids[0]).sum()) == 16            # 32px / patch 8 = 16 tokens
            assert not ((lab != -100) & (ids == model.image_token_ids[0])).any()
            assert not ((b["full_labels"][i] != -100) & (ids == model.image_token_ids[0])).any()
        dec = tok.decode(lab[lab != -100]).strip()
        assert dec.startswith(s.answer), dec
        assert "Question" not in dec
        al = b["alt_labels"][i]          # sample alternates win over alt_text; alt_k = (epoch + i) % n_alt
        assert tok.decode(al[al != -100]).strip().startswith(s.alt_answers[i % 2])
    nb = SeqCollator(model, alt_text="[REDACTED]")([SeqDataset([SeqSample("q?", "ans", ss[0].images)])[0]])
    assert tok.decode(nb["alt_labels"][0][nb["alt_labels"][0] != -100]).strip().startswith("[REDACTED]")
    assert b["labels"].equal(b["answer_labels"])
    full = SeqCollator(model, loss_on="full")([SeqDataset(ss)[0]])
    assert full["labels"].equal(full["full_labels"]) and (full["full_labels"] != -100).sum() > (b["answer_labels"][0] != -100).sum()


def test_loss_depends_on_answer(paths):
    model = _load(paths["llava"])
    col = SeqCollator(model)
    a = SeqSample("What is this?", "a red square", [random_image(1)])
    b = SeqSample("What is this?", "a blue circle", [random_image(1)])
    with torch.no_grad():
        la = model(**{k: v for k, v in col([SeqDataset([a])[0]]).items() if k in ("input_ids", "attention_mask", "pixel_values", "labels")}).loss
        lb = model(**{k: v for k, v in col([SeqDataset([b])[0]]).items() if k in ("input_ids", "attention_mask", "pixel_values", "labels")}).loss
    assert torch.isfinite(la) and torch.isfinite(lb) and not torch.isclose(la, lb)


# ----------------------------------------------------------------------------- trainable specs
def test_trainable_specs(paths):
    model = _load(paths["llava"])
    assert model.apply_trainable("lm")["vision"] == 0 and model.apply_trainable("lm")["lm"] > 0
    assert model.apply_trainable("vision")["lm"] == 0 and model.apply_trainable("vision")["vision"] > 0
    assert model.apply_trainable("projector")["projector"] > 0 and model.apply_trainable("projector")["lm"] == 0
    s = model.apply_trainable("lm+projector")
    assert s["lm"] > 0 and s["projector"] > 0 and s["vision"] == 0
    model.apply_trainable("lora:r=4,alpha=8,target=lm")
    s = model.trainable_summary()
    assert 0 < s["lm"] < 3_000_000 and s["vision"] == 0
    with pytest.raises(Exception):
        _load(paths["llm"]).apply_trainable("vision")


# ----------------------------------------------------------------------------- every method
def _run_method(name, rmodel, forget, retain, n_epochs=1, hp=None):
    col = SeqCollator(rmodel, alt_text="[REDACTED]")
    loaders = build_unlearn_loaders(forget, retain, col, batch_size=2)
    ev = SeqUnlearningEvaluator({"Forget": forget, "Retain": retain, "Forget_noimg": strip_images(forget)}, col,
                                batch_size=4, max_new_tokens=4, gen_kwargs={"use_cache": True})
    hp = dict(hp or {})
    if name == "MM-RMU":
        hp.setdefault("layer_id", 1)
    hp.setdefault("ref_mode", "cache")
    u, setup_kw = build_unlearner_split(name, rmodel, hparams=hp)
    u.set_evaluator(ev)
    setup_kw.update(optimizer="AdamW(lr=1e-3)"); setup_kw.pop("n_epochs")
    u.setup(**setup_kw)
    u.fit(loaders, n_epochs=n_epochs, record_type="Epoch")
    return u


# methods that need extra inputs (collator / target set / concept names / deeper models) are covered by tests/test_mm_papers.py
_PAPER_ONLY = {"MM-Finetune", "MM-ASRU", "MM-ASRUSteer", "MM-SafetyMirage-NPO", "MM-SafetyMirage-RMU", "MM-SIU"}


@pytest.mark.parametrize("name", [n for n in list_algorithms(modality="seq") if n.startswith("MM-") and n not in _PAPER_ONLY])
def test_methods_image_text(paths, name):
    rmodel = _load(paths["llava"])
    before = {n: p.detach().clone() for n, p in rmodel.model.named_parameters()}
    forget, retain = _samples(True, 8)[:4], _samples(True, 8)[4:]
    u = _run_method(name, rmodel, forget, retain)
    assert "after" in u.results and "Forget" in u.results["after"]
    for role in ("Forget", "Retain", "Forget_noimg"):
        r = u.results["after"][role]
        assert r["n"] == 4 and all(torch.isfinite(torch.tensor(r[k])) for k in ("lp_mean", "nll_seq", "mink", "rouge_l"))
        assert len(r["gen"]) == 4
    assert u.history and all(torch.isfinite(torch.tensor(v)) for v in u.history[-1].values())
    changed = [n for n, p in rmodel.model.named_parameters() if not torch.equal(p, before[n])]
    assert changed, "no parameter moved"
    if name == "MM-RMU":
        assert set(changed) == set(rmodel.mlp_out_proj_names([0, 1]))
    else:
        assert any(rmodel.component_of(n) == "lm" for n in changed)
        assert all(torch.isfinite(p).all() for p in rmodel.model.parameters())


@pytest.mark.parametrize("name", ["MM-NPO", "MM-RMU", "MM-AltPO", "MM-FLAT", "MM-PDU"])
def test_methods_text(paths, name):
    rmodel = _load(paths["llm"])
    forget, retain = _samples(False, 8)[:4], _samples(False, 8)[4:]
    u = _run_method(name, rmodel, forget, retain, hp={"primal_dual": True} if name == "MM-PDU" else None)
    assert "Forget" in u.results["after"] and u.results["after"]["Forget"]["n"] == 4


def test_ref_mode_model_matches_cache(paths):
    """NPO with a frozen reference copy must give the same first-step cost as the cached statistics."""
    costs = []
    for mode in ("cache", "model"):
        torch.manual_seed(0)
        rmodel = _load(paths["llava"])
        forget, retain = _samples(True, 4)[:2], _samples(True, 4)[2:]
        col = SeqCollator(rmodel)
        loaders = build_unlearn_loaders(forget, retain, col, batch_size=2, seed=0)
        from torchunlearn.unlearn.trainers.seq import NPO
        u = NPO(rmodel, ref_mode=mode, alpha=1.0).setup(optimizer="AdamW(lr=0.0)")
        u._prepare_reference(loaders)
        batch = next(iter(loaders))
        costs.append(float(u.calculate_cost(batch)))
    assert abs(costs[0] - costs[1]) < 1e-3, costs


# ----------------------------------------------------------------------------- finetune -> unlearn -> save
def test_finetune_then_unlearn_then_roundtrip(paths, tmp_path):
    torch.manual_seed(0)
    rmodel = _load(paths["llava"])
    data = _samples(True, 8)
    forget, retain = split_forget_retain(data, by="group", groups=["g0"])
    assert len(forget) == 2 and len(retain) == 6
    col = SeqCollator(rmodel)
    ev = SeqUnlearningEvaluator({"Forget": forget, "Retain": retain}, col, batch_size=4, gen=False)
    lp0 = ev.evaluate(rmodel, quick=True)["Forget"]["lp_mean"]
    from torchunlearn.unlearn.trainers.seq import SeqFinetune, GradAscent
    SeqFinetune(rmodel, trainable="lm").setup(optimizer="AdamW(lr=5e-3)").fit(make_loader(data, col, batch_size=4), n_epochs=15)
    lp1 = ev.evaluate(rmodel, quick=True)["Forget"]["lp_mean"]
    assert lp1 > lp0 + 0.5, (lp0, lp1)
    GradAscent(rmodel, trainable="lm").set_evaluator(ev).setup(optimizer="AdamW(lr=5e-3)").fit(
        build_unlearn_loaders(forget, retain, col, batch_size=2), n_epochs=3)
    lp2 = ev.evaluate(rmodel, quick=True)["Forget"]["lp_mean"]
    assert lp2 < lp1 - 0.5, (lp1, lp2)
    out = tmp_path / "unlearned"
    rmodel.save_pretrained(str(out))
    re = _load(str(out))
    b = col([SeqDataset(forget)[0]])
    inp = {k: v for k, v in b.items() if k in ("input_ids", "attention_mask", "pixel_values")}
    with torch.no_grad():
        assert torch.allclose(rmodel(**inp).logits, re(**inp).logits, atol=1e-5)
    assert re.image_token_ids == rmodel.image_token_ids


def test_lora_roundtrip(paths, tmp_path):
    rmodel = _load(paths["llava"])
    rmodel.apply_trainable("lora:r=4,alpha=8,target=lm")
    forget, retain = _samples(True, 4)[:2], _samples(True, 4)[2:]
    col = SeqCollator(rmodel)
    from torchunlearn.unlearn.trainers.seq import GradDiff
    GradDiff(rmodel).setup(optimizer="AdamW(lr=1e-3)").fit(build_unlearn_loaders(forget, retain, col, batch_size=2), n_epochs=1)
    out = tmp_path / "lora"
    rmodel.save_pretrained(str(out))            # adapter_config.json + base path -> from_pretrained merges it
    assert (out / "adapter_config.json").exists()
    merged = _load(str(out))
    assert merged.modality == "image_text"


def test_grad_ckpt_moves_only_declared_params(paths):
    """Non-reentrant checkpointing must still deliver gradients to a few matrices deep inside a frozen model."""
    rmodel = _load(paths["llava"])
    forget, retain = _samples(True, 4)[:2], _samples(True, 4)[2:]
    col = SeqCollator(rmodel)
    from torchunlearn.unlearn.trainers.seq import RMU
    before = {n: p.detach().clone() for n, p in rmodel.model.named_parameters()}
    RMU(rmodel, layer_id=1, grad_ckpt=True).setup(optimizer="AdamW(lr=1e-3)").fit(
        build_unlearn_loaders(forget, retain, col, batch_size=2), n_epochs=1)
    changed = {n for n, p in rmodel.model.named_parameters() if not torch.equal(p, before[n])}
    assert changed == set(rmodel.mlp_out_proj_names([0, 1]))
