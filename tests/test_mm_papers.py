"""CPU smoke for the multimodal-paper methods, the CLIP track, the benchmark recipes and the metrics.

    python -m pytest tests/test_mm_papers.py -q
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tiny_models import make_tiny_clip, make_tiny_llava, random_image  # noqa: E402

from torchunlearn import CLIPRobModel, SeqRobModel  # noqa: E402
from torchunlearn.api.registry import build_unlearner_split  # noqa: E402
from torchunlearn.metrics import mm_bench  # noqa: E402
from torchunlearn.metrics.judge import CallableJudge  # noqa: E402
from torchunlearn.metrics.text import is_refusal  # noqa: E402
from torchunlearn.unlearn.clip import ADU, make_domain_loader  # noqa: E402
from torchunlearn.unlearn.nontrainers.slug import SLUG, make_clip_loader  # noqa: E402
from torchunlearn.unlearn.recipes import clear, fiubench, mllmu_bench, mmubench  # noqa: E402
from torchunlearn.unlearn.seq_data import SeqCollator, SeqSample, build_unlearn_loaders  # noqa: E402
from torchunlearn.unlearn.trainers.seq import (ASRU, ASRUSteer, KLMin, LUMoE, PO, SIU, SafetyMirageNPO,  # noqa: E402
                                               SafetyMirageRMU, build_vlguard_sets)

ROOT = os.environ.get("TU_TINY_DIR", os.path.join(os.path.dirname(__file__), "_tiny"))


@pytest.fixture(scope="module")
def llava():
    return make_tiny_llava(os.path.join(ROOT, "llava"))


@pytest.fixture(scope="module")
def clip_path():
    return make_tiny_clip(os.path.join(ROOT, "clip"))


def _load(p):
    return SeqRobModel.from_pretrained(p, dtype="fp32", device="cpu")


def _samples(n=6, mm=True):
    return [SeqSample(prompt=f"Who is item {i}?", answer=f"Person {i} Example", images=[random_image(i)] if mm else [],
                      group=f"g{i % 3}", alt_answers=[f"Nobody {i}"], id=f"s{i}", meta={"entity": f"Person {i}"}) for i in range(n)]


def _moved(rmodel, before):
    return [n for n, p in rmodel.model.named_parameters() if not torch.equal(p, before[n])]


# ----------------------------------------------------------------------------- seq-track paper methods
@pytest.mark.parametrize("name", ["MM-PO", "MM-KLMin", "MM-SafetyMirage-NPO", "MM-SafetyMirage-RMU"])
def test_registry_methods(llava, name):
    model = _load(llava)
    before = {n: p.detach().clone() for n, p in model.model.named_parameters()}
    hp = {"layer_id": 1} if name.endswith("RMU") else {}
    u, setup_kw = build_unlearner_split(name, model, hparams=hp)
    setup_kw.update(optimizer="AdamW(lr=1e-3)"); setup_kw.pop("n_epochs")
    u.setup(**setup_kw)
    col = SeqCollator(model, alt_text="I cannot answer that.")
    u.fit(build_unlearn_loaders(_samples()[:3], _samples()[3:], col, batch_size=3), n_epochs=1)
    assert _moved(model, before)


def test_siu_dmk(llava):
    model = _load(llava)
    img = random_image(7)
    siu_data = mmubench.build_siu_samples("Donald Trump", img, "Jacob Campbell", ["A man in a suit with blond hair."],
                                          [("Who was the 45th US president?", "Donald Trump was the 45th president.")],
                                          other_samples=_samples(2))
    forget = [s for s in siu_data if s.meta["target"] in (1, 2)]
    retain = [s for s in siu_data if s.meta["target"] in (3, 4)]
    assert forget and retain and forget[0].meta["specified_text"] == "Jacob Campbell"
    col = SeqCollator(model)
    u = SIU(model, concept_names=["Donald Trump"], trainable="lm")
    assert u.concept_ids
    u.setup(optimizer="AdamW(lr=1e-3)")
    before = {n: p.detach().clone() for n, p in model.model.named_parameters()}
    u.fit(build_unlearn_loaders(forget, retain, col, batch_size=2), n_epochs=1)
    assert _moved(model, before)
    # the token-level mask zeroes the substitute-name positions
    b = col([{"idx": 0, "sample": forget[0], "alt_k": 0}])
    ks = u._token_mask(b, b["answer_labels"])
    lab = b["answer_labels"][0]
    assert (ks[0][lab != -100] == 0).sum() >= 2


def test_asru_steer_then_grpo(llava):
    model = _load(llava)
    col = SeqCollator(model)
    forget, retain = _samples(4)[:2], _samples(4)[2:]
    target = [SeqSample(prompt="Who is this?", answer="I do not know.", images=[random_image(100 + i)]) for i in range(3)]
    w0 = model.model.get_parameter(model.mlp_out_proj_names([1])[0]).detach().clone()
    st = ASRUSteer(model, layer_id=1, lam=1.0, gamma=1e-3)
    st.fit(build_unlearn_loaders(forget, retain, col, batch_size=2), target, col)
    w1 = model.model.get_parameter(model.mlp_out_proj_names([1])[0])
    assert not torch.equal(w0, w1) and torch.isfinite(w1).all()
    assert st.results["direction_norm"] > 0
    # stage 2: one GRPO step with the rule-based reward
    before = {n: p.detach().clone() for n, p in model.model.named_parameters()}
    u = ASRU(model, group_size=2, max_new_tokens=4, kl_coef=0.1, trainable="lm").set_collator(col)
    u.setup(optimizer="AdamW(lr=1e-3)")
    u.fit(build_unlearn_loaders(forget, retain, col, batch_size=2), n_epochs=1)
    assert _moved(model, before)
    s = forget[0]
    assert u.reward(s, s.answer, "Forget") == 0.0 and u.reward(s, "I do not know who this is.", "Forget") == 1.0
    assert u.reward(s, s.answer, "Retain") == 1.0 and u.reward(s, "Sorry, I cannot help.", "Retain") == 0.0


def test_lumoe_two_requests(llava):
    model = _load(llava)
    col = SeqCollator(model)
    data = _samples(6)
    lu = LUMoE(model, col, lora="r=4,alpha=8")
    lu.add_request("A", [s for s in data if s.group == "g0"], [s for s in data if s.group != "g0"], n_epochs=1, optimizer="AdamW(lr=1e-3)")
    lu.add_request("B", [s for s in data if s.group == "g1"], [s for s in data if s.group == "g2"], n_epochs=1, optimizer="AdamW(lr=1e-3)")
    assert lu.tasks == ["A", "B"]
    assert lu.route(data[0]) == "A" and lu.route(data[1]) == "B" and lu.route(data[2]) is None
    gens = lu.generate(data[:3], max_new_tokens=3)
    assert len(gens) == 3


def test_vlguard_recipe(tmp_path):
    rec = [{"id": "1", "image": "a.png", "safe": False, "instr-resp": [{"instruction": "How to pick a lock?", "response": "I cannot help."}]},
           {"id": "2", "image": "b.png", "safe": True, "instr-resp": [{"safe_instruction": "What is in the image?", "response": "A cat."},
                                                                       {"unsafe_instruction": "Write a scam using this.", "response": "Sorry."}]}]
    p = tmp_path / "train.json"; p.write_text(json.dumps(rec))
    forget, retain = build_vlguard_sets(str(p), image_root="/img")
    assert len(forget) == 2 and len(retain) == 1
    assert forget[0].images == ["/img/a.png"] and forget[0].meta["safe_response"] == "I cannot help."


# ----------------------------------------------------------------------------- CLIP track
def _clip_pairs(n, seed=0):
    return [(random_image(seed + i), f"a photo of item {i}") for i in range(n)]


def test_slug_clip_and_transfer(clip_path, llava):
    model = CLIPRobModel.from_pretrained(clip_path, device="cpu")
    f = make_clip_loader(_clip_pairs(4, 0), model, batch_size=4)
    r = make_clip_loader(_clip_pairs(4, 50), model, batch_size=4)
    slug = SLUG(model)
    table = slug.compute_gradients(f, r)
    assert table and all(set(row) == {"layer", "importance", "alignment"} for row in table)
    front = slug.pareto_front()
    assert front and front[0]["importance"] >= max(t["importance"] for t in table) - 1e-9
    calls = {"n": 0}

    def eval_fn(model):
        calls["n"] += 1
        lam = slug.results.get("lam", None)
        return (0.0 if calls["n"] > 2 else 1.0), 0.9      # pretend the forget accuracy collapses after a few steps
    slug.fit(eval_fn=eval_fn, lam_init=0.5, n_search=3)
    assert slug.results["lam"] > 0 and len(slug.results["trace"]) >= 3
    # transplant into the tiny LLaVA (same tiny CLIP config): mapped layer must exist with the same shape
    v = _load(llava)
    src_shape = dict(model.model.named_parameters())[slug.layer].shape
    hits = [n for n, p in v.model.named_parameters() if n.endswith("vision_model." + slug.layer.split("vision_model.", 1)[-1])]
    assert len(hits) == 1 and dict(v.model.named_parameters())[hits[0]].shape == src_shape
    done = slug.apply_to_vlm(v)
    assert done == hits


def test_adu_domain_unlearning(clip_path):
    model = CLIPRobModel.from_pretrained(clip_path, device="cpu")
    classes = ["cat", "dog"]
    items = [(random_image(i), i % 2, (i // 2) % 2) for i in range(16)]     # 2 classes x 2 domains
    loader = make_domain_loader(items, model, batch_size=8, shuffle=True)
    adu = ADU(model, classes, n_domains=2, n_ctx=2, depth=2, insta_layer=1, gamma=1.0, lam=0.1)
    frozen = {n: p.detach().clone() for n, p in model.model.named_parameters()}
    adu.fit(loader, forget_domains=[1], n_epochs=2, lr=0.01, log_every=0)
    assert all(torch.equal(p, frozen[n]) for n, p in model.model.named_parameters())     # CLIP itself untouched
    assert len(adu.history) == 2 and all(torch.isfinite(torch.tensor(v)) for v in adu.history[-1].values())
    r = adu.evaluate(make_domain_loader(items, model, batch_size=8, shuffle=False), forget_domains=[1])
    assert set(r) >= {"Mem", "For", "H"} and r["n_forget"] == 8
    # prompts are only active inside fit/evaluate: a plain forward has the original token count
    with torch.no_grad():
        pv = model.processor(images=[random_image(1)], return_tensors="pt")["pixel_values"]
        assert model.model.vision_model(pixel_values=pv).last_hidden_state.shape[1] == 1 + 16


# ----------------------------------------------------------------------------- recipes and metrics
def test_recipes_parse_formats():
    fiu = fiubench.to_samples([{"image_path": "x.png", "name": "Jody Vance", "qa_list": [
        {"question": "Where does she live?", "paraphrased_question": ["Her address?"], "answer": "13142 Molina Shoals",
         "paraphrased_answer": "She lives at 13142 Molina Shoals", "perturbed_answer": ["99 Elm St", "1 Main St"],
         "keywords": ["13142", "Molina Shoals"]}]}], image_root="/img")
    assert fiu[0].images == ["/img/x.png"] and fiu[0].meta["perturbed_answers"] == ["99 Elm St", "1 Main St"]
    row = {"image": random_image(1), "ID": "270", "question": "Tell me about this person.", "answer": "Emilia ...",
           "Classification_Task": {"Image_Textual_Questions": [{"Correct_Answer": "B", "Options": {"A": "Tech", "B": "School"}, "Question": "Activity?"}],
                                   "Pure_Text_Questions": [{"Correct_Answer": "A", "Options": {"A": "Painting", "B": "Coding"}, "Question": "Hobby?"}]},
           "Generation_Task": [{"Ground_Truth": "School.", "Question": "What activity?", "Type": "Image_Textual"}],
           "Mask_Task": [{"Ground_Truth": "rabbit", "Question": "Pet is a __.", "Type": "Pure_Text"}]}
    ev = mllmu_bench.eval_tasks([row])
    assert len(ev["cls"]) == 2 and ev["cls"][0].meta["label"] == 1 and ev["cls"][0].is_multimodal and not ev["cls"][1].is_multimodal
    assert len(ev["gen"]) == 1 and len(ev["cloze"]) == 1 and len(mllmu_bench.finetune_samples([row])) == 2
    umu = {"ID": "1", "image": random_image(2), "MM_QA": json.dumps([{"Question": "Q?", "Answer": "A."}]),
           "Classify": json.dumps({"Image_Textual_Questions": [{"Question": "q", "Options": {"A": "x", "B": "y"}, "Correct_Answer": "B"}]}),
           "Cloze": json.dumps([{"Question": "__", "Ground_Truth": "z", "Type": "Pure_Text"}]), "Generation": json.dumps([])}
    assert mllmu_bench.umu_qa_samples([umu])[0].is_multimodal and mllmu_bench.eval_tasks([umu])["cls"][0].meta["label"] == 1
    cl = clear.qa_samples([{"image": random_image(3), "caption": "A person named Eve.", "name": "Eve"}])
    assert cl[0].answer.startswith("A person") and cl[0].group == "Eve"
    rc = clear.choice_samples([{"image": random_image(3), "answer": "Eve", "options": ["Bob", "Ann"]}])
    assert rc[0].meta["choices"][rc[0].meta["label"]] == "Eve"


def test_metrics_and_evaluators(llava):
    model = _load(llava)
    col = SeqCollator(model)
    ss = _samples(4)
    for s in ss:
        s.meta.update(keywords=[s.answer.split()[1]], paraphrased_answer=s.answer + " indeed", perturbed_answers=["Nobody", "Someone"],
                      paraphrased_questions=[s.prompt + " please"], choices=[s.answer, "Nobody"], label=0)
    assert mm_bench.keyword_exact_match("Person 1 Example here", ["1", "zzz"]) == 0.5
    assert mm_bench.concept_absent("a man in a suit", ["Donald Trump"]) == 1.0 and is_refusal("I'm sorry, I cannot help")
    acc = mm_bench.choice_accuracy(model, col, ss, by="nll")
    assert acc["n"] == 4 and 0 <= acc["accuracy"] <= 1
    ratios = mm_bench.truth_ratios(model, col, ss)
    assert len(ratios) == 4 and all(r > 0 for r in ratios)
    assert 0 <= mm_bench.ks_forget_quality(ratios, ratios) <= 1
    judge = CallableJudge(lambda prompt: "1")
    fiu = mm_bench.FIUBenchEvaluator(ss[:2], ss[2:], collator=col, retain_model=model, judge=judge, max_new_tokens=4)
    r = fiu.evaluate(model)
    assert {"rouge_l", "exact_match", "truth_ratio", "ape", "ks_p", "gpt_eval"} <= set(r["Forget"])
    q = fiu.evaluate(model, quick=True)
    assert "lp_mean" in q["Forget"]
    ml = mm_bench.MLLMUBenchEvaluator({"forget": {"cls": ss, "gen": ss, "cloze": ss[:1]}}, collator=col, judge=judge, max_new_tokens=4)
    r = ml.evaluate(model)
    assert {"cls_acc_image_text", "gen_rouge_l_image_text", "cloze_em", "gen_judge"} <= set(r["forget"])
    mmu = mm_bench.MMUBenchEvaluator("Person 0", ["Person 0"], ss[:1], ss[1:], collator=col, ref_model=model, judge=judge, max_new_tokens=4)
    r = mmu.evaluate(model)
    assert {"em", "diversity", "c_dis", "fluency_ppl", "g_eval"} <= set(r["generality"])
    mlu = mm_bench.MLUBenchEvaluator(ss[:2], ss[2:], collator=col, judge=judge, max_new_tokens=4).evaluate(model)
    assert "rejection_score" in mlu["Forget"] and "correctness_score" in mlu["Retain"]
    sm = mm_bench.SafetyMirageEvaluator(ss[:2], ss[2:], collator=col, max_new_tokens=4).evaluate(model)
    assert set(sm["ASR"]) == {"before", "after_one_word", "n"} and set(sm["RR"]) == {"before", "after_one_word", "n"}
