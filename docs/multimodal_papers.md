# Multimodal (VLM) unlearning 

Unlearning vision-language models (LVLM or CLIP-style).
Fidelity levels: 
**official** = ported from released code
**paper** = re-implemented from the paper's equations (no code consulted or none released)
**recipe** = the paper's contribution is data / protocol on top of an existing method
**not implemented** with the reason.

| venue | paper | what it is | torchunlearn entry | fidelity |
|---|---|---|---|---|
| NeurIPS 2024 | **SIU** -- Single Image Unlearning (Li et al., arXiv 2405.12523) | method: multifaceted FT data + Dual-Masked KL | `MM-SIU` (`trainers/seq/siu.py`), data builder `recipes/mmubench.build_siu_samples`, MMUBench metrics `metrics/mm_bench.MMUBenchEvaluator` | paper |
| ICLR 2025 | **FIUBench** (Ma et al., arXiv 2411.03554) | benchmark + 4 baselines (GA, GD, KL, PO) | recipe `recipes/fiubench.py`, metrics `FIUBenchEvaluator` (ROUGE-L, keyword EM, truth ratio, KS p vs retain model, Min-k, APE, judge GPT-eval); baselines `MM-GradAscent`, `MM-GradDiff`, `MM-KLMin`, `MM-PO` | paper (baseline losses as in the paper) |
| ICML 2025 | **SLUG** -- Single Layer Unlearning Gradient (Cai et al., arXiv 2407.11867) | method for CLIP, transplantable to LLaVA's vision tower | `CLIP-SLUG` (`nontrainers/slug.py`, `SLUG.apply_to_vlm`), `scripts/run_clip.py slug` | paper (importance / alignment / Pareto / binary search as described; official code consulted for layer naming) |
| NeurIPS 2025 (spotlight) | **ADU** -- Approximate Domain Unlearning for VLMs (Kawamura et al., arXiv 2510.08132) | task + method for CLIP (deep vision prompts, InstaPG, domain-disentangling loss) | `unlearn/clip/adu.py` (`ADU`), `recipes/domains.py`, `scripts/run_clip.py adu`; metrics Mem / For / H | paper |
| NeurIPS 2025 | **UMU-Bench** (Wang et al.) | benchmark (unimodal vs multimodal knowledge, 653 profiles) | recipe `recipes/mllmu_bench.py` (`eval_tasks`, `umu_qa_samples`), metrics `MLLMUBenchEvaluator` (classification / cloze / generation, split by modality) | recipe |
| ICLR 2026 | **Safety Mirage** (Chen et al., arXiv 2503.11832) | VLM safety via NPO / RMU on VLGuard | `MM-SafetyMirage-NPO`, `MM-SafetyMirage-RMU`, `build_vlguard_sets`, metrics `SafetyMirageEvaluator` (ASR / RR before and after the one-word attack) | recipe + paper |
| ICML 2026 | **ASRU** -- Activation Steering meets Reinforcement Unlearning (arXiv 2605.15687) | method: closed-form steering of one down-projection + GRPO with rule-based reward | `MM-ASRUSteer` (stage 1), `MM-ASRU` (stage 2, `trainers/seq/grpo.py` + `asru.py`) | paper (repo README consulted: layer 17, forget scale 3; GRPO is a single on-policy step per rollout batch) |
| ICML 2026 | **MLUBench + LUMoE** (arXiv 2606.12809) | lifelong benchmark + mixture of LoRA experts trained with PO, routed by entity | `trainers/seq/lumoe.py` (`LUMoE`, `NameMatchRouter`), recipe `recipes/mlubench.py`, metrics `MLUBenchEvaluator` (judge rejection / correctness) | paper (router = name match by default; the paper's GLM-4V router is pluggable) |
| ICML 2026 | **Beyond Sample-Level Forgetting** (Wang et al.) | causal decoupling + contrastive semantic editing | **not implemented**: no arXiv / code; OpenReview PDF not retrievable at survey time | - |
| ICML 2026 | **UnHype** (arXiv 2602.03410) | CLIP-guided hypernetwork LoRA unlearning for *diffusion* models | **not implemented**: text-to-image concept erasure, outside the VLM scope | - |
| NeurIPS 2025 workshop | CAGUL (arXiv 2510.07567) | training-free visual-token transformation | not implemented (workshop paper, no weight change) | - |

Methods that are evaluated as baselines in these papers and already exist: `MM-GradAscent`, `MM-GradDiff`,
`MM-NPO`, `MM-RMU`, `MM-DPO` (= IDK-DPO), plus `MM-KLMin` and `MM-PO` added here.

## Per-method notes

### MM-SIU (NeurIPS 2024)
Build the four-target data with `build_siu_samples(concept, image, new_name, visual_descriptions, facts, other_samples)`;
targets 1-2 are the Forget loader (image-conditioned), 3-4 the Retain loader.  `concept_names` gives the vocabulary
mask K_V; `meta["specified_text"]` of each sample gives the token mask K_S (the substitute name).  Reference model
copy is mandatory (`ref_mode="model"`).  Paper hyperparameters: `ce_weight=0.9, dmk_weight=0.75`, LoRA, Adam 3e-4,
batch 4, 6 steps.  MMUBench metrics: `MMUBenchEvaluator(concept, names, train_imgs, test_imgs, collator, ref_model, judge)`
gives efficacy EM, generality EM / G-Eval / C-Dis, fluency (masked perplexity), diversity.

### FIUBench (ICLR 2025)
Stage I: `MM-Finetune` on `fiubench.to_samples(rows)` (all 20 QA per identity, NLL, paper: lr 2e-5, 10 epochs).
Stage II: split by identity (`fiubench.split(ratio=0.05)`), run any `MM-*` method.  `FIUBenchEvaluator` needs
`meta` = keywords / paraphrased_answer / perturbed_answers / paraphrased_questions (filled by the recipe) and a
retain-only model for the KS test.  GPT-eval uses `metrics.judge` (local HF judge or OpenAI).

### CLIP-SLUG (ICML 2025)
`SLUG(clip).compute_gradients(forget_loader, retain_loader)` once; `pareto_front()`; `fit(eval_fn=...)` runs the binary
search on lambda with `eval_fn(model) -> (forget_acc, test_acc)`; `apply_to_vlm(seq_rmodel)` copies the updated
layer into an LVLM whose vision tower is the same CLIP (LLaVA-1.5 <-> openai/clip-vit-large-patch14-336).

### ADU (NeurIPS 2025)
`ADU(clip, class_names, n_domains, n_ctx=8, depth=9, gamma=30, lam=10).fit(loader, forget_domains, n_epochs=50, lr=0.0025)`.
The prompts are injected with forward pre-hooks on the CLIP vision encoder layers (installed only inside
fit / evaluate), so the CLIP weights never change; `evaluate` returns Mem / For / H.

### Safety Mirage (ICLR 2026)
`build_vlguard_sets(train.json, image_root, harmful_responses=None)` -> forget (unsafe) / retain (safe).  The NPO
variant unlearns the whole unsafe text (`full_labels`), the RMU variant needs harmful responses (the paper used
Llama-2-13B-Chat generations; pass them through `harmful_responses` keyed by `<id>#<k>`).

### ASRU (ICML 2026)
```
python scripts/run_mm.py unlearn --method MM-ASRUSteer --hp layer_id=17 lam=3 --target jsonl:unseen.jsonl ... --save-model
python scripts/run_mm.py unlearn --method MM-ASRU --model <steered> --hp group_size=4 kl_coef=0.1 --lr 1e-6 ...
```
Retain loader = the *boundary* set (retain samples similar to the forget set).  Reward thresholds are the paper's.

### LUMoE (ICML 2026)
```
lu = LUMoE(rmodel, collator, lora="r=16,alpha=32")
for task, (forget, retain) in recipes.mlubench.load_tasks(root).items(): lu.add_request(task, forget, retain)
MLUBenchEvaluator(forget, retain, collator, generate_fn=lu.generate, judge=judge).evaluate(rmodel)
```

## Candidates whose 2026 venue could not be verified (preprints, not implemented)
One Modality to Forget Them All / CrossInf (2607.16442), ViKeR (2601.22020), Null-Space Constrained Contrastive Visual
Forgetting (2605.05909), HFRU (2605.08031), Stochastic Meta-Unlearning (2607.18615), Knowledge Holes / SPAR
(2608.01849), AIM (2608.28312, EMNLP 2026 Findings), PAVA (2608.30649, EMNLP 2026 Findings), LEMUR (2608.11691,
inference-time), Attribute Unlearning (2608.01008, AAAI 2026).  Re-check neurips.cc/virtual/2026 once it is online.

## Verification
`python -m pytest tests/test_mm_papers.py -q` (CPU, tiny random-init LLaVA / CLIP): 12 tests -- every registered
MM-* paper method moves parameters, SIU's token mask zeroes the substitute-name positions, ASRU stage 1 rewrites the
chosen down-projection and stage 2 runs a GRPO step with the exact reward table, LUMoE trains two routed experts,
SLUG ranks layers / searches lambda / transplants into the tiny LLaVA, ADU trains prompts without touching CLIP and
reports Mem / For / H, all recipes parse the real record formats (MLLMU-Bench rows verified against the cached HF
dataset), all evaluators return their metric keys.
