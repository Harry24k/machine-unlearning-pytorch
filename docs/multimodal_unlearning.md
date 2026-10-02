# Multimodal (VLM) and LLM unlearning

Implementation of every token-level method serves plain LLMs **and** vision-language models.
The modality lives in the model wrapper and the collator; the methods only see `labels` (-100 outside the
answer) and forward whatever other tensors the processor produced (`pixel_values`, `image_grid_thw`, ...).

## Files

| file | role |
|---|---|
| `torchunlearn/nn/seqmodel.py` | `SeqRobModel.from_pretrained(path_or_id, modality="auto", trainable=...)` — any HF causal LM or `AutoModelForImageTextToText` checkpoint (base, fine-tuned, or a PEFT adapter dir, merged on load). `apply_trainable("lm" / "vision" / "projector" / "lm+projector" / "lora:r=16,target=lm" / "re:<regex>")`, `decoder_layers()`, `image_token_ids`, `save_pretrained`. |
| `torchunlearn/unlearn/seq_data.py` | `SeqSample(prompt, answer, images, group, alt_answers)`; recipes `from_jsonl / from_llava_json / from_hf / load_spec`; `split_forget_retain(by="random"|"group"|"ids")`; `SeqCollator` (answer-only labels via chat template, `full_labels`, `alt_*` for DPO/AltPO/FLAT); `make_loader`, `build_unlearn_loaders`. |
| `torchunlearn/unlearn/trainers/seq/` | `base.py` (`SeqUnlearner`), one file per method, `finetune.py` (`SeqFinetune`). |
| `torchunlearn/metrics/seq.py` | `SeqUnlearningEvaluator({role: samples}, collator)` — log-prob/token, seq NLL, Min-k%, greedy ROUGE-L / EM / includes, kept generations. |
| `torchunlearn/metrics/text.py` | ROUGE-L / EM / includes without external packages (ported from mmforge). |
| `torchunlearn/api/algorithms.py` | registry names `MM-*` (`describe("MM-NPO")`), `modality="seq"`; `build_unlearner_split` constructs with hparams. |
| `scripts/run_mm.py` | CLI: `finetune` / `unlearn` / `eval`. |
| `tests/test_seq_smoke.py`, `tests/tiny_models.py` | CPU smoke on a random-init tiny LLaVA + tiny Llama (24 tests). |

## Methods (registry name → class)

MM-Finetune (build the model to unlearn from / retrain / retain-only FT), MM-GradAscent, MM-GradDiff, MM-NPO,
MM-SimNPO, MM-DPO, MM-AltPO, MM-WGA, MM-SatImp, MM-UNDIAL, MM-PDU, MM-FLAT, MM-RMU.

Multimodal-paper methods (see `docs/multimodal_papers.md` for the venue table and fidelity notes):
MM-PO, MM-KLMin (FIUBench / MLLMU baselines), MM-SIU (NeurIPS 2024), MM-ASRUSteer + MM-ASRU (ICML 2026),
MM-SafetyMirage-NPO / -RMU (ICLR 2026), `LUMoE` (ICML 2026, class API), CLIP track: CLIP-SLUG (ICML 2025),
`ADU` (NeurIPS 2025).  Benchmarks: FIUBench, MLLMU-Bench, UMU-Bench, CLEAR, MLUBench, MMUBench, VLGuard recipes in
`torchunlearn/unlearn/recipes/`, evaluators in `torchunlearn/metrics/mm_bench.py`, LLM judge in `metrics/judge.py`.
Loss formulas are byte-identical to `trainers/llm_pii.py` (the Enron/PII sweeps); only the batch contract changed.
Not ported on purpose: CEU / JensUn (excluded by the user), REVS (PII-span specific, Llama-only).

Common hparams: `gamma`, `alpha`, `retain_loss_type` (NLL | KL | JSD | none), `retain_loss_on` (answer | full),
`ref_mode` (cache | model), `trainable`, `grad_ckpt`, `stop_lp` + `stop_role`, `eval_every`, `traj_every`.

## Usage (API)

```python
from torchunlearn import SeqRobModel, SeqUnlearningEvaluator
from torchunlearn.unlearn.seq_data import from_llava_json, split_forget_retain, SeqCollator, build_unlearn_loaders, strip_images
from torchunlearn.unlearn.trainers.seq import NPO

m = SeqRobModel.from_pretrained("llava-hf/llava-1.5-7b-hf", device="cuda:0", trainable="lm")   # or "lora:r=16,target=lm"
data = from_llava_json("train.json", image_root="imgs", group_key="id")
forget, retain = split_forget_retain(data, by="group", ratio=0.1, seed=0)

col = SeqCollator(m, alt_text="[REDACTED]")                      # answer-only labels; alt_* for DPO/FLAT
loaders = build_unlearn_loaders(forget, retain, col, batch_size=4)
ev = SeqUnlearningEvaluator({"Forget": forget, "Retain": retain, "Forget_noimg": strip_images(forget)}, col)

u = NPO(m, beta=0.1, alpha=1.0).set_evaluator(ev)
u.setup(optimizer="AdamW(lr=1e-5)", n_epochs=5)
u.fit(loaders, n_epochs=5)            # prints before/after tables; u.results, u.history
m.save_pretrained("runs/npo/model")
```

Fine-tuning first (base checkpoint → model that knows the data):

```python
from torchunlearn.unlearn.trainers.seq import SeqFinetune
from torchunlearn.unlearn.seq_data import make_loader
SeqFinetune(m, trainable="lm").setup(optimizer="AdamW(lr=2e-5)").fit(make_loader(data, col, batch_size=8), n_epochs=3)
```

## Usage (CLI)

```bash
python scripts/run_mm.py finetune --model Qwen/Qwen2-VL-7B-Instruct --data llava:train.json --image-root imgs \
    --trainable lm --epochs 3 --lr 2e-5 --batch-size 4 --mem-fraction 0.4 --out runs/ft
python scripts/run_mm.py unlearn --model runs/ft/model --data llava:train.json --image-root imgs --split group:0.1 \
    --method MM-NPO --hp beta=0.1 --trainable lm --epochs 5 --lr 1e-5 --out runs/npo --save-model
python scripts/run_mm.py eval --model runs/npo/model --data Forget=jsonl:forget.jsonl Retain=jsonl:retain.jsonl --out runs/eval
```

`results.json` holds hparams, before/after per role, deltas, trajectory, time.  Text-only models use the same
commands (the collator just has no images).

## Design decisions (do not undo)

- **Answer span = longest common token prefix** between the prompt-only and the prompt+answer rendering of the
  chat template, not a length difference.  A template ending in `assistant: ` tokenises its trailing space alone,
  but in the full text that space merges into the first answer token; the length difference silently dropped
  that token in the first implementation (caught by the smoke test).  Image tokens are always on the prompt side,
  so no per-model image-token counting is needed.  Without a chat template, `template="plain"` uses the
  tokenizer's offset mapping (text-only).
- **`full_labels` never include image tokens.**  `retain_loss_on="full"` for an LVLM therefore means "all text
  tokens of the turn", not the image placeholders (training on those is meaningless).
- **Alternates**: `SeqSample.alt_answers` (rotated per epoch, AltPO) win over `SeqCollator(alt_text=...)`.
  Alternate batches carry `alt_input_ids / alt_attention_mask / alt_labels`; image tensors are shared.
- **RMU is structure-agnostic**: `SeqRobModel.decoder_layers()` finds the language-model block list (never the
  vision encoder) and `mlp_out_proj_names` picks the last Linear of each block's MLP (down_proj / fc2 / c_proj).
  LLaVA in transformers 4.57 is `model.language_model.layers`, plain Llama is `model.layers`.
- **`trainable=None` keeps the model's current `requires_grad`** (set at `from_pretrained(trainable=...)`).
  The CLI loads with `--trainable` (default `lm`) and passes the method's `trainable` hparam only when
  `--trainable` was given explicitly, so RMU keeps its 3-matrix default instead of training the whole LM
  (that mistake OOMed the first RMU smoke).  Shared-GPU rule: pass `--mem-fraction`.
- **Gradient checkpointing is non-reentrant** (`use_reentrant=False`): reentrant checkpointing silently returns
  no gradient when the block inputs do not require grad (frozen embeddings + a few trainable matrices deep
  inside, i.e. exactly RMU).
- **Epoch = one pass over Forget** (retain cycled), as in the LLM module.  No gradient accumulation
  (`--batch-size` is the effective batch).
- **Saving is HF format** (`save_pretrained`); never pass `save_path` to `fit` (it would dump `.pth` of a 7B model).
  A LoRA run saves an adapter dir; `from_pretrained` on it loads the base and merges.

## Model support (base env transformers 4.57.6)

Any `AutoModelForImageTextToText` family: LLaVA-1.5/NeXT/OneVision, Qwen2-VL / Qwen2.5-VL, Idefics2/3, SmolVLM,
InternVL (HF port), Gemma-3, PaliGemma, Pixtral/Mistral-3, mllama, Phi-4-MM.  Qwen3.5 / Qwen3-VL need a
transformers 5.x env (see `chaewon-pod-hard-constraints`).  CLIP / SigLIP / BLIP are *not* covered: they have no
token NLL, so the 12 methods do not apply as-is (contrastive variants are a separate track).
