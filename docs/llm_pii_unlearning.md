# LLM / PII scope-aware unlearning in torchunlearn (T/S/C/G)

2026-09-07. Files
- `torchunlearn/unlearn/llm_data.py` — scope-aware forget/retain construction from the 27/28-field facts
  (`build_spec`: scope fact | subject_relation | subject → T, S, C, G, retain), Enron-pilot adapter,
  `SpanDataset` (span-masked labels), `TSCGEvaluator` (log-prob + greedy extraction per role).
- `torchunlearn/unlearn/trainers/llm_pii.py` — `LLMRobModel`, `LLMUnlearner` base, 13 gradient methods.
- `torchunlearn/unlearn/nontrainers/revs.py` — REVS (non-gradient neuron edit).
- `torchunlearn/api/algorithms.py` — registry entries `LLM-*` (`describe("LLM-NPO")`).
- `scripts/run_pii_unlearn.py` (one method), `scripts/run_all_pii.sh` (all methods), `scripts/summarise_pii.py`.
- `prac.ipynb` — section "LLM / PII scope-aware unlearning" (cells 34–55).

## Selected methods (14; three domains each covered)

| domain | method | year / ref | forget loss (ported from) | retain term | ref model |
|---|---|---|---|---|---|
| classic | GradAscent | Jang'23 | −CE on span tokens (open-unlearning) | – | – |
| classic | GradDiff | Liu'22 | −CE + α·retain | NLL (or KL) | KL only |
| classic | NPO | Zhang'24 | −2/β·logσ(β(NLL−NLL_ref)), β=0.1 | NLL | cached |
| classic | SimNPO | Fan'24 | −2/β·logσ(β(NLL/n−δ)), β=4.5, γ=0.125 | NLL | – |
| classic | DPO (IdkDPO) | Rafailov'23 / Maini'24 | win="[REDACTED]" vs lose=span, β=0.1 | NLL | cached |
| classic | RMU | Li'24 | MSE(h_7, 20·u) on span; down_proj 5–7 trainable | EMBED_DIFF | cached |
| SOTA'25 | WGA | Wang'25 | −(p^β·CE), β=1 | NLL | – |
| SOTA'25 | SatImp | Wang'25 | −(p^5(1−p)^1·CE), γ=0.1 | NLL | – |
| SOTA'25 | CE-U | Wang'25 | CE → softmax(logits, true=−∞); skip 1st span token | – | – |
| SOTA'25 | UNDIAL | Dong'25 | CE(student, softmax(teacher − 10·onehot)) | – (α=0) | cached |
| SOTA'25 | PDU | 2025 | (max−mean logit)² on span; optional dual α | NLL | – |
| SOTA'25-09 | JensUn | 2025 | JSD(model, cyclic one-hot "No way") | JSD vs ref | cached |
| SOTA ICLR'25 | FLAT | Wang'25 | f-div (TV) contrastive p([REDACTED]) vs p(span) | – | – |
| PII-specific | REVS | Ashuach ACL'25 | rank editing of down_proj neurons, no gradient | – | – |

Excluded: PoRT (ICLR'26) — released code is inference-time only (IPC + post-judgment), no weight change,
same reason as ECO / in-context / LoRA. Not included: ELM, SPUL (weight-space but no faithful public loss
found in the candidate list).

## Design decisions (from the candidate doc)
- Span-level: labels = −100 outside the PII value span; every gradient method uses the same batches, so
  scope changes only the forget set. Window = 400 chars before / 120 after the mention, ≤512 tokens.
- Scope: `build_spec(facts, scope, targets)`; S = same subject other facts, C = other subjects in T docs,
  G = unrelated subject and document (n_g=50). Retain pool = the G-like pool minus the G probes (no overlap),
  `retain_kind=text` alternative uses random text windows from documents containing no T/S/C/G item.
- DPO / FLAT alternate answer: fixed token string `--alt "[REDACTED]"` (base models, raw-text FT).
- Reference model: `ref_mode=cache` computes reference quantities once at fit start with the untouched model
  (exactly equal to a frozen copy) — no extra 16 GB per 8B model. `ref_mode=model` deep-copies (needed only
  for KL/JSD retain on `retain_loss_on=window`).
- Epoch = one pass over the forget loader (retain batches cycled), as in open-unlearning.
- Matched forgetting (pilot rule): `--stop-lp -5 --eval-every 10` stops when mean T log-prob/token ≤ −5.
- REVS needs the Llama residual structure (pre-norm, `mlp.down_proj`, `model.norm`, `lm_head`):
  Llama-3.1 and the pilot Qwen2 work; OLMo-2 (post-norm) and Qwen3.5 (hybrid) are rejected with an error.

## Commands
```bash
cd "/home1/irteam/_[chaewon]/_[torchunlearn]"
scripts/run_all_pii.sh pilot                                          # 14 methods on the Enron pilot (GPU 1)
scripts/run_all_pii.sh llama31_8b --scope subject --targets "N:allen-p/_sent_mail/586.#jim murnan"
scripts/run_all_pii.sh olmo2_7b   --scope fact    --targets Fd142db50f309 F29a0a7c75db2
scripts/run_all_pii.sh llama31_8b --scope subject_relation --targets "N:allen-p/_sent_mail/586.#jim murnan:EMAIL"
scripts/run_all_pii.sh qwen35_9b  --scope subject --targets "N:allen-p/_sent_mail/586.#jim murnan"      # overlay env auto-set
METHODS="NPO RMU REVS" GPU=0 EPOCHS=10 LR=2e-5 STOP_LP=-5 scripts/run_all_pii.sh pilot
python scripts/run_pii_unlearn.py --pilot --method NPO --epochs 5 --lr 1e-5 --hp beta=0.1 alpha=1.0
python scripts/summarise_pii.py out/pii_unlearn/pilot
```
Memory: 8B full-parameter AdamW ≈ 96 GB (+activations). Use `--hp trainable='model\.layers\.(2[0-9]|3[01])\..*'`,
`--hp grad_ckpt=true`, or `--optimizer "SGD(lr=1e-3)"` when the GPU is shared.

## Smoke test (pilot, 1 epoch, Qwen2-1.5B M_T, GPU1, 2026-09-07)
See `out/pii_unlearn/_smoke/smoke_all.log` and `out/pii_unlearn/_smoke/*/results.json`.

All 14 methods ran end to end (1 epoch = 3 iterations of bs 4 on 10 T spans; lr 1e-5; `--no-extract` except GradAscent).
GradAscent/GradDiff/NPO/SimNPO/DPO ran before the epoch definition was changed to 'one pass over forget' (they did 20 iterations).
REVS used the Llama-8B rank margins on the 1.5B/152k-vocab pilot model, hence the S/C/G damage — margins must be tuned per model.

| method | ΔT lp | ΔS lp | ΔC lp | ΔG lp | T extr 0→1 | S extr 0→1 | C extr 0→1 | G extr 0→1 | iters | time(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| CEU_e1 | -1.279 | -0.009 | -0.010 | -0.019 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | - | 11.4 |
| DPO_e1 | -1.823 | -0.089 | -0.032 | +0.043 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | - | 57.3 |
| FLAT_e1 | -0.531 | -0.012 | -0.005 | -0.009 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | - | 11.6 |
| GradAscent_e1 | -0.658 | -0.005 | -0.015 | +0.003 | 0.90→0.30 | 0.72→0.72 | 0.68→0.67 | 0.83→1.00 | - | 230.4 |
| GradDiff_e1 | -7.039 | -0.155 | -0.057 | -0.023 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | - | 44.1 |
| JensUn_e1 | +0.006 | -0.003 | -0.007 | -0.007 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | - | 14.7 |
| NPO_e1 | -3.622 | -0.078 | -0.025 | +0.008 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | - | 42.1 |
| PDU_e1 | -0.793 | -0.004 | -0.017 | -0.021 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | - | 13.1 |
| REVS_e1 | -6.894 | -2.173 | -0.714 | -0.545 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | - | 78.2 |
| RMU_e1 | +0.005 | -0.005 | -0.001 | +0.004 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | - | 13.6 |
| SatImp_e1 | -0.012 | -0.030 | -0.001 | +0.012 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | - | 15.1 |
| SimNPO_e1 | -0.363 | -0.162 | -0.066 | +0.126 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | - | 40.7 |
| UNDIAL_e1 | -1.278 | +0.002 | -0.007 | +0.001 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | - | 12.5 |
| WGA_e1 | -0.457 | -0.021 | -0.002 | +0.017 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | - | 15.2 |

