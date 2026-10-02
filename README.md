<div align="center">

# 🧠 Machine-Unlearning-PyTorch

**A PyTorch library for machine unlearning across vision classifiers, large language models, and multimodal (vision-language) models — make your models forget, on demand.**

<a href="https://github.com/Harry24k/machine-unlearning-pytorch/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-brightgreen?style=flat-square" /></a>
<a href="https://pypi.org/project/torchunlearn/"><img alt="PyPI" src="https://img.shields.io/pypi/v/torchunlearn.svg?color=orange&style=flat-square" /></a>
<img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.8-blue?style=flat-square" />
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E%3D1.7.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
<img alt="Domains" src="https://img.shields.io/badge/domains-vision%20%7C%20LLM%20%7C%20multimodal-blueviolet?style=flat-square" />
<a href="https://colab.research.google.com/github/Harry24k/machine-unlearning-pytorch/blob/main/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" /></a>

[Blog Post](https://trustworthyai.co.kr/article/2025/uam-eng/) · [NeurIPS 2025 Paper](https://neurips.cc/virtual/2025/poster/116406) · [Demo Notebook](demo.ipynb) · [LLM / Multimodal Guide](docs/multimodal_unlearning.md) · [VLM Paper Table](docs/multimodal_papers.md)

</div>

<br>

Machine unlearning removes the influence of specific training data from a trained model, as if that data was never used.

| Use case | What gets removed |
|:--|:--|
| 🔒 **Privacy** | GDPR "right to be forgotten", PII in LLMs and VLMs |
| 🛠 **Data correction** | Mislabeled or corrupted samples |
| ⚖️ **Bias mitigation** | Biased training data |
| 🛡 **Security & safety** | Backdoors, poisoned examples, unsafe VLM behaviour |

## Contents

**Start** — [Installation](#installation) · [Domains at a glance](#domains-at-a-glance) · [Project layout](#project-layout)<br>
**Vision** — [Quick start](#quick-start) · [Forgetting scenarios](#forgetting-scenarios) · [Methods](#vision-methods) · [Evaluation](#evaluation) · [Benchmark results](#benchmark-results)<br>
**LLM** — [QA-level forgetting](#qa--instruction-level-forgetting) · [Span-level (PII) forgetting](#span-level-pii-forgetting)<br>
**Multimodal** — [Vision-language models](#vision-language-models-lvlm) · [Methods from the literature](#methods-from-the-multimodal-literature) · [CLIP](#clip--dual-encoders)<br>
**More** — [Registry and CLI](#registry-and-cli) · [Tests](#tests) · [Related projects](#related-projects) · [Citation](#citation) · [Contributing](#contributing) · [License](#license)

<br>

## Installation

| Track | Requirements |
|:--|:--|
| Vision | Python ≥ 3.8, PyTorch ≥ 1.7.1 |
| LLM / Multimodal | Python ≥ 3.9, PyTorch ≥ 2.0, `transformers` 4.5x (Qwen3.5 / Qwen3-VL need 5.x) |

```bash
pip install torchunlearn            # vision only
pip install "torchunlearn[llm]"     # + transformers, peft, datasets, Pillow, accelerate
```

Latest development version:

```bash
pip install "git+https://github.com/Harry24k/machine-unlearning-pytorch.git#egg=torchunlearn[llm]"
```

> [!NOTE]
> Models and datasets are **not** distributed with the library. Point the wrappers at any local checkpoint or Hugging Face id you have access to.

<br>

## Domains at a glance

Every domain follows the same workflow:

**Wrap the model → Build Forget / Retain loaders → Pick a method → `setup()` → `fit()` → Evaluate**

| | 🖼 Vision | 📝 LLM | 🖼📝 Multimodal |
|:--|:--|:--|:--|
| **Unlearn from** | Image classifiers (ResNet, ViT, …) | Any HF causal LM — base or fine-tuned | VLMs (LLaVA, Qwen-VL, Idefics, Gemma-3, …) and CLIP-style dual encoders |
| **Granularity** | Random subset · Whole class | QA pair · Text span (PII) | Image + text QA pair |
| **Wrapper** | `RobModel` | `SeqRobModel` · `LLMRobModel` | `SeqRobModel` (LVLM) · `CLIPRobModel` (CLIP) |
| **Evaluate** | Retain / forget / test acc, MIA, ZRF, adversarial robustness | Answer log-prob, ROUGE-L / EM, truth ratio + KS, Min-k% MIA, T/S/C/G probes | FIUBench, MLLMU-Bench, UMU-Bench, CLEAR, MLUBench, MMUBench, VLGuard, text-only leakage probe |
| **Go to** | [Vision →](#vision) | [LLM →](#llm) | [Multimodal →](#multimodal) |

> [!TIP]
> LLM and multimodal share **one code path**: a method only sees the answer-token `labels` and forwards whatever else the processor produced (`pixel_values`, `image_grid_thw`, …). A method written once runs on a text LLM and on an image+text model alike.

<br>

## Project layout

| Path | What's inside |
|:--|:--|
| 📦 **`torchunlearn/nn/`** | `RobModel` (vision) · `SeqRobModel` (LLM + LVLM) · `CLIPRobModel` (dual encoder) |
| 📦 **`torchunlearn/unlearn/`** | All unlearning methods and data utilities ↓ |
| &emsp;└ `trainers/` | Vision trainers (Finetune, NegGrad, SalUn, SCRUB, ARU, AMUN, SFRon, RFE, MUMis, FaLW, …) · `llm_pii.py` (`LLM-*`) |
| &emsp;&emsp;└ `seq/` | Sequence trainers shared by LLM and LVLM (`MM-*`): 12 token-level methods, PO, KLMin, SIU, GRPO + ASRU, LUMoE, SafetyMirage, Finetune |
| &emsp;└ `nontrainers/` | FisherForget, Influence, NegMerge, SISA, REM, Amnesiac, REVS (LLM), SLUG (CLIP) |
| &emsp;└ `clip/` | ADU — domain unlearning for CLIP |
| &emsp;└ `recipes/` | Benchmark loaders: FIUBench, MLLMU-Bench (+UMU), CLEAR, MLUBench, MMUBench, domains, VLGuard |
| &emsp;└ `seq_data.py` | `SeqSample`, `SeqCollator` (answer-only labels), splits, loaders |
| &emsp;└ `llm_data.py` | Span-level PII data (`SpanDataset`, `build_spec`, `TSCGEvaluator`) |
| 📦 **`torchunlearn/metrics/`** | `UnlearningEvaluator` (vision) · `SeqUnlearningEvaluator` · `mm_bench` · `judge` (LLM-as-judge) · `text` |
| 📦 **`torchunlearn/benchmarks/`** | `BenchmarkSuite` (vision) |
| 📦 **`torchunlearn/api/`** | Registry: `register_unlearner` / `describe` / `build_unlearner(_split)`, HParam specs |
| 📦 **`torchunlearn/attacks/` `optim/` `utils/`** | Adversarial attacks, UAM minimizer, datasets and vision models |
| ▶️ **`scripts/`** | `run_mm.py` (LLM + LVLM) · `run_clip.py` (SLUG, ADU) · `run_pii_unlearn.py` (span-level) |
| 📄 **`docs/`** | `multimodal_unlearning.md` · `multimodal_papers.md` · `llm_pii_unlearning.md` |
| ✅ **`tests/`** | CPU smoke tests on tiny random-init models (LLaVA, Llama, CLIP) |

<br>

## Vision

### Quick start

```python
import torchunlearn
from torchunlearn.unlearn import Finetune
from torchunlearn.utils.data import UnlearnDataSetup, MergedLoaders

# 1. Wrap your model
model = torchunlearn.utils.load_model(model_name="ResNet18", n_classes=10)
rmodel = torchunlearn.RobModel(
    model, n_classes=10,
    normalization_used={"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
)

# 2. Load the checkpoint you want to unlearn from
rmodel.load_dict("./models/CIFAR10_Standard/last.pth")

# 3. Retain / Forget / Test loaders
setup = UnlearnDataSetup(data_name="CIFAR10", n_classes=10,
                         mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
train_loaders, test_loaders = setup.get_loaders_for_rand(batch_size=128, ratio=0.1, stratified=True)
merged_loader = MergedLoaders(train_loaders)        # {"Retain": ..., "Forget": ...} consumed together

# 4. Unlearn
trainer = Finetune(rmodel)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5, save_path="./models/unlearned")
```

> [!NOTE]
> [`demo.ipynb`](demo.ipynb) expects a pretrained checkpoint at `./models/CIFAR10_Standard/last.pth` (not distributed) and a GPU runtime.

### Forgetting scenarios

**Random forgetting** — forget a randomly sampled subset of training data (e.g., 10%):

```python
train_loaders, test_loaders = setup.get_loaders_for_rand(
    batch_size=128,
    ratio=0.1,        # fraction to forget
    stratified=True,  # keep class distribution
    seed=42,
)
```

**Classwise forgetting** — forget all samples belonging to a specific class:

```python
train_loaders, test_loaders = setup.get_loaders_for_classwise(
    batch_size=128,
    omit_label=1,     # class index to forget
    train_shuffle_and_transform=True,
)
```

### Vision methods

#### Training-based

| Method | Idea | Paper |
|:--|:--|:--|
| **Finetune** | Fine-tune on the retain set only | [Warnecke et al., NDSS 2023](https://arxiv.org/abs/2108.11577) |
| **NegGrad** | Negative gradient on the forget set | [Golatkar et al., CVPR 2020](https://arxiv.org/abs/1911.04933) |
| **RandomLabel** | Relabel the forget set with random labels | [Golatkar et al., CVPR 2020](https://arxiv.org/abs/1911.04933) |
| **L1Sparse** | L1 sparsity regularization during fine-tuning | [Jia et al., NeurIPS 2023](https://arxiv.org/abs/2304.04934) |
| **SCRUB** | Alternating KL-max / KL-min distillation | [Kurmanji et al., NeurIPS 2023](https://arxiv.org/abs/2302.09880) |
| **BadTeacher** | Competent / bad-teacher distillation | [Chundawat et al., AAAI 2023](https://arxiv.org/abs/2205.08096) |
| **BoundaryShrink** | Nearest-class re-targeting to shrink the forget-class boundary | [Chen et al., CVPR 2023](https://arxiv.org/abs/2303.11570) |
| **SalUn** | Saliency-masked random-label fine-tuning | [Fan et al., ICLR 2024](https://arxiv.org/abs/2310.12508) |
| **SFRon** | Saliency forgetting in a remain-preserving manifold | [Huang et al., NeurIPS 2024](https://arxiv.org/abs/2409.19732) |
| **AMUN** | Fine-tune on the nearest adversarial example of each forget sample | [Ebrahimpour-Boroojeny et al., ICML 2025](https://icml.cc/virtual/2025/poster/46097) |
| **UAM** ⭐ | Unlearning-Aware Minimization *(a minimizer — wraps any trainer)* | [Kim et al., NeurIPS 2025](https://neurips.cc/virtual/2025/poster/116406) |
| **ARU** | Adversarial Retain-free Unlearning | [Yoon et al., 2026](https://ieeexplore.ieee.org/document/11414433) |
| **RFE** | Two-phase augmented Lagrangian + W2-regularized gradient projection | [Cheng et al., ICLR 2026](https://arxiv.org/abs/2603.26569) |
| **MUMis** | Suppress input sensitivity on the forget set *(retain-free)* | [Cheng et al., ICLR 2026](https://arxiv.org/abs/2402.15109) |
| **FaLW** | Instance-wise loss reweighting for long-tailed forget sets | [Yu et al., 2026](https://arxiv.org/abs/2601.18650) |

#### Non-training

| Method | Idea | Paper |
|:--|:--|:--|
| **FisherForget** | Fisher-information weight perturbation | [Golatkar et al., CVPR 2020](https://arxiv.org/abs/1911.04933) |
| **Influence** | Newton-step influence-function removal | [Izzo et al., AISTATS 2021](https://arxiv.org/abs/2002.10077) |
| **SISA** | Sharded, isolated, sliced, aggregated retraining | [Bourtoule et al., S&P 2021](https://arxiv.org/abs/1912.03817) |
| **Amnesiac** | Revert the updates of specific training batches | [Graves et al., AAAI 2021](https://arxiv.org/abs/2010.10981) |
| **NegMerge** | Sign-consensual weight merging | [Kim, Han & Choe, ICML 2025](https://arxiv.org/abs/2410.05583) |
| **REM** | Redirection for erasing memory | — |

> [!IMPORTANT]
> **UAM** is a minimizer, not a trainer — plug it in through `Trainer.setup(minimizer=...)`.<br>
> **MUMis** never reads the Retain split — pass the Forget loader directly to `fit`.

<details>
<summary><b>Usage examples for each method</b></summary>

<br>

**NegGrad**

```python
opt = "SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)"
NegGrad(rmodel, retain_lambda=0.5).setup(optimizer=opt, n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)
```

**UAM** (via the `Standard` trainer)

```python
Standard(rmodel).setup(optimizer=opt, minimizer=f"UAM(rho={rho}, cosine_total_step={cosine_total_step}, gamma={gamma})",
                       n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)
```

**ARU · AMUN · SFRon**

```python
ARU(rmodel, margin=1.0, eps=0.05, steps=50, omit_label=1).setup(optimizer=opt, n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)
AMUN(rmodel, attack="deepfool", steps=20).setup(optimizer=opt, n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)
SFRon(rmodel, saliency_ratio=0.5, slow_alpha=0.5, slow_every=5).setup(optimizer=opt, n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)
```

**FaLW** — needs a held-out validation loader; set `estimate_every > 1` to amortize the per-step validation pass

```python
trainer = FaLW(rmodel, tau=0.15)
trainer.prepare(train_loaders["Forget"], val_loader)
trainer.setup(optimizer=opt, n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)
```

**MUMis** — retain-free; unbounded objective, so use a small lr + early stopping and watch `Clean(R)`

```python
MUMis(rmodel, other_lambda=1.0, n_other=1).setup(optimizer="SGD(lr=1e-4)", n_epochs=1).fit(train_loaders=train_loaders["Forget"], n_epochs=1)
```

**RFE** — `"Adjacent"` = retain samples entangled with the forget set (e.g. sibling CIFAR-100 subclasses)

```python
merged_loader = MergedLoaders({"Retain": train_loaders["Retain"], "Forget": train_loaders["Forget"], "Adjacent": adjacent_loader})
RFE(rmodel, phase1_steps=100, constraint_tol=0.05, w2_lambda=1.0).setup(optimizer=opt, n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)
```

**FisherForget · NegMerge** (non-training)

```python
FisherForget(rmodel).fit(train_loaders, alphas=[1e-9, 1e-8, 1e-7, 1e-6], repeat=3, save_path="./models/fisher")
NegMerge(rmodel).fit(train_loaders, lrs=[1e-4, 5e-4, 1e-3], epochs=1, repeats=3, scaling=1.0, consensus_ratio=1.0,
                     aggregation="mean", save_path="./models/negmerge")
```

</details>

### Evaluation

**During unlearning** — register the loaders you want tracked, then train as usual:

```python
trainer.record_rob({
    "(R)":  train_loaders["Retain"],
    "(F)":  train_loaders["Forget"],
    "(Te)": test_loaders["Test"],
}, n_limit=1000)

trainer.fit(train_loaders=merged_loader, n_epochs=5,
            save_path="./models/unlearned",
            save_best={"Clean(R)": "HB", "Clean(F)": "LBO"},
            record_type="Epoch")
```

<details>
<summary>Sample training log (Finetune, CIFAR-10, 10% random forgetting)</summary>

```text
[Finetune]
Training Information.
-Epochs: 5
-Optimizer: SGD (lr: 0.01, momentum: 0.9, weight_decay: 0.0005)
-Record Type: Epoch
-Device: cuda:0
---------------------------------------------------------------------
Epoch   Cost     Clean(R)   Clean(F)   Clean(Te)   lr       s/it
=====================================================================
1       0.0913   96.4844    91.0156    91.7969     0.0100   0.0572
2       0.0524   96.3867    68.6523    91.0156     0.0100   0.0541
3       0.0884   95.8008    54.3945    91.0156     0.0100   0.0521
4       0.0525   96.5820    45.0195    90.0391     0.0100   0.0519
5       0.1073   97.3633    33.0078    91.4062     0.0100   0.0522
---------------------------------------------------------------------
```

</details>

**After unlearning**

```python
from torchunlearn.metrics import UnlearningEvaluator

evaluator = UnlearningEvaluator(
    unlearned_model=rmodel,
    train_loaders=train_loaders,
    test_loaders=test_loaders,
    retrained_model=retrained_rmodel,  # optional; required for the UG metric
    n_limit=1000,
)
evaluator.print_report()
```

**Comparing several methods**

```python
from torchunlearn.benchmarks import BenchmarkSuite

suite = BenchmarkSuite(train_loaders, test_loaders, retrained_model=retrained_rmodel)
suite.add("Finetune", ft_model)
suite.add("NegGrad", ng_model)
suite.print_table()
```

### Benchmark results

CIFAR-10 · ResNet-18 · 5 epochs · SGD (lr 0.01, momentum 0.9, wd 5e-4) · averaged over 3 seeds

| Metric | Meaning | Goal |
|:--|:--|:--|
| **RA** | Retain accuracy | ↑ |
| **FA** | Forget accuracy | Match Retrain |
| **TA** | Test accuracy | ↑ |
| **ΔAcc** | \|ΔRA\| + \|ΔFA\| + \|ΔTA\| vs. Retrain | ↓ |

<details>
<summary><b>🎲 Random forgetting — 10% of training data</b></summary>

| Algorithm | RA | FA | TA | Time (s) | **ΔAcc** |
|:---|---:|---:|---:|---:|---:|
| *Retrain (oracle)* | *TODO* | *TODO* | *TODO* | *TODO* | *0.00* |
| Finetune | 100.00 | 99.84 | 94.26 | 32.2 | 94.90 |
| NegGrad | 10.00 | 10.00 | 10.00 | 48.6 | 169.20 |
| RandomLabel | 49.71 | 51.24 | 46.82 | 44.1 | 133.31 |
| L1Sparse | 100.00 | 99.92 | 94.28 | 43.8 | 95.00 |
| SCRUB | 31.69 | 32.02 | 31.86 | 35.1 | 147.07 |
| BadTeacher | 99.70 | 99.74 | 93.11 | 42.7 | 93.35 |
| BoundaryShrink | 87.71 | 82.34 | 80.77 | 46.1 | 92.46 |
| SalUn | 99.64 | 99.72 | 92.95 | 63.6 | 93.41 |
| ARU | 90.35 | 90.12 | 84.75 | 42.9 | 93.62 |
| FisherForget | 9.99 | 10.00 | 10.01 | 65.5 | 169.20 |
| Influence | 99.98 | 99.94 | 94.48 | 49.2 | 95.20 |
| NegMerge | 99.46 | 99.24 | 93.00 | 29.4 | 92.70 |
| Standard | 99.99 | 99.56 | 93.75 | 46.4 | 94.10 |
| **UAM** | **100.00** | **85.72** | **85.32** | *TODO* | **87.40** |

</details>

<details>
<summary><b>🏷️ Classwise forgetting — one class</b></summary>

| Algorithm | RA | FA | TA | Time (s) | **ΔAcc** |
|:---|---:|---:|---:|---:|---:|
| *Retrain (oracle)* | *TODO* | *TODO* | *TODO* | *TODO* | *0.00* |
| Finetune | 100.00 | 95.17 | 94.24 | 36.3 | 99.11 |
| NegGrad | 11.11 | 0.00 | 11.11 | 40.5 | 168.08 |
| RandomLabel | 83.32 | 5.73 | 77.48 | 45.3 | 35.23 |
| L1Sparse | 100.00 | 99.80 | 94.06 | 44.2 | 103.56 |
| SCRUB | 14.31 | 0.00 | 14.27 | 34.0 | 161.72 |
| BadTeacher | 99.41 | 15.54 | 92.70 | 44.1 | 17.95 |
| BoundaryShrink | 98.41 | 0.00 | 91.52 | 40.0 | 2.59 |
| SalUn | 99.60 | 0.00 | 92.66 | 88.0 | 2.64 |
| ARU | 93.18 | 39.66 | 86.72 | 49.9 | 50.06 |
| FisherForget | 97.34 | 1.24 | 90.62 | 67.4 | 3.66 |
| Influence | 99.98 | 99.98 | 94.19 | 46.2 | 103.85 |
| NegMerge | 96.31 | 0.08 | 89.58 | 29.0 | 4.49 |
| Standard | 99.99 | 81.13 | 93.87 | 47.2 | 84.69 |
| **UAM** | **100.00** | **0.00** | **90.84** | *TODO* | **4.86** |

</details>

<br>

## LLM

Works on **any Hugging Face causal LM you load yourself** — a base model, your fine-tuned checkpoint, or a PEFT adapter directory.

| | QA / instruction-level | Span-level (PII) |
|:--|:--|:--|
| **Forget unit** | A (prompt, answer) pair | A text span inside a document (a name, an address, …) |
| **Loss on** | Answer tokens only | Span tokens only |
| **Wrapper** | `SeqRobModel` | `LLMRobModel` |
| **Methods** | `MM-*` | `LLM-*` |
| **Evaluation** | Log-prob, NLL, Min-k% MIA, ROUGE-L / EM, truth ratio + KS | Scope-aware Target / Same-subject / Co-document / Global (T/S/C/G) probes |

### QA / instruction-level forgetting

The answer span is found by rendering the chat template twice and keeping what remains after the longest common token prefix — no tokenizer-specific boundary bugs. Build the model to unlearn from with `SeqFinetune` (stage I), then unlearn (stage II).

```python
from torchunlearn import SeqRobModel, SeqUnlearningEvaluator
from torchunlearn.unlearn.seq_data import from_jsonl, split_forget_retain, SeqCollator, make_loader, build_unlearn_loaders
from torchunlearn.unlearn.trainers.seq import SeqFinetune, NPO

model = SeqRobModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device="cuda:0", trainable="lora:r=16,target=lm")
data = from_jsonl("qa.jsonl", fields={"prompt": "question", "answer": "answer", "group": "subject"})   # text-only: no images
col = SeqCollator(model, alt_text="I don't know.")                 # alt_* = refusal / alternate for DPO, PO, FLAT

# Stage I — fine-tune
SeqFinetune(model).setup(optimizer="AdamW(lr=2e-5)").fit(make_loader(data, col, batch_size=8), n_epochs=3)

# Stage II — unlearn
forget, retain = split_forget_retain(data, by="group", ratio=0.1, seed=0)
ev = SeqUnlearningEvaluator({"Forget": forget, "Retain": retain}, col)
NPO(model, beta=0.1, alpha=1.0).set_evaluator(ev).setup(optimizer="AdamW(lr=1e-5)", n_epochs=5) \
   .fit(build_unlearn_loaders(forget, retain, col, batch_size=4), n_epochs=5)
model.save_pretrained("runs/npo/model")
```

The `MM-` prefix means "sequence model" — text-only or image+text.

| Methods | `MM-GradAscent` · `MM-GradDiff` · `MM-NPO` · `MM-SimNPO` · `MM-DPO` (IDK-DPO) · `MM-AltPO` · `MM-RMU` · `MM-WGA` · `MM-SatImp` · `MM-UNDIAL` · `MM-PDU` · `MM-FLAT` · `MM-PO` · `MM-KLMin` · `MM-Finetune` |
|:--|:--|

| Evaluation | Where |
|:--|:--|
| Log-prob / token, sequence NLL, Min-k% MIA, ROUGE-L / EM / includes on greedy generations | `SeqUnlearningEvaluator` |
| TOFU-style truth ratio and KS forget quality, multiple-choice accuracy | `torchunlearn.metrics.mm_bench` |
| LLM-judge rubrics | `torchunlearn.metrics.judge` |

### Span-level (PII) forgetting

14 methods (`LLM-GradAscent` … `LLM-FLAT`, `LLM-CEU`, `LLM-JensUn`, and non-gradient `LLM-REVS`) share the same loss formulas as the `MM-*` family. Full walkthrough: [docs/llm_pii_unlearning.md](docs/llm_pii_unlearning.md) and `scripts/run_pii_unlearn.py`.

```python
from torchunlearn.unlearn.llm_data import load_docs, load_facts, build_spec, SpanDataset, make_collate, TSCGEvaluator
from torchunlearn.unlearn.trainers.llm_pii import LLMRobModel, LLM_TRAINERS
```

<br>

## Multimodal

### Vision-language models (LVLM)

Same sequence trainers as the LLM track, now with images. `SeqRobModel.from_pretrained` detects the modality from the config and loads `AutoModelForImageTextToText`.

| | |
|:--|:--|
| **Supported families** | LLaVA-1.5 / NeXT / OneVision · Qwen2-VL / Qwen2.5-VL · Idefics2/3 · SmolVLM · InternVL · Gemma-3 · PaliGemma · Pixtral · Phi-4-MM · … |
| **`trainable=`** | `lm` · `vision` · `projector` · `lm+projector` · `lora:...` · a regex |

```python
from torchunlearn import SeqRobModel, SeqUnlearningEvaluator
from torchunlearn.unlearn.seq_data import from_llava_json, split_forget_retain, SeqCollator, build_unlearn_loaders, strip_images
from torchunlearn.unlearn.trainers.seq import GradDiff

model = SeqRobModel.from_pretrained("llava-hf/llava-1.5-7b-hf", device="cuda:0", trainable="lm")
data = from_llava_json("train.json", image_root="imgs", group_key="id")       # LLaVA "conversations" JSON
forget, retain = split_forget_retain(data, by="group", ratio=0.1)
col = SeqCollator(model)                                                      # answer-only labels, image tokens never targeted
ev = SeqUnlearningEvaluator({"Forget": forget, "Retain": retain,
                             "Forget_noimg": strip_images(forget)}, col)      # cross-modal leakage probe (text only)
GradDiff(model, grad_ckpt=True).set_evaluator(ev).setup(optimizer="AdamW(lr=1e-5)", n_epochs=5) \
        .fit(build_unlearn_loaders(forget, retain, col, batch_size=2), n_epochs=5)
```

### Methods from the multimodal literature

| Method | Venue | Idea | Use via |
|:--|:--|:--|:--|
| **SIU** | NeurIPS 2024 | Single Image Unlearning — multifaceted FT data + dual-masked KL | `MM-SIU` |
| **FIUBench** | ICLR 2025 | Fictitious facial identities; GA / GD / KL / PO baselines | `recipes.fiubench` |
| **SLUG** | ICML 2025 | Single-layer single-gradient unlearning for CLIP, transplantable into LLaVA | `CLIP-SLUG` |
| **ADU** | NeurIPS 2025 | Approximate domain unlearning for CLIP | `unlearn.clip.ADU` |
| **UMU-Bench** | NeurIPS 2025 | Unimodal vs. multimodal knowledge, 653 profiles | `recipes.mllmu_bench` |
| **Safety Mirage** | ICLR 2026 | VLM safety via NPO / RMU unlearning on VLGuard | `MM-SafetyMirage-NPO` · `-RMU` |
| **ASRU** | ICML 2026 | Closed-form activation steering + GRPO with a rule-based refusal reward | `MM-ASRUSteer` · `MM-ASRU` |
| **LUMoE** | ICML 2026 | Lifelong unlearning with switchable LoRA experts and an entity router (MLUBench) | `trainers.seq.LUMoE` |

<details>
<summary>Full entry points (recipes, evaluators, source files)</summary>

<br>

| Method | Entry points |
|:--|:--|
| SIU | `MM-SIU`, `recipes.mmubench.build_siu_samples` |
| FIUBench | `recipes.fiubench`, `mm_bench.FIUBenchEvaluator`, `MM-KLMin`, `MM-PO` |
| SLUG | `CLIP-SLUG` (`nontrainers/slug.py`, `apply_to_vlm`) |
| ADU | `unlearn.clip.ADU`, `recipes.domains` (deep vision prompts, InstaPG, domain-disentangling loss) |
| UMU-Bench | `recipes.mllmu_bench`, `mm_bench.MLLMUBenchEvaluator` |
| Safety Mirage | `MM-SafetyMirage-NPO`, `MM-SafetyMirage-RMU`, `build_vlguard_sets`, `SafetyMirageEvaluator` |
| ASRU | `MM-ASRUSteer`, `MM-ASRU` (`trainers/seq/grpo.py`) |
| MLUBench / LUMoE | `trainers.seq.LUMoE`, `recipes.mlubench`, `MLUBenchEvaluator` |

</details>

**Benchmarks with ready recipes and evaluators:** FIUBench · MLLMU-Bench · UMU-Bench · CLEAR · MLUBench · MMUBench · VLGuard

Fidelity notes and the methods deliberately left out are in [docs/multimodal_papers.md](docs/multimodal_papers.md).

### CLIP / dual encoders

```python
from torchunlearn import CLIPRobModel, SeqRobModel
from torchunlearn.unlearn.nontrainers.slug import SLUG, make_clip_loader

model = CLIPRobModel.from_pretrained("openai/clip-vit-large-patch14-336")
slug = SLUG(model)
slug.compute_gradients(make_clip_loader(forget_pairs, model), make_clip_loader(retain_pairs, model))
slug.fit(eval_fn=lambda model: (forget_acc(model), test_acc(model)))          # binary search on the step size
slug.apply_to_vlm(SeqRobModel.from_pretrained("llava-hf/llava-1.5-7b-hf"))   # same vision tower -> LLaVA forgets too
```

<br>

## Registry and CLI

Every method is registered with its hyper-parameter spec and modality (`vision` · `seq` · `clip`):

```python
from torchunlearn.api import list_algorithms, describe, build_unlearner_split

list_algorithms(modality="seq")      # ['MM-Finetune', 'MM-GradAscent', ..., 'MM-ASRU', ..., 'LLM-NPO', ...]
describe("MM-NPO")                   # setup kwargs (optimizer, n_epochs, ...) and hparams with defaults
u, setup_kw = build_unlearner_split("MM-NPO", rmodel, hparams={"beta": 0.1, "trainable": "lm"})
```

**LLM / LVLM** — same commands; text-only models simply have no images:

```bash
python scripts/run_mm.py finetune --model Qwen/Qwen2-VL-7B-Instruct --data llava:train.json --image-root imgs \
    --trainable lm --epochs 3 --out runs/ft

python scripts/run_mm.py unlearn --model runs/ft/model --data llava:train.json --image-root imgs --split group:0.1 \
    --method MM-NPO --hp beta=0.1 --trainable lm --grad-ckpt --epochs 5 --lr 1e-5 --out runs/npo --save-model

python scripts/run_mm.py eval --model runs/npo/model \
    --data Forget=jsonl:forget.jsonl Retain=jsonl:retain.jsonl --out runs/eval
```

**CLIP**

```bash
python scripts/run_clip.py slug --model openai/clip-vit-large-patch14-336 --forget f.jsonl --retain r.jsonl \
    --eval-forget ef.jsonl --eval-test et.jsonl --classes classes.txt --out runs/slug

python scripts/run_clip.py adu --model openai/clip-vit-base-patch16 --root /data/office_home \
    --forget-domains Clipart --out runs/adu
```

| Data spec | Meaning |
|:--|:--|
| `jsonl:<path>` | Records; map fields with `--fields prompt=question,answer=answer,images=image,group=subject` |
| `llava:<path.json>` | LLaVA "conversations" JSON (VLGuard / MLLMU exports) |
| `hf:<name>[:<split>]` | Hugging Face datasets |

> [!TIP]
> On shared GPUs, always pass `--mem-fraction`.

<br>

## Tests

```bash
python -m pytest tests -q
```

CPU-only, on tiny random-init models (LLaVA, Llama, CLIP built from configs; only the `sshleifer/tiny-gpt2` tokenizer is downloaded). Covers:

| Area | Checks |
|:--|:--|
| Data | Answer-mask invariants |
| Methods | Every registered sequence method moves only its declared parameters |
| Lifecycle | Fine-tune → unlearn → save → reload round-trips, LoRA adapters |
| Paper methods | SIU, ASRU, LUMoE, SLUG, ADU |
| Benchmarks | Recipe parsers and every evaluator's metric keys |

<br>

## Related projects

| Project | Description |
|:--|:--|
| [**MAIR**](https://github.com/Harry24k/MAIR) | Adversarial training framework (NeurIPS'23) |
| [**Torchattacks**](https://github.com/Harry24k/adversarial-attacks-pytorch) | Adversarial attack library |
| [**RobustBench**](https://robustbench.github.io/) | Adversarially trained models & benchmarks |
| [**open-unlearning**](https://github.com/locuslab/open-unlearning) | Reference implementations of the LLM losses ported here |

<br>

## Citation

If you use this library in your research, please cite:

```bibtex
@inproceedings{kim2025unlearning,
  title     = {Unlearning-Aware Minimization},
  author    = {Kim, Hoki and Kim, Keonwoo and Chae, Sungwon and Yoon, Sangwon},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {38},
  year      = {2025}
}
```

Paper: [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/116406) · [OpenReview](https://openreview.net/forum?id=kAuckbcMvi)

Please also cite the original papers of the methods and benchmarks you use — references are in the registry (`describe("<name>")`) and in [docs/multimodal_papers.md](docs/multimodal_papers.md).

<br>

## Contributing

Issues and pull requests are welcome. To add a new unlearning method:

1. Put training-based methods in `torchunlearn/unlearn/trainers/` (subclass `Unlearner`; sequence methods subclass `SeqUnlearner` in `trainers/seq/`) and non-training methods in `torchunlearn/unlearn/nontrainers/`.
2. Export the class from `torchunlearn/unlearn/__init__.py`, add it to `__all__`, and register it in `torchunlearn/api/algorithms.py` with its hyper-parameters and modality.
3. Add a row to its domain's method table with a link to the original paper.
4. Report benchmark numbers against the Retrain oracle (vision) or the benchmark's own protocol (LLM / multimodal), and add a CPU test on the tiny models in `tests/`.

<br>

## License

Released under the [MIT License](LICENSE).

<div align="center">
<sub>Built with ❤️ by the <a href="https://trustworthyai.co.kr">TrustworthyAI Lab</a></sub>
</div>
