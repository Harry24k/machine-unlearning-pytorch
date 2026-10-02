<div align="center">

# 🧠 Machine-Unlearning-PyTorch

### One PyTorch library for machine unlearning — vision classifiers, large language models, and multimodal (vision-language) models.

<p>
<a href="https://github.com/Harry24k/machine-unlearning-pytorch/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-brightgreen?style=flat-square" /></a>
<a href="https://pypi.org/project/torchunlearn/"><img alt="PyPI" src="https://img.shields.io/pypi/v/torchunlearn.svg?color=orange&style=flat-square" /></a>
<img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.8-blue?style=flat-square" />
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E%3D1.7.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
<img alt="Transformers" src="https://img.shields.io/badge/transformers-4.5x-FFD21E?style=flat-square" />
</p>

<p>
<img alt="Vision" src="https://img.shields.io/badge/🖼%20Vision-21%20methods-blueviolet?style=for-the-badge" />
<img alt="LLM" src="https://img.shields.io/badge/📝%20LLM-17%20methods-blueviolet?style=for-the-badge" />
<img alt="Multimodal" src="https://img.shields.io/badge/🖼📝%20Multimodal-8%20paper%20methods%20·%207%20benchmarks-blueviolet?style=for-the-badge" />
</p>

📰 <a href="https://trustworthyai.co.kr/article/2025/uam-eng/">Blog Post</a> &nbsp;·&nbsp;
📄 <a href="https://neurips.cc/virtual/2025/poster/116406">NeurIPS 2025 Paper</a> &nbsp;·&nbsp;
📓 <a href="demo.ipynb">Demo Notebook</a>
<a href="https://colab.research.google.com/github/Harry24k/machine-unlearning-pytorch/blob/main/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:18px;vertical-align:middle" /></a> &nbsp;·&nbsp;
📚 <a href="docs/multimodal_unlearning.md">LLM / Multimodal Guide</a> &nbsp;·&nbsp;
🗂 <a href="docs/multimodal_papers.md">VLM Paper Table</a>

</div>

<br>

> **Torchunlearn** gives every domain the same *PyTorch-like* workflow:
> **wrap the model → build forget / retain loaders → pick a method → `setup()` → `fit()` → evaluate.**

Machine unlearning removes the influence of specific training data from a trained model, as if that data was never used:

| | |
|:--|:--|
| 🔒 **Privacy** | GDPR "right to be forgotten", PII removal from LLMs and VLMs |
| 🛠 **Data correction** | remove mislabeled or corrupted samples |
| ⚖️ **Bias mitigation** | eliminate biased training data |
| 🛡 **Security & safety** | purge backdoors / poisoned examples, remove unsafe behaviour from VLMs |

<br>

## 🧭 Choose your domain

<table>
<tr>
<th width="33%" align="center">🖼 Vision</th>
<th width="33%" align="center">📝 LLM</th>
<th width="33%" align="center">🖼📝 Multimodal</th>
</tr>
<tr valign="top">
<td>

**Unlearn from** image classifiers (ResNet, ViT, …)

**Wrapper** `RobModel`

**Methods** Finetune · NegGrad · RandomLabel · L1Sparse · SCRUB · BadTeacher · BoundaryShrink · SalUn · UAM · ARU · AMUN · SFRon · RFE · MUMis · FaLW · FisherForget · Influence · NegMerge · SISA · REM · Amnesiac

**Evaluate** retain / forget / test accuracy, MIA, ZRF, adversarial robustness, `BenchmarkSuite`

[→ Vision section](#-vision)

</td>
<td>

**Unlearn from** any HF causal LM — base or fine-tuned, QA-level or span-level (PII)

**Wrapper** `SeqRobModel` · `LLMRobModel`

**Methods** GradAscent · GradDiff · NPO · SimNPO · DPO · AltPO · RMU · WGA · SatImp · CEU · UNDIAL · PDU · JensUn · FLAT · PO · KL-Min · REVS

**Evaluate** answer log-prob, ROUGE-L / EM, truth ratio + KS test, Min-k% MIA, scope-aware T/S/C/G probes

[→ LLM section](#-llm)

</td>
<td>

**Unlearn from** vision-language models (LLaVA, Qwen-VL, Idefics, Gemma-3, …) and CLIP-style dual encoders

**Wrapper** `SeqRobModel` (LVLM) · `CLIPRobModel` (CLIP)

**Methods** every LLM method on image+text, plus SIU · ASRU · Safety Mirage · LUMoE · SLUG · ADU

**Evaluate** FIUBench, MLLMU-Bench, UMU-Bench, CLEAR, MLUBench, MMUBench, VLGuard evaluators, text-only leakage probe

[→ Multimodal section](#-multimodal)

</td>
</tr>
</table>

> 💡 The LLM and multimodal domains share **one code path**: a method only sees the answer-token `labels` and forwards
> whatever else the processor produced (`pixel_values`, `image_grid_thw`, …). A method written once runs on a text
> LLM and on an image+text model alike.

<br>

## 📋 Table of Contents

<table>
<tr valign="top">
<td>

**Getting started**
- [Installation](#-installation)
- [Project layout](#-project-layout)
- [Registry and CLI](#-registry-and-cli)
- [Tests](#-tests)

</td>
<td>

**Domains**
- [🖼 Vision](#-vision) — [Quick Start](#quick-start) · [Scenarios](#forgetting-scenarios) · [Methods](#supported-methods) · [Evaluation](#evaluation) · [Benchmarks](#benchmark-results)
- [📝 LLM](#-llm) — [QA-level](#qa--instruction-level-forgetting) · [Span-level PII](#span-level-pii-forgetting)
- [🖼📝 Multimodal](#-multimodal) — [LVLM](#vision-language-models) · [Paper methods](#methods-from-the-multimodal-unlearning-literature) · [CLIP](#clip--dual-encoders)

</td>
<td>

**Project**
- [Related Projects](#-related-projects)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)

</td>
</tr>
</table>

<br>

## 🔨 Installation

```bash
pip install torchunlearn            # vision only
pip install "torchunlearn[llm]"     # + transformers, peft, datasets, Pillow, accelerate  (LLM / multimodal)
```

Latest development version:

```bash
pip install "git+https://github.com/Harry24k/machine-unlearning-pytorch.git#egg=torchunlearn[llm]"
```

| track | requirements |
|:--|:--|
| Vision | Python ≥ 3.8, PyTorch ≥ 1.7.1 (`torchvision`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `tqdm`) |
| LLM / Multimodal | Python ≥ 3.9, PyTorch ≥ 2.0, `transformers` 4.5x (Qwen3.5 / Qwen3-VL need 5.x), `peft`, `datasets`, `Pillow` |

> Models and datasets are **not** distributed with the library — point the wrappers at any local checkpoint or
> Hugging Face id you have access to.

<br>

## 🗂 Project layout

<details>
<summary><b>Click to expand</b></summary>

```
torchunlearn/
├── nn/            RobModel (vision)  ·  SeqRobModel (LLM + LVLM)  ·  CLIPRobModel (dual encoder)
├── unlearn/
│   ├── trainers/      vision trainers (finetune, neggrad, salun, scrub, aru, amun, sfron, rfe, mumis, falw, ...)  ·  llm_pii.py (LLM-*)
│   │   └── seq/       sequence trainers shared by LLM and LVLM (MM-*): base, 12 token-level methods,
│   │                  po, klmin, siu, grpo + asru, lumoe, safetymirage, finetune
│   ├── nontrainers/   fisherforget, influence, negmerge, sisa, rem, amnesiac, revs (LLM), slug (CLIP)
│   ├── clip/          ADU (domain unlearning for CLIP)
│   ├── recipes/       benchmark loaders: fiubench, mllmu_bench (+UMU), clear, mlubench, mmubench, domains, vlguard
│   ├── seq_data.py    SeqSample, SeqCollator (answer-only labels), splits, loaders
│   └── llm_data.py    span-level PII data (SpanDataset, build_spec, TSCGEvaluator)
├── metrics/       UnlearningEvaluator (vision) · SeqUnlearningEvaluator · mm_bench (benchmark evaluators) · judge (LLM-as-judge) · text
├── benchmarks/    BenchmarkSuite (vision)
├── api/           registry: register_unlearner / describe / build_unlearner(_split), HParam specs (modality = vision | seq | clip)
├── attacks/ optim/ utils/   adversarial attacks, UAM minimizer, datasets and vision models
scripts/           run_mm.py (LLM + LVLM: finetune | unlearn | eval)  ·  run_clip.py (slug | adu)  ·  run_pii_unlearn.py (span-level)
docs/              multimodal_unlearning.md (guide)  ·  multimodal_papers.md (paper table)  ·  llm_pii_unlearning.md
tests/             CPU smoke tests on tiny random-init models (LLaVA, Llama, CLIP)
```

</details>

<br>

---

<br>

## 🖼 Vision

### Quick Start

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

> **Demo notebook.** [`demo.ipynb`](demo.ipynb) expects a pretrained checkpoint at
> `./models/CIFAR10_Standard/last.pth` (not distributed) and a GPU runtime.

### Forgetting Scenarios

<table>
<tr><th>🎲 Random forgetting</th><th>🏷️ Classwise forgetting</th></tr>
<tr valign="top"><td>

```python
train_loaders, test_loaders = setup.get_loaders_for_rand(
    batch_size=128,
    ratio=0.1,        # fraction to forget
    stratified=True,  # keep class distribution
    seed=42,
)
```

</td><td>

```python
train_loaders, test_loaders = setup.get_loaders_for_classwise(
    batch_size=128,
    omit_label=1,     # class index to forget
    train_shuffle_and_transform=True,
)
```

</td></tr>
</table>

### Supported Methods

**Training-based**

| Method | Description | Reference |
|:---|:---|:---|
| **Finetune** | Fine-tune on the retain set only | [Warnecke et al., NDSS 2023](https://arxiv.org/abs/2108.11577) |
| **NegGrad** | Negative gradient on forget set | [Golatkar et al., CVPR 2020](https://arxiv.org/abs/1911.04933) |
| **RandomLabel** | Relabel forget set with random labels | [Golatkar et al., CVPR 2020](https://arxiv.org/abs/1911.04933) |
| **L1Sparse** | L1 sparsity regularization during fine-tuning | [Jia et al., NeurIPS 2023](https://arxiv.org/abs/2304.04934) |
| **SCRUB** | Alternating KL-max / KL-min distillation | [Kurmanji et al., NeurIPS 2023](https://arxiv.org/abs/2302.09880) |
| **BadTeacher** | Competent / bad-teacher knowledge distillation | [Chundawat et al., AAAI 2023](https://arxiv.org/abs/2205.08096) |
| **BoundaryShrink** | Nearest-class re-targeting to shrink the forget-class boundary | [Chen et al., CVPR 2023](https://arxiv.org/abs/2303.11570) |
| **SalUn** | Saliency-masked random-label fine-tuning | [Fan et al., ICLR 2024](https://arxiv.org/abs/2310.12508) |
| **UAM** | Unlearning-Aware Minimization | [Kim et al., NeurIPS 2025](https://neurips.cc/virtual/2025/poster/116406) |
| **ARU** | Adversarial Retain-free Unlearning | [Yoon et al., 2026](https://ieeexplore.ieee.org/document/11414433) |
| **AMUN** | Fine-tune on the nearest adversarial example of each forget sample | [Ebrahimpour-Boroojeny et al., ICML 2025](https://icml.cc/virtual/2025/poster/46097) |
| **SFRon** | Saliency forgetting in a remain-preserving manifold (fast/slow update) | [Huang et al., NeurIPS 2024](https://arxiv.org/abs/2409.19732) |
| **RFE** | Two-phase augmented Lagrangian + W2-regularized gradient projection | [Cheng et al., ICLR 2026](https://arxiv.org/abs/2603.26569) |
| **MUMis** | Suppress input sensitivity on the forget set (needs no retain data) | [Cheng et al., ICLR 2026](https://arxiv.org/abs/2402.15109) |
| **FaLW** | Forgetting-aware instance-wise loss reweighting for long-tailed forget sets | [Yu et al., 2026](https://arxiv.org/abs/2601.18650) |

**Non-training**

| Method | Description | Reference |
|:---|:---|:---|
| **FisherForget** | Fisher information matrix weight perturbation | [Golatkar et al., CVPR 2020](https://arxiv.org/abs/1911.04933) |
| **Influence** | Newton-step influence function removal | [Izzo et al., AISTATS 2021](https://arxiv.org/abs/2002.10077) |
| **NegMerge** | Sign-consensual weight merging | [Kim, Han & Choe, ICML 2025](https://arxiv.org/abs/2410.05583) |
| **SISA** | Sharded, isolated, sliced, aggregated retraining | [Bourtoule et al., S&P 2021](https://arxiv.org/abs/1912.03817) |
| **REM** | Redirection for erasing memory | — |
| **Amnesiac** | Revert the updates of specific training batches | [Graves et al., AAAI 2021](https://arxiv.org/abs/2010.10981) |

> **MUMis is retain-free but still training-based** — it never reads the Retain split, so pass the Forget loader directly to `fit`.
> **UAM is a minimizer, not a trainer** — it wraps any trainer through `Trainer.setup(minimizer=...)`.

<details>
<summary><b>Usage examples (NegGrad · UAM · ARU · AMUN · SFRon · FaLW · MUMis · RFE · FisherForget · NegMerge)</b></summary>

```python
from torchunlearn.unlearn import NegGrad, Standard, ARU, AMUN, SFRon, FaLW, MUMis, RFE, FisherForget, NegMerge
opt = "SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)"

# NegGrad
NegGrad(rmodel, retain_lambda=0.5).setup(optimizer=opt, n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)

# UAM (via the Standard trainer)
Standard(rmodel).setup(optimizer=opt, minimizer=f"UAM(rho={rho}, cosine_total_step={cosine_total_step}, gamma={gamma})",
                       n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)

# ARU / AMUN / SFRon
ARU(rmodel, margin=1.0, eps=0.05, steps=50, omit_label=1).setup(optimizer=opt, n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)
AMUN(rmodel, attack="deepfool", steps=20).setup(optimizer=opt, n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)
SFRon(rmodel, saliency_ratio=0.5, slow_alpha=0.5, slow_every=5).setup(optimizer=opt, n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)

# FaLW (needs a held-out validation loader; set estimate_every > 1 to amortize the per-step validation pass)
trainer = FaLW(rmodel, tau=0.15)
trainer.prepare(train_loaders["Forget"], val_loader)
trainer.setup(optimizer=opt, n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)

# MUMis (retain-free; unbounded objective -> small lr + early stopping, watch Clean(R))
MUMis(rmodel, other_lambda=1.0, n_other=1).setup(optimizer="SGD(lr=1e-4)", n_epochs=1).fit(train_loaders=train_loaders["Forget"], n_epochs=1)

# RFE ("Adjacent" = retain samples entangled with the forget set, e.g. sibling CIFAR-100 subclasses)
merged_loader = MergedLoaders({"Retain": train_loaders["Retain"], "Forget": train_loaders["Forget"], "Adjacent": adjacent_loader})
RFE(rmodel, phase1_steps=100, constraint_tol=0.05, w2_lambda=1.0).setup(optimizer=opt, n_epochs=5).fit(train_loaders=merged_loader, n_epochs=5)

# Non-training
FisherForget(rmodel).fit(train_loaders, alphas=[1e-9, 1e-8, 1e-7, 1e-6], repeat=3, save_path="./models/fisher")
NegMerge(rmodel).fit(train_loaders, lrs=[1e-4, 5e-4, 1e-3], epochs=1, repeats=3, scaling=1.0, consensus_ratio=1.0,
                     aggregation="mean", save_path="./models/negmerge")
```

</details>

### Evaluation

<table>
<tr><th>During unlearning</th><th>After unlearning</th></tr>
<tr valign="top"><td>

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

</td><td>

```python
from torchunlearn.metrics import UnlearningEvaluator
from torchunlearn.benchmarks import BenchmarkSuite

UnlearningEvaluator(unlearned_model=rmodel, train_loaders=train_loaders,
                    test_loaders=test_loaders,
                    retrained_model=retrained_rmodel,   # for UG
                    n_limit=1000).print_report()

suite = BenchmarkSuite(train_loaders, test_loaders, retrained_model=retrained_rmodel)
suite.add("Finetune", ft_model); suite.add("NegGrad", ng_model)
suite.print_table()
```

</td></tr>
</table>

<details>
<summary><b>Sample training log</b> (Finetune, CIFAR-10, 10% random forgetting)</summary>

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

### Benchmark Results

CIFAR-10 / ResNet-18 · 5 epochs · SGD (lr 0.01, momentum 0.9, wd 5e-4) · 3 seeds.
**RA** retain acc (↑) · **FA** forget acc (should match Retrain) · **TA** test acc (↑) · **ΔAcc** = |ΔRA| + |ΔFA| + |ΔTA| vs Retrain (↓)

<details>
<summary><b>🎲 Random forgetting — 10% of training data</b></summary>

| Algorithm | RA | FA | TA | time(s) | **ΔAcc** |
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
<summary><b>🏷️ Classwise forgetting — forget one class</b></summary>

| Algorithm | RA | FA | TA | time(s) | **ΔAcc** |
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

---

<br>

## 📝 LLM

Two granularities, both on **any Hugging Face causal LM you load yourself** — a base model, your fine-tuned
checkpoint, or a PEFT adapter directory.

| | QA / instruction-level | Span-level (PII) |
|:--|:--|:--|
| **Forget unit** | a (prompt, answer) pair | a text span inside a document (a name, an address, …) |
| **Wrapper** | `SeqRobModel` | `LLMRobModel` |
| **Registry prefix** | `MM-*` (shared with multimodal) | `LLM-*` |
| **Loss target** | answer tokens only | span tokens only |
| **Evaluation** | `SeqUnlearningEvaluator`, truth ratio / KS, judge | scope-aware **T/S/C/G** probes (`TSCGEvaluator`) |
| **Docs** | [docs/multimodal_unlearning.md](docs/multimodal_unlearning.md) | [docs/llm_pii_unlearning.md](docs/llm_pii_unlearning.md) |

### QA / instruction-level forgetting

The chat template is rendered twice (prompt-only vs prompt+answer) and the answer span is what remains after the
longest common token prefix, so there are no tokenizer-specific boundary bugs. Build the model to unlearn from with
`SeqFinetune`, then unlearn.

```python
from torchunlearn import SeqRobModel, SeqUnlearningEvaluator
from torchunlearn.unlearn.seq_data import from_jsonl, split_forget_retain, SeqCollator, make_loader, build_unlearn_loaders
from torchunlearn.unlearn.trainers.seq import SeqFinetune, NPO

model = SeqRobModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device="cuda:0", trainable="lora:r=16,target=lm")
data  = from_jsonl("qa.jsonl", fields={"prompt": "question", "answer": "answer", "group": "subject"})   # text-only
col   = SeqCollator(model, alt_text="I don't know.")            # alt_* = refusal / alternate answer for DPO, PO, FLAT

# Stage I — fine-tune so the model knows the data
SeqFinetune(model).setup(optimizer="AdamW(lr=2e-5)").fit(make_loader(data, col, batch_size=8), n_epochs=3)

# Stage II — unlearn
forget, retain = split_forget_retain(data, by="group", ratio=0.1, seed=0)
ev = SeqUnlearningEvaluator({"Forget": forget, "Retain": retain}, col)
NPO(model, beta=0.1, alpha=1.0).set_evaluator(ev).setup(optimizer="AdamW(lr=1e-5)", n_epochs=5) \
   .fit(build_unlearn_loaders(forget, retain, col, batch_size=4), n_epochs=5)
model.save_pretrained("runs/npo/model")
```

<details>
<summary><b>Methods and metrics</b></summary>

| group | registry names |
|:--|:--|
| classic | `MM-GradAscent` · `MM-GradDiff` · `MM-NPO` · `MM-SimNPO` · `MM-DPO` (IDK-DPO) · `MM-AltPO` · `MM-RMU` |
| 2025 | `MM-WGA` · `MM-SatImp` · `MM-UNDIAL` · `MM-PDU` · `MM-FLAT` |
| baselines | `MM-PO` · `MM-KLMin` · `MM-Finetune` |

Evaluation: log-prob / token, sequence NLL, Min-k% MIA, ROUGE-L / EM / includes on greedy generations
(`SeqUnlearningEvaluator`); TOFU-style truth ratio and KS forget quality, multiple-choice accuracy, LLM-judge rubrics
(`torchunlearn.metrics.mm_bench`, `torchunlearn.metrics.judge`).

</details>

### Span-level (PII) forgetting

Labels cover only the span tokens; evaluation probes **T**arget / **S**ame-subject / **C**o-document / **G**lobal
leakage. 14 methods (`LLM-GradAscent` … `LLM-FLAT`, `LLM-CEU`, `LLM-JensUn`, non-gradient `LLM-REVS`) share the
loss formulas of the `MM-*` family. See `scripts/run_pii_unlearn.py`.

```python
from torchunlearn.unlearn.llm_data import load_docs, load_facts, build_spec, SpanDataset, make_collate, TSCGEvaluator
from torchunlearn.unlearn.trainers.llm_pii import LLMRobModel, LLM_TRAINERS
```

<br>

---

<br>

## 🖼📝 Multimodal

### Vision-language models

`SeqRobModel.from_pretrained` detects the modality from the config and loads `AutoModelForImageTextToText`
(LLaVA-1.5 / NeXT / OneVision, Qwen2-VL / Qwen2.5-VL, Idefics2/3, SmolVLM, InternVL, Gemma-3, PaliGemma, Pixtral,
Phi-4-MM, …). `trainable` selects what moves: `lm` · `vision` · `projector` · `lm+projector` · `lora:…` · a regex.

```python
from torchunlearn import SeqRobModel, SeqUnlearningEvaluator
from torchunlearn.unlearn.seq_data import from_llava_json, split_forget_retain, SeqCollator, build_unlearn_loaders, strip_images
from torchunlearn.unlearn.trainers.seq import GradDiff

model = SeqRobModel.from_pretrained("llava-hf/llava-1.5-7b-hf", device="cuda:0", trainable="lm")
data  = from_llava_json("train.json", image_root="imgs", group_key="id")      # LLaVA "conversations" JSON
forget, retain = split_forget_retain(data, by="group", ratio=0.1)
col = SeqCollator(model)                                                      # answer-only labels; image tokens never targeted
ev  = SeqUnlearningEvaluator({"Forget": forget, "Retain": retain,
                              "Forget_noimg": strip_images(forget)}, col)     # cross-modal leakage probe (text only)
GradDiff(model, grad_ckpt=True).set_evaluator(ev).setup(optimizer="AdamW(lr=1e-5)", n_epochs=5) \
        .fit(build_unlearn_loaders(forget, retain, col, batch_size=2), n_epochs=5)
```

### Methods from the multimodal unlearning literature

ICLR / ICML / NeurIPS 2024–2026 — the full table with fidelity notes is in [docs/multimodal_papers.md](docs/multimodal_papers.md).

| venue | paper | what it does | in torchunlearn |
|:--|:--|:--|:--|
| NeurIPS 2024 | **SIU** | single-image unlearning: multifaceted FT data + dual-masked KL | `MM-SIU` · `recipes.mmubench.build_siu_samples` |
| ICLR 2025 | **FIUBench** | fictitious facial identities; baselines GA / GD / KL / PO | `recipes.fiubench` · `FIUBenchEvaluator` · `MM-KLMin` · `MM-PO` |
| ICML 2025 | **SLUG** | single-layer, single-gradient unlearning for CLIP, transplantable into LLaVA | `CLIP-SLUG` · `SLUG.apply_to_vlm` |
| NeurIPS 2025 | **ADU** | approximate domain unlearning for CLIP (deep vision prompts, InstaPG, domain-disentangling loss) | `unlearn.clip.ADU` · `recipes.domains` |
| NeurIPS 2025 | **UMU-Bench** | unimodal vs multimodal knowledge, 653 profiles | `recipes.mllmu_bench` · `MLLMUBenchEvaluator` |
| ICLR 2026 | **Safety Mirage** | VLM safety by NPO / RMU unlearning on VLGuard | `MM-SafetyMirage-NPO` · `MM-SafetyMirage-RMU` · `SafetyMirageEvaluator` |
| ICML 2026 | **ASRU** | closed-form activation steering of one down-projection + GRPO with a rule-based refusal reward | `MM-ASRUSteer` · `MM-ASRU` |
| ICML 2026 | **MLUBench / LUMoE** | lifelong unlearning with switchable LoRA experts and an entity router | `trainers.seq.LUMoE` · `recipes.mlubench` · `MLUBenchEvaluator` |

**Benchmarks with recipes + evaluators:** FIUBench · MLLMU-Bench · UMU-Bench · CLEAR · MLUBench · MMUBench · VLGuard

### CLIP / dual encoders

```python
from torchunlearn import CLIPRobModel, SeqRobModel
from torchunlearn.unlearn.nontrainers.slug import SLUG, make_clip_loader

model = CLIPRobModel.from_pretrained("openai/clip-vit-large-patch14-336")
slug  = SLUG(model)
slug.compute_gradients(make_clip_loader(forget_pairs, model), make_clip_loader(retain_pairs, model))
slug.fit(eval_fn=lambda model: (forget_acc(model), test_acc(model)))          # binary search on the step size
slug.apply_to_vlm(SeqRobModel.from_pretrained("llava-hf/llava-1.5-7b-hf"))   # same vision tower → LLaVA forgets too
```

<br>

---

<br>

## 🧾 Registry and CLI

Every method is registered with its hyper-parameter spec and modality (`vision` · `seq` · `clip`):

```python
from torchunlearn.api import list_algorithms, describe, build_unlearner_split

list_algorithms(modality="seq")      # ['MM-Finetune', 'MM-GradAscent', ..., 'MM-ASRU', ..., 'LLM-NPO', ...]
describe("MM-NPO")                   # setup kwargs (optimizer, n_epochs, ...) and hparams with defaults
u, setup_kw = build_unlearner_split("MM-NPO", rmodel, hparams={"beta": 0.1, "trainable": "lm"})
```

LLM and LVLM share the same commands — text-only models simply have no images:

```bash
# LLM / LVLM
python scripts/run_mm.py finetune --model Qwen/Qwen2-VL-7B-Instruct --data llava:train.json --image-root imgs --trainable lm --epochs 3 --out runs/ft
python scripts/run_mm.py unlearn  --model runs/ft/model --data llava:train.json --image-root imgs --split group:0.1 \
                                  --method MM-NPO --hp beta=0.1 --trainable lm --grad-ckpt --epochs 5 --lr 1e-5 --out runs/npo --save-model
python scripts/run_mm.py eval     --model runs/npo/model --data Forget=jsonl:forget.jsonl Retain=jsonl:retain.jsonl --out runs/eval

# CLIP
python scripts/run_clip.py slug --model openai/clip-vit-large-patch14-336 --forget f.jsonl --retain r.jsonl \
                                --eval-forget ef.jsonl --eval-test et.jsonl --classes classes.txt --out runs/slug
python scripts/run_clip.py adu  --model openai/clip-vit-base-patch16 --root /data/office_home --forget-domains Clipart --out runs/adu
```

| data spec | meaning |
|:--|:--|
| `jsonl:<path>` | records; map fields with `--fields prompt=question,answer=answer,images=image,group=subject` |
| `llava:<path.json>` | LLaVA "conversations" JSON (VLGuard / MLLMU exports) |
| `hf:<name>[:<split>]` | Hugging Face datasets |

> On shared GPUs always pass `--mem-fraction`.

<br>

## ✅ Tests

```bash
python -m pytest tests -q
```

CPU-only, on tiny random-init models (LLaVA, Llama, CLIP built from configs; only the `sshleifer/tiny-gpt2`
tokenizer is downloaded): answer-mask invariants, every registered sequence method moving only its declared
parameters, fine-tune → unlearn → save → reload round-trips, LoRA adapters, the paper methods (SIU, ASRU, LUMoE,
SLUG, ADU), benchmark recipe parsers and every evaluator's metric keys.

<br>

## 🔗 Related Projects

- [**MAIR**](https://github.com/Harry24k/MAIR) — Adversarial Training Framework (NeurIPS'23)
- [**Torchattacks**](https://github.com/Harry24k/adversarial-attacks-pytorch) — Adversarial Attack Library
- [**RobustBench**](https://robustbench.github.io/) — Adversarially Trained Models & Benchmarks
- [**open-unlearning**](https://github.com/locuslab/open-unlearning) — reference implementations of the LLM losses ported here

<br>

## 📝 Citation

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

Please also cite the original papers of the methods and benchmarks you use — references are listed next to each
registry entry (`describe("<name>")`) and in [docs/multimodal_papers.md](docs/multimodal_papers.md).

<br>

## 🤝 Contributing

Issues and pull requests are welcome. Adding a new unlearning method:

1. Training-based methods go in `torchunlearn/unlearn/trainers/` (subclass `Unlearner`; sequence methods subclass
   `SeqUnlearner` in `trainers/seq/`), non-training methods in `torchunlearn/unlearn/nontrainers/`.
2. Export the class from `torchunlearn/unlearn/__init__.py` (and `__all__`); register it in
   `torchunlearn/api/algorithms.py` with its hyper-parameters and modality.
3. Add a row to the method table of its domain with a link to the original paper.
4. Report numbers against the Retrain oracle (vision) or the benchmark's own protocol (LLM / multimodal), and add a
   CPU test on the tiny models in `tests/`.

<br>

## 📄 License

Released under the [MIT License](LICENSE).

<div align="center">
<sub>Built with ❤️ by the <a href="https://trustworthyai.co.kr">TrustworthyAI Lab</a></sub>
</div>
