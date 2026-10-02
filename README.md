<div align="center">

# 🧠 Machine-Unlearning-PyTorch

**A PyTorch library for machine unlearning across vision classifiers, large language models, and multimodal (vision-language) models — make your models forget, on demand.**

<a href="https://github.com/Harry24k/machine-unlearning-pytorch/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-brightgreen?style=flat-square" /></a>
<a href="https://pypi.org/project/torchunlearn/"><img alt="PyPI" src="https://img.shields.io/pypi/v/torchunlearn.svg?color=orange&style=flat-square" /></a>
<img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.8-blue?style=flat-square" />
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E%3D1.7.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
<img alt="Domains" src="https://img.shields.io/badge/domains-vision%20%7C%20LLM%20%7C%20multimodal-blueviolet?style=flat-square" />

<br>

📰 <a href="https://trustworthyai.co.kr/article/2025/uam-eng/">Blog Post</a> &nbsp;&middot;&nbsp;
📄 <a href="https://neurips.cc/virtual/2025/poster/116406">NeurIPS 2025 Paper</a> &nbsp;&middot;&nbsp;
<a href="demo.ipynb">Demo Notebook</a> &nbsp;&middot;&nbsp;
<a href="https://colab.research.google.com/github/Harry24k/machine-unlearning-pytorch/blob/main/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:20px;" /></a>
&nbsp;&middot;&nbsp; 📚 <a href="docs/multimodal_unlearning.md">LLM / multimodal guide</a> &nbsp;&middot;&nbsp;
🗂 <a href="docs/multimodal_papers.md">VLM paper table</a>

</div>

---

**Torchunlearn** is a PyTorch library providing a unified, *PyTorch-like* interface for state-of-the-art machine unlearning algorithms.

Machine unlearning removes the influence of specific training data from a trained model, as if that data was never used. This matters for:

- 🔒 **Privacy compliance** — GDPR "right to be forgotten"
- 🛠 **Data correction** — remove mislabeled or corrupted samples
- ⚖️ **Bias mitigation** — eliminate biased training data
- 🛡 **Security** — purge backdoor or poisoned examples, remove unsafe behaviour from VLMs

Every domain uses the same workflow — wrap the model, build forget / retain loaders, pick a method, `setup()`, `fit()`, evaluate:

| domain | what you unlearn from | model wrapper | methods | evaluation |
|:--|:--|:--|:--|:--|
| 🖼 **Vision** | image classifiers (ResNet, ViT, ...) | `RobModel` | 16 training-based + 5 non-training (Finetune, NegGrad, SalUn, SCRUB, UAM, ARU, AMUN, SFRon, RFE, MUMis, FaLW, FisherForget, NegMerge, SISA, REM, ...) | retain / forget / test accuracy, MIA, ZRF, adversarial robustness, `BenchmarkSuite` |
| 📝 **LLM** | any HF causal LM (base or fine-tuned), QA-level or span-level (PII) forgetting | `SeqRobModel` / `LLMRobModel` | GradAscent, GradDiff, NPO, SimNPO, DPO, AltPO, RMU, WGA, SatImp, CEU, UNDIAL, PDU, JensUn, FLAT, PO, KL-Min, REVS | answer log-prob, ROUGE-L / EM, truth ratio + KS test, Min-k% MIA, scope-aware T/S/C/G probes |
| 🖼📝 **Multimodal** | vision-language models (LLaVA, Qwen-VL, Idefics, Gemma-3, ...) and CLIP-style dual encoders | `SeqRobModel` (LVLM), `CLIPRobModel` (CLIP) | every LLM method on image+text inputs, plus SIU, ASRU, Safety Mirage, LUMoE, SLUG, ADU | FIUBench, MLLMU-Bench, UMU-Bench, CLEAR, MLUBench, MMUBench, VLGuard evaluators, text-only leakage probe |

The LLM and multimodal domains share one code path: a method only sees the answer-token `labels` and forwards
whatever else the processor produced (`pixel_values`, `image_grid_thw`, ...), so a method written once runs on a
text LLM and on an image+text model alike.

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Project layout](#-project-layout)
- [Vision](#-vision)
  - [Quick Start](#quick-start)
  - [Forgetting Scenarios](#forgetting-scenarios)
  - [Supported Methods](#supported-methods)
  - [Evaluation](#evaluation)
  - [Benchmark Results](#benchmark-results)
- [LLM](#-llm)
- [Multimodal](#-multimodal)
- [Registry and CLI](#-registry-and-cli)
- [Tests](#-tests)
- [Related Projects](#-related-projects)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔨 Installation

**Requirements:** Python >= 3.8, PyTorch >= 1.7.1 (vision); Python >= 3.9, PyTorch >= 2.0 and `transformers` 4.5x for the LLM / multimodal tracks.

```bash
pip install torchunlearn            # vision only
pip install "torchunlearn[llm]"     # + transformers, peft, datasets, Pillow, accelerate
```

Or, for the latest development version:

```bash
pip install "git+https://github.com/Harry24k/machine-unlearning-pytorch.git#egg=torchunlearn[llm]"
```

Models and datasets are **not** distributed with the library: point the wrappers at any local checkpoint or
Hugging Face id you have access to.  Qwen3.5 / Qwen3-VL need `transformers` 5.x.

---

## 🗂 Project layout

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

---

## 🖼 Vision

### Quick Start

```python
import torchunlearn
from torchunlearn.unlearn import Finetune
from torchunlearn.utils.data import UnlearnDataSetup, MergedLoaders

# 1. Wrap your model
model = torchunlearn.utils.load_model(model_name="ResNet18", n_classes=10)
rmodel = torchunlearn.RobModel(
    model,
    n_classes=10,
    normalization_used={"mean": [0.4914, 0.4822, 0.4465],
                        "std": [0.2023, 0.1994, 0.2010]},
)

# 2. Load a pretrained checkpoint (the model you want to unlearn from)
rmodel.load_dict("./models/CIFAR10_Standard/last.pth")

# 3. Set up data loaders (Retain / Forget / Test)
setup = UnlearnDataSetup(
    data_name="CIFAR10",
    n_classes=10,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)
train_loaders, test_loaders = setup.get_loaders_for_rand(
    batch_size=128, ratio=0.1, stratified=True
)
# train_loaders -> {"Retain": ..., "Forget": ...}
# test_loaders  -> {"Test": ...}

# Most trainers consume Retain and Forget batches together:
merged_loader = MergedLoaders(train_loaders)

# 4. Unlearn!
trainer = Finetune(rmodel)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5,
            save_path="./models/unlearned")
```

> **Note on the demo notebook.** [`demo.ipynb`](demo.ipynb) expects a pretrained
> checkpoint at `./models/CIFAR10_Standard/last.pth`, which is not distributed with
> this repository. Train one first (or drop in your own) before running it. The
> notebook also calls `.cuda()`, so a GPU runtime is required.

### Forgetting Scenarios

**Random forgetting** — forget a randomly sampled subset of training data (e.g., 10%):

```python
train_loaders, test_loaders = setup.get_loaders_for_rand(
    batch_size=128,
    ratio=0.1,        # fraction to forget
    stratified=True,  # preserve class distribution
    seed=42,
)
```

**Classwise forgetting** — forget all samples belonging to a specific class:

```python
train_loaders, test_loaders = setup.get_loaders_for_classwise(
    batch_size=128,
    omit_label=1,                     # class index to forget
    train_shuffle_and_transform=True,
)
```

### Supported Methods

**Training-based methods**

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

**Non-training methods**

| Method | Description | Reference |
|:---|:---|:---|
| **FisherForget** | Fisher information matrix weight perturbation | [Golatkar et al., CVPR 2020](https://arxiv.org/abs/1911.04933) |
| **Influence** | Newton-step influence function removal | [Izzo et al., AISTATS 2021](https://arxiv.org/abs/2002.10077) |
| **NegMerge** | Sign-consensual weight merging | [Kim, Han & Choe, ICML 2025](https://arxiv.org/abs/2410.05583) |
| **SISA** | Sharded, isolated, sliced, aggregated retraining | [Bourtoule et al., S&P 2021](https://arxiv.org/abs/1912.03817) |
| **REM** | Redirection for erasing memory | — |
| **Amnesiac** | Revert the updates of specific training batches | [Graves et al., AAAI 2021](https://arxiv.org/abs/2010.10981) |

> **MUMis is retain-free but still training-based.** It runs the normal
> trainer loop; it simply never reads the Retain split, so you can pass the
> Forget loader directly to `fit`.

> **UAM is a minimizer, not a trainer.** It wraps any trainer through the
> `minimizer=` argument of `Trainer.setup` — see the example below.

<details>
<summary><b>Click to expand usage examples</b></summary>

**Finetune**

```python
from torchunlearn.unlearn import Finetune

trainer = Finetune(rmodel)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**NegGrad**

```python
from torchunlearn.unlearn import NegGrad

trainer = NegGrad(rmodel, retain_lambda=0.5)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**UAM** (via the `Standard` trainer)

```python
from torchunlearn.unlearn import Standard

trainer = Standard(rmodel)
trainer.setup(
    optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)",
    minimizer=f"UAM(rho={rho}, cosine_total_step={cosine_total_step}, gamma={gamma})",
    n_epochs=5,
)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**ARU**

```python
from torchunlearn.unlearn import ARU

trainer = ARU(rmodel, margin=1.0, eps=0.05, steps=50, omit_label=1)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**AMUN**

```python
from torchunlearn.unlearn import AMUN

trainer = AMUN(rmodel, attack="deepfool", steps=20)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**SFRon**

```python
from torchunlearn.unlearn import SFRon

trainer = SFRon(rmodel, saliency_ratio=0.5, slow_alpha=0.5, slow_every=5)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**FaLW** (needs a held-out validation loader)

```python
from torchunlearn.unlearn import FaLW

trainer = FaLW(rmodel, tau=0.15)
trainer.prepare(train_loaders["Forget"], val_loader)  # balance factor + target-distribution source
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

> FaLW estimates a per-class Gaussian over the model's true-class probability
> on *unseen* (validation) data at every step, which is faithful to the paper's
> Algorithm 1 but costs one validation pass per optimization step. Set
> `estimate_every > 1` to amortize this at some fidelity cost.

**MUMis** (needs no retain data)

```python
from torchunlearn.unlearn import MUMis

trainer = MUMis(rmodel, other_lambda=1.0, n_other=1)
trainer.setup(optimizer="SGD(lr=1e-4)", n_epochs=1)
trainer.fit(train_loaders=train_loaders["Forget"], n_epochs=1)
```

> The MUMis objective is unbounded below, so it needs a small learning rate
> and early stopping. Watch `Clean(R)` and stop as soon as `Clean(F)` drops.

**RFE** (retain-forget entanglement)

```python
from torchunlearn.unlearn import RFE

# "Adjacent" holds the retain samples semantically entangled with the
# forget set -- e.g. sibling subclasses under the same CIFAR-100 superclass.
merged_loader = MergedLoaders({
    "Retain":   train_loaders["Retain"],
    "Forget":   train_loaders["Forget"],
    "Adjacent": adjacent_loader,
})

trainer = RFE(rmodel, phase1_steps=100, constraint_tol=0.05, w2_lambda=1.0)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**FisherForget**

```python
from torchunlearn.unlearn import FisherForget

unlearner = FisherForget(rmodel)
unlearner.fit(train_loaders, alphas=[1e-9, 1e-8, 1e-7, 1e-6], repeat=3,
              save_path="./models/fisher")
```

**NegMerge**

```python
from torchunlearn.unlearn import NegMerge

unlearner = NegMerge(rmodel)
unlearner.fit(train_loaders, lrs=[1e-4, 5e-4, 1e-3], epochs=1, repeats=3,
              scaling=1.0, consensus_ratio=1.0, aggregation="mean",
              save_path="./models/negmerge")
```

</details>

### Evaluation

**During unlearning** — register the loaders you want tracked, then train as usual:

```python
loaders_with_flags = {
    "(R)":  train_loaders["Retain"],
    "(F)":  train_loaders["Forget"],
    "(Te)": test_loaders["Test"],
}

trainer.record_rob(loaders_with_flags, n_limit=1000)
trainer.fit(
    train_loaders=merged_loader,
    n_epochs=5,
    save_path="./models/unlearned",
    save_best={"Clean(R)": "HB", "Clean(F)": "LBO"},
    record_type="Epoch",
)
```

**Sample training log** (Finetune, CIFAR-10, 10% random forgetting):

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

### Benchmark Results

Evaluated on **CIFAR-10 / ResNet-18**.
Training methods run for **5 epochs** with SGD (lr=0.01, momentum=0.9, wd=5e-4).
Results averaged over 3 seeds.

**Metrics**

| Symbol | Meaning | Direction |
|:---|:---|:---|
| RA | Retain accuracy | higher is better |
| FA | Forget accuracy | should match the Retrain oracle |
| TA | Test accuracy | higher is better |
| time(s) | Wall-clock unlearning time | lower is better |
| ΔAcc | \|ΔRA\| + \|ΔFA\| + \|ΔTA\| vs. the Retrain oracle | lower is better |

<details>
<summary><b>🎲 Random Forgetting — 10% of training data</b></summary>

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
<summary><b>🏷️ Classwise Forgetting — Forget one class</b></summary>

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

---

## 📝 LLM

Two granularities, both on **any Hugging Face causal LM you load yourself** (base model, your fine-tuned checkpoint,
or a PEFT adapter directory).

### QA / instruction-level forgetting (`SeqRobModel`, `MM-*` methods)

The forget unit is a (prompt, answer) pair; the loss is applied to the answer tokens only (the chat template is
rendered twice and the answer span is what remains after the longest common token prefix, so no tokenizer-specific
boundary bugs).  Build the model to unlearn from with `SeqFinetune`, then unlearn.

```python
from torchunlearn import SeqRobModel, SeqUnlearningEvaluator
from torchunlearn.unlearn.seq_data import from_jsonl, split_forget_retain, SeqCollator, make_loader, build_unlearn_loaders
from torchunlearn.unlearn.trainers.seq import SeqFinetune, NPO

model = SeqRobModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device="cuda:0", trainable="lora:r=16,target=lm")
data = from_jsonl("qa.jsonl", fields={"prompt": "question", "answer": "answer", "group": "subject"})   # text-only: no images
col = SeqCollator(model, alt_text="I don't know.")                 # alt_* = refusal / alternate for DPO, PO, FLAT

SeqFinetune(model).setup(optimizer="AdamW(lr=2e-5)").fit(make_loader(data, col, batch_size=8), n_epochs=3)   # stage I

forget, retain = split_forget_retain(data, by="group", ratio=0.1, seed=0)
ev = SeqUnlearningEvaluator({"Forget": forget, "Retain": retain}, col)
NPO(model, beta=0.1, alpha=1.0).set_evaluator(ev).setup(optimizer="AdamW(lr=1e-5)", n_epochs=5) \
   .fit(build_unlearn_loaders(forget, retain, col, batch_size=4), n_epochs=5)                                 # stage II
model.save_pretrained("runs/npo/model")
```

Methods: `MM-GradAscent`, `MM-GradDiff`, `MM-NPO`, `MM-SimNPO`, `MM-DPO` (IDK-DPO), `MM-AltPO`, `MM-RMU`, `MM-WGA`,
`MM-SatImp`, `MM-UNDIAL`, `MM-PDU`, `MM-FLAT`, `MM-PO`, `MM-KLMin`, `MM-Finetune` (the `MM-` prefix means
"sequence model", text-only or image+text).  Evaluation: log-prob / token, sequence NLL, Min-k% MIA, ROUGE-L / EM /
includes on greedy generations (`SeqUnlearningEvaluator`); TOFU-style truth ratio and KS forget quality, multiple-choice
accuracy and LLM-judge rubrics (`torchunlearn.metrics.mm_bench`, `metrics.judge`).

### Span-level (PII) forgetting (`LLMRobModel`, `LLM-*` methods)

The forget unit is a text span inside a document (a name, an address, ...); labels cover the span tokens only and
the evaluation is **scope-aware**: Target / Same-subject / Co-document / Global probes (T/S/C/G).  14 methods
(`LLM-GradAscent` ... `LLM-FLAT`, `LLM-CEU`, `LLM-JensUn`, non-gradient `LLM-REVS`) share the same loss formulas as
the `MM-*` family.  See [docs/llm_pii_unlearning.md](docs/llm_pii_unlearning.md) and `scripts/run_pii_unlearn.py`.

```python
from torchunlearn.unlearn.llm_data import load_docs, load_facts, build_spec, SpanDataset, make_collate, TSCGEvaluator
from torchunlearn.unlearn.trainers.llm_pii import LLMRobModel, LLM_TRAINERS
```

---

## 🖼📝 Multimodal

### Vision-language models (LVLM): the same sequence trainers, now with images

`SeqRobModel.from_pretrained` detects the modality from the config and loads `AutoModelForImageTextToText`
(LLaVA-1.5 / NeXT / OneVision, Qwen2-VL / Qwen2.5-VL, Idefics2/3, SmolVLM, InternVL, Gemma-3, PaliGemma, Pixtral,
Phi-4-MM, ...).  `trainable` selects what moves: `lm`, `vision`, `projector`, `lm+projector`, `lora:...`, or a regex.

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

### Methods from the multimodal unlearning literature (ICLR / ICML / NeurIPS 2024–2026)

| venue | paper | registry / class |
|---|---|---|
| NeurIPS 2024 | **SIU** – Single Image Unlearning (multifaceted FT data + dual-masked KL) | `MM-SIU`, `recipes.mmubench.build_siu_samples` |
| ICLR 2025 | **FIUBench** – fictitious facial identities; baselines GA / GD / KL / PO | `recipes.fiubench`, `mm_bench.FIUBenchEvaluator`, `MM-KLMin`, `MM-PO` |
| ICML 2025 | **SLUG** – single-layer single-gradient unlearning for CLIP, transplantable into LLaVA | `CLIP-SLUG` (`nontrainers/slug.py`, `apply_to_vlm`) |
| NeurIPS 2025 | **ADU** – approximate domain unlearning for CLIP (deep vision prompts, InstaPG, domain-disentangling loss) | `unlearn.clip.ADU`, `recipes.domains` |
| NeurIPS 2025 | **UMU-Bench** – unimodal vs multimodal knowledge, 653 profiles | `recipes.mllmu_bench`, `mm_bench.MLLMUBenchEvaluator` |
| ICLR 2026 | **Safety Mirage** – VLM safety by NPO / RMU unlearning on VLGuard | `MM-SafetyMirage-NPO`, `MM-SafetyMirage-RMU`, `build_vlguard_sets`, `SafetyMirageEvaluator` |
| ICML 2026 | **ASRU** – closed-form activation steering of one down-projection + GRPO with a rule-based refusal reward | `MM-ASRUSteer`, `MM-ASRU` (`trainers/seq/grpo.py`) |
| ICML 2026 | **MLUBench / LUMoE** – lifelong unlearning with switchable LoRA experts and an entity router | `trainers.seq.LUMoE`, `recipes.mlubench`, `MLUBenchEvaluator` |

Benchmarks with ready recipes and evaluators: **FIUBench**, **MLLMU-Bench**, **UMU-Bench**, **CLEAR**, **MLUBench**,
**MMUBench**, **VLGuard**.  The paper table with fidelity notes and the methods that were deliberately not
implemented is in [docs/multimodal_papers.md](docs/multimodal_papers.md).

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

---

## 🧾 Registry and CLI

Every method is registered with its hyper-parameter spec and modality (`vision` | `seq` | `clip`):

```python
from torchunlearn.api import list_algorithms, describe, build_unlearner_split
list_algorithms(modality="seq")          # ['MM-Finetune', 'MM-GradAscent', ..., 'MM-ASRU', ..., 'LLM-NPO', ...]
describe("MM-NPO")                       # setup kwargs (optimizer, n_epochs, ...) and hparams with defaults
u, setup_kw = build_unlearner_split("MM-NPO", rmodel, hparams={"beta": 0.1, "trainable": "lm"})
```

Command line (LLM and LVLM share the same commands; text-only models simply have no images):

```bash
python scripts/run_mm.py finetune --model Qwen/Qwen2-VL-7B-Instruct --data llava:train.json --image-root imgs --trainable lm --epochs 3 --out runs/ft
python scripts/run_mm.py unlearn  --model runs/ft/model --data llava:train.json --image-root imgs --split group:0.1 \
                                  --method MM-NPO --hp beta=0.1 --trainable lm --grad-ckpt --epochs 5 --lr 1e-5 --out runs/npo --save-model
python scripts/run_mm.py eval     --model runs/npo/model --data Forget=jsonl:forget.jsonl Retain=jsonl:retain.jsonl --out runs/eval
python scripts/run_clip.py slug   --model openai/clip-vit-large-patch14-336 --forget f.jsonl --retain r.jsonl --eval-forget ef.jsonl --eval-test et.jsonl --classes classes.txt --out runs/slug
python scripts/run_clip.py adu    --model openai/clip-vit-base-patch16 --root /data/office_home --forget-domains Clipart --out runs/adu
```

Data specs: `jsonl:<path>` (+ `--fields prompt=question,answer=answer,images=image,group=subject`), `llava:<path.json>`,
`hf:<name>[:<split>]`.  Shared GPUs: pass `--mem-fraction`.

---

## ✅ Tests

```bash
python -m pytest tests -q
```

The suites run on CPU with tiny random-init models (LLaVA, Llama, CLIP built from configs; only the
`sshleifer/tiny-gpt2` tokenizer is downloaded): answer-mask invariants, every registered sequence method moving only
its declared parameters, fine-tune → unlearn → save → reload round-trips, LoRA adapters, the paper methods (SIU, ASRU,
LUMoE, SLUG, ADU), benchmark recipe parsers and every evaluator's metric keys.

---

## 🔗 Related Projects

- [**MAIR**](https://github.com/Harry24k/MAIR) — Adversarial Training Framework (NeurIPS'23)
- [**Torchattacks**](https://github.com/Harry24k/adversarial-attacks-pytorch) — Adversarial Attack Library
- [**RobustBench**](https://robustbench.github.io/) — Adversarially Trained Models & Benchmarks
- [**open-unlearning**](https://github.com/locuslab/open-unlearning) — reference implementations of the LLM losses ported here

---

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

Please also cite the original papers of the methods and benchmarks you use (references are listed next to each
entry in the registry: `describe("<name>")`, and in `docs/multimodal_papers.md`).

---

## 🤝 Contributing

Issues and pull requests are welcome. If you are adding a new unlearning method:

1. Put training-based methods in `torchunlearn/unlearn/trainers/` (subclass `Unlearner`; sequence methods subclass
   `SeqUnlearner` in `trainers/seq/`) and non-training methods in `torchunlearn/unlearn/nontrainers/`.
2. Export the new class from `torchunlearn/unlearn/__init__.py` and add it to `__all__`; register it in
   `torchunlearn/api/algorithms.py` with its hyper-parameters and modality.
3. Add a row to the method table of its domain with a link to the original paper.
4. Report benchmark numbers against the Retrain oracle (vision) or the benchmark's own protocol (LLM / multimodal),
   and add a CPU test on the tiny models in `tests/`.

---

## 📄 License

Released under the [MIT License](LICENSE).

<div align="center">
<sub>Built with ❤️ by the <a href="https://trustworthyai.co.kr">TrustworthyAI Lab</a></sub>
</div>
