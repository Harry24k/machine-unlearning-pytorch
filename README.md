<div align="center">

# 🧠 Machine-Unlearning-PyTorch

**A PyTorch library for efficient machine unlearning — make your models forget, on demand.**

<a href="https://github.com/Harry24k/machine-unlearning-pytorch/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-brightgreen?style=flat-square" /></a>
<a href="https://pypi.org/project/torchunlearn/"><img alt="PyPI" src="https://img.shields.io/pypi/v/torchunlearn.svg?color=orange&style=flat-square" /></a>
<img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue?style=flat-square" />
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E%3D1.7.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />

<br>

📰 <a href="https://trustworthyai.co.kr/article/2025/uam-eng/">Blog Post</a> &nbsp;&middot;&nbsp;
📄 <a href="https://neurips.cc/virtual/2025/loc/san-diego/poster/116406">NeurIPS 2025 Paper</a> &nbsp;&middot;&nbsp;
<a href="demo.ipynb">Demo Notebook</a> &nbsp;&middot;&nbsp;
  <a href="https://colab.research.google.com/github/Harry24k/machine-unlearning-pytorch/blob/main/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:20px;" /></a>

</div>

---

**Torchunlearn** is a PyTorch library providing a unified, *PyTorch-like* interface for state-of-the-art machine unlearning algorithms.

Machine unlearning removes the influence of specific training data from a trained model, as if that data was never used. This matters for:

- 🔒 **Privacy compliance** — GDPR "right to be forgotten"
- 🛠 **Data correction** — remove mislabeled or corrupted samples
- ⚖️ **Bias mitigation** — eliminate biased training data
- 🛡 **Security** — purge backdoor or poisoned examples
     
---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Forgetting Scenarios](#-forgetting-scenarios)
- [Supported Methods](#-supported-methods)
- [Benchmark Results](#-benchmark-results)
- [Evaluation](#-evaluation)
- [Related Projects](#-related-projects)
- [Citation](#-citation)
- [Contributing](#-contributing)
           
---

## ⚡ Quick Start

```python
import torchunlearn
from torchunlearn.unlearn.trainers.finetune import Finetune

# 1. Wrap your model
rmodel = torchunlearn.RobModel(model, n_classes=10,
                           normalization_used={'mean': [0.5], 'std': [0.5]})

# 2. Setup data loaders (Retain / Forget / Test)
setup = torchunlearn.utils.data.UnlearnDataSetup(
data_name="CIFAR10", n_classes=10,
mean=[0.4914, 0.4822, 0.4465],
std=[0.2023, 0.1994, 0.2010])
train_loaders, test_loaders = setup.get_loaders_for_rand(
batch_size=128, ratio=0.1, stratified=True)

# 3. Load a pretrained model
rmodel.load_dict('save_dict.pth')

# 4. Unlearn!
trainer = Finetune(rmodel)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", n_epochs=5)
trainer.fit(train_loaders=train_loaders['Retain'], n_epochs=5,
        save_path="./models/unlearned")
```

---

## 🔨 Installation

**Requirements:** Python >= 3.6, PyTorch >= 1.7.1

```bash
pip install torchunlearn
```

---

## 🎯 Forgetting Scenarios

### Random Forgetting
Forget a randomly sampled subset of training data (e.g., 10%):

```python
train_loaders, test_loaders = setup.get_loaders_for_rand(
batch_size=128,
ratio=0.1,        # fraction to forget
stratified=True,  # preserve class distribution
seed=42
)
```

### Classwise Forgetting
Forget all samples belonging to a specific class:

```python
train_loaders, test_loaders = setup.get_loaders_for_classwise(
batch_size=128,
omit_label=1,                    # class index to forget
train_shuffle_and_transform=True
)
```

---

## 🔬 Supported Methods

### Training-based Methods

| Method | Description | Reference |
|:------:|:-------------:|:-----------:|
| **Finetune** | Retrain on retain set only | Baseline |
| **NegGrad** | Negative gradient on forget set | [Golatkar et al., 2020](https://arxiv.org/abs/2004.09932) |
| **RandomLabel** | Relabel forget set with random labels | [Golatkar et al., 2020](https://arxiv.org/abs/2004.09932) |
| **L1Sparse** | L1 sparsity regularization on forget set | [Jia et al., 2023](https://arxiv.org/abs/2304.04934) |
| **SCRUB** | Alternating KL-max / KL-min distillation | [Kurmanji et al., 2023](https://arxiv.org/abs/2302.09621) |
| **BadTeacher** | Competent / bad-teacher knowledge distillation | [Chundawat et al., 2023](https://arxiv.org/abs/2205.08096) |
| **BoundaryShrink** | Nearest-class re-targeting to shrink forget-class boundary | [Chen et al., 2023](https://arxiv.org/abs/2301.11557) |
| **SalUn** | Saliency-masked random-label fine-tuning | [Fan et al., 2024](https://arxiv.org/abs/2304.04934) |
| **UAM** | Unlearning-Aware Minimization | [Kim et al., 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/116406) |
| **ARU** | Adversarial Retain-free Unlearning | [Yoon et al., 2026](https://ieeexplore.ieee.org/document/11414433) |

### Non-Training Methods

| Method | Description | Reference |
|:------:|:-------------:|:-----------:|
| **FisherForget** | Fisher information matrix weight perturbation | [Golatkar et al., 2020](https://arxiv.org/abs/2004.09932) |
| **Influence** | Newton-step influence function removal | [Izzo et al., 2021](https://arxiv.org/abs/2012.09822) |
| **Amnesiac** | Revert learning from specific training batches | [Graves et al., 2021](https://arxiv.org/abs/2010.10981) |
| **NegMerge** | Consensual weight negation for unlearning | [Kim et al., 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/116406) |

<details>
<summary>Click to expand usage examples</summary>

**Finetune**
```python
from torchunlearn.unlearn.trainers.finetune import Finetune
trainer = Finetune(rmodel)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**NegGrad**
```python
from torchunlearn.unlearn.trainers.neggrad import NegGrad
trainer = NegGrad(rmodel, retain_lambda=0.5)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**UAM**
```python
from torchunlearn.unlearn.trainers.standard import Standard
trainer = Standard(rmodel)
trainer.setup(
optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)",
minimizer=f"UAM(rho={rho}, cosine_total_step={cosine_total_step}, gamma={gamma})",
n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**FisherForget**
```python
from torchunlearn.unlearn.nontrainers.fisherforget import FisherForget
unlearner = FisherForget(rmodel)
unlearner.fit(train_loaders, alphas=[1e-9, 1e-8, 1e-7, 1e-6], repeat=3,
          save_path="./models/fisher")
```

**NegMerge**
```python
from torchunlearn.unlearn.nontrainers.negmerge import NegMerge
unlearner = NegMerge(rmodel)
unlearner.fit(train_loaders, lrs=[1e-4, 5e-4, 1e-3], epochs=1, repeats=3,
          scaling=1.0, consensus_ratio=1.0, aggregation="mean",
          save_path="./models/negmerge")
```

</details>

---

## 📊 Benchmark Results

Evaluated on **CIFAR-10 / ResNet-18**. 
Training methods run for **5 epochs** with SGD (lr=0.01, momentum=0.9, wd=5e-4). 
Results averaged over 3 seeds.

**Metrics:**
- RA = Retain accuracy (higher is better)
- FA = Forget accuracy (lower is better)
- TA = Test accuracy
- time(s) = wall-clock unlearning time in seconds
- $\Delta\mathcal{Acc}$ = $|\Delta\text{RA}| + |\Delta\text{FA}| + |\Delta\text{TA}|$ vs. Retrain oracle (lower is better)

### 🎲 Random Forgetting — 10% of training data

| Algorithm | RA | FA | TA | time(s) | **ΔAcc** |
|---|:---:|:---:|:---:|:---:|:---:|
| Finetune | 100.00 | 99.84 | 94.26 | 32.2 | 94.90 |
| NegGrad | 10.00 | 10.00 | 10.00 | 48.6 | 169.20 |
| SCRUB | 31.69 | 32.02 | 31.86 | 35.1 | 147.07 |
| BadTeacher | 99.70 | 99.74 | 93.11 | 42.7 | 93.35 |
| BoundaryShrink | 87.71 | 82.34 | 80.77 | 46.1 | 92.46 |
| SalUn | 99.64 | 99.72 | 92.95 | 63.6 | 93.41 |
| FisherForget | 9.99 | 10.00 | 10.01 | 65.5 | 169.20 |
| Influence | 99.98 | 99.94 | 94.48 | 49.2 | 95.20 |
| Standard | 99.99 | 99.56 | 93.75 | 46.4 | 94.10 |
| ARU | 90.35 | 90.12 | 84.75 | 42.9 | 93.62 |
| L1Sparse | 100.00 | 99.92 | 94.28 | 43.8 | 95.00 |
| RandomLabel | 49.71 | 51.24 | 46.82 | 44.1 | 133.31 |
| Amnesiac | 99.98 | 99.94 | 94.48 | 25.9 | 95.20 |
| NegMerge | 99.46 | 99.24 | 93.00 | 29.4 | 92.70 |
| UAM | 100.00 | 85.72 | 85.32 | — | 87.40 |

### 🏷️ Classwise Forgetting — Forget one class

| Algorithm | RA | FA | TA | time(s) | **ΔAcc** |
|---|:---:|:---:|:---:|:---:|:---:|
| Finetune | 100.00 | 95.17 | 94.24 | 36.3 | 99.11 |
| NegGrad | 11.11 | 0.00 | 11.11 | 40.5 | 168.08 |
| SCRUB | 14.31 | 0.00 | 14.27 | 34.0 | 161.72 |
| BadTeacher | 99.41 | 15.54 | 92.70 | 44.1 | 17.95 |
| BoundaryShrink | 98.41 | 0.00 | 91.52 | 40.0 | 2.59 |
| SalUn | 99.60 | 0.00 | 92.66 | 88.0 | 2.64 |
| FisherForget | 97.34 | 1.24 | 90.62 | 67.4 | 3.66 |
| Influence | 99.98 | 99.98 | 94.19 | 46.2 | 103.85 |
| Standard | 99.99 | 81.13 | 93.87 | 47.2 | 84.69 |
| ARU | 93.18 | 39.66 | 86.72 | 49.9 | 50.06 |
| L1Sparse | 100.00 | 99.80 | 94.06 | 44.2 | 103.56 |
| RandomLabel | 83.32 | 5.73 | 77.48 | 45.3 | 35.23 |
| Amnesiac | 99.98 | 99.98 | 94.19 | 30.1 | 103.85 |
| NegMerge | 96.31 | 0.08 | 89.58 | 29.0 | 4.49 |
| UAM | 100.00 | 0.00 | 90.84 | — | 4.86 |

## 📈 Evaluation

```python
loaders_with_flags = {
"(R)":  train_loaders['Retain'],
"(F)":  train_loaders['Forget'],
"(Te)": test_loaders['Test'],
}

trainer.record_rob(loaders_with_flags, n_limit=1000)
trainer.fit(
train_loaders=merged_loader, n_epochs=5,
save_path="./models/unlearned",
save_best={"Clean(R)": "HB", "Clean(F)": "LBO"},
record_type="Epoch"
)
```

**Sample training log** (Finetune, CIFAR-10, 10% random forgetting):

```
Epoch | Cost   | Clean(R) | Clean(F) | Clean(Te)
------+--------+----------+----------+----------
1   | 0.0913 |  96.48%  |  91.02%  |  91.80%
2   | 0.0524 |  96.39%  |  68.65%  |  91.02%
3   | 0.0884 |  95.80%  |  54.39%  |  91.02%
4   | 0.0525 |  96.58%  |  45.02%  |  90.04%
5   | 0.1073 |  97.36%  |  33.01%  |  91.41%
```

---

## 🔗 Related Projects

- [**MAIR**](https://github.com/Harry24k/MAIR) — Adversarial Training Framework (NeurIPS'23)
- [**Torchattacks**](https://github.com/Harry24k/adversarial-attacks-pytorch) — Adversarial Attack Library
- [**RobustBench**](https://robustbench.github.io/) — Adversarially Trained Models & Benchmarks

---

## 📝 Citation

 ```bibtex
  @article{kim2025unlearning,
    title     = {Unlearning-Aware Minimization},
    author    = {Kim, Hoki and Kim, Keonwoo and Chae, Sungwon and Yoon, Sangwon},
    booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    volume    = {39},
    pages     = {--},
    year      = {2025}
  }
  ```

  <sub>Built with ❤️ by the <a href="https://trustworthyai.co.kr">TrustworthyAI Lab</a></sub>
  </div></summary>
