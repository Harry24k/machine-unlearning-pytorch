# Machine-Unlearning-PyTorch

<p>
  <a href="https://github.com/Harry24k/machine-unlearning-pytorch/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/Harry24k/machine-unlearning-pytorch?&color=brightgreen" /></a>
  <a href="https://pypi.org/project/torchunlearn/"><img alt="Pypi" src="https://img.shields.io/pypi/v/torchunlearn.svg?&color=orange" /></a>
</p>

> For a detailed introduction to [Unlearning-Aware Minimization](https://neurips.cc/virtual/2025/loc/san-diego/poster/116406), please refer to our lab’s article available at: [https://trustworthyai.co.kr/article/2025/uam-eng/](https://trustworthyai.co.kr/article/2025/uam-eng/).

**Torchunlearn is a PyTorch library that provides machine unlearning methods to make trained models forget specific data.**

Machine unlearning is the process of removing the influence of specific training data from a trained model, as if that data had never been used during training. This is crucial for:
- **Privacy compliance** (e.g., GDPR "right to be forgotten")
- **Data correction** (removing mislabeled or corrupted data)
- **Bias mitigation** (eliminating biased training samples)
- **Security** (removing backdoor or poisoned data)

It contains *PyTorch-like* interface and functions that make it easier for PyTorch users to implement machine unlearning.

```python
import torchunlearn
from torchunlearn.unlearn.trainers.finetune import Finetune

# Wrap your model
rmodel = torchunlearn.RobModel(model, n_classes=10, 
                               normalization_used={'mean': [0.5], 'std': [0.5]})

# Setup data loaders (Retain, Forget, Test)
setup = torchunlearn.utils.data.UnlearnDataSetup(data_name="CIFAR10", 
                                                  n_classes=10, 
                                                  mean=[0.4914, 0.4822, 0.4465], 
                                                  std=[0.2023, 0.1994, 0.2010])
train_loaders, test_loaders = setup.get_loaders_for_rand(batch_size=128, 
                                                          ratio=0.1, 
                                                          stratified=True)

# Load your model
rmodel.load_dict('save_dict.pth')

# Start unlearning
trainer = Finetune(rmodel)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", 
              n_epochs=5)
trainer.fit(train_loaders=train_loaders['Retain'], n_epochs=5, 
            save_path="./models/unlearned")
```


## :bar_chart: Key Features

### Data Setup Utilities

- **UnlearnDataSetup**: Automatic data splitting for unlearning scenarios
- **MergedLoaders**: Combine multiple data loaders (Retain + Forget)
- Support for multiple datasets: CIFAR10, CIFAR100, TinyImageNet, MNIST variants

### Unified Interface

All unlearning methods follow a consistent PyTorch-like API:
```python
# Initialize
trainer = UnlearningMethod(rmodel, **method_params)

# Setup training
trainer.setup(optimizer="...", scheduler="...", n_epochs=...)

# Train (unlearn)
trainer.fit(train_loaders=..., n_epochs=..., save_path="...")

# Evaluate
accuracy = rmodel.eval_accuracy(data_loader=test_loader)
```

## :hammer: Requirements and Installation

**Requirements**

- PyTorch version >=1.7.1
- Python version >=3.6

**Installation**

```bash
# pip
pip install torchunlearn

# source
pip install git+https://github.com/Harry24k/machine-unlearning-pytorch.git

# git clone
git clone https://github.com/Harry24k/machine-unlearning-pytorch.git
cd machine-unlearning-pytorch/
pip install -e .
```

## :rocket: Getting Started

**[Demo](https://github.com/Harry24k/machine-unlearning-pytorch/blob/master/demo.ipynb)**

### Basic Usage

```python
import torch
import torchunlearn
from torchunlearn.utils.data import UnlearnDataSetup, MergedLoaders

# Configuration
MODEL_NAME = "ResNet18"
DATA_NAME = "CIFAR10"
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]
N_CLASSES = 10

# Load model
model = torchunlearn.utils.load_model(model_name=MODEL_NAME, n_classes=N_CLASSES)
rmodel = torchunlearn.RobModel(model, n_classes=N_CLASSES, 
                               normalization_used={'mean': MEAN, 'std': STD})

# Setup data
setup = UnlearnDataSetup(data_name=DATA_NAME, n_classes=N_CLASSES, 
                         mean=MEAN, std=STD)
```

### Forgetting Scenarios

**Random Forgetting** - Forget a random subset of training data:

```python
train_loaders, test_loaders = setup.get_loaders_for_rand(
    batch_size=128, 
    ratio=0.1,           # 10% of data to forget
    stratified=True,     # Maintain class distribution
    seed=42
)
```

**Classwise Forgetting** - Forget all samples from specific classes:

```python
train_loaders, test_loaders = setup.get_loaders_for_classwise(
    batch_size=128, 
    omit_label=1,       # Forget class 1
    train_shuffle_and_transform=True
)
```

### Unlearning Methods



|  **Method**  | **Description** | **Reference** |
|:------------:|-----------------|---------------|
| **Finetune** | Retrain on retain set only | Baseline method |
| **NegGrad** | Negative gradient descent on forget set | [Golatkar et al., 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Golatkar_Eternal_Sunshine_of_the_Spotless_Net_Selective_Forgetting_in_Deep_CVPR_2020_paper.html) |
| **RandomLabel** | Train forget set with random labels | [Golatkar et al., 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Golatkar_Eternal_Sunshine_of_the_Spotless_Net_Selective_Forgetting_in_Deep_CVPR_2020_paper.html) |
| **L1Sparse** | L1 sparsity regularization | [Jia et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a204aa68ab4e970e1ceccfb5b5cdc5e4-Abstract-Conference.html) |
| **UAM** | Unlearning-Aware Minimization | [Kim et al., 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/116406) |

### Non-Training Methods

|  **Method**  | **Description** | **Reference** |
|:------------:|-----------------|---------------|
| **FisherForget** | Fisher information matrix-based | [Golatkar et al., 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Golatkar_Eternal_Sunshine_of_the_Spotless_Net_Selective_Forgetting_in_Deep_CVPR_2020_paper.html) |
| **Influence** | Influence function with Newton's method | [Izzo et al., 2021](https://proceedings.mlr.press/v130/izzo21a.html) |

**Finetune** - Simply retrain on retain set:

```python
from torchunlearn.unlearn.trainers.finetune import Finetune

trainer = Finetune(rmodel)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", 
              n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**Negative Gradient** - Apply negative gradient on forget set:

```python
from torchunlearn.unlearn.trainers.neggrad import NegGrad

trainer = NegGrad(rmodel, retain_lambda=0.5)  # Balance forget/retain
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", 
              n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**Random Label** - Relabel forget set randomly:

```python
from torchunlearn.unlearn.trainers.randomlabel import RandomLabel

trainer = RandomLabel(rmodel, retain_lambda=0.5)
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", 
              n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**L1 Sparse** - Apply L1 regularization:

```python
from torchunlearn.unlearn.trainers.l1sparse import L1Sparse

trainer = L1Sparse(rmodel, gamma=1e-5)  # L1 regularization strength
trainer.setup(optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", 
              n_epochs=5)
trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**UAM** - Apply Unlearning-Aware Minimization:

```python
from torchunlearn.unlearn.trainers.standard import Standard

trainer = Standard(rmodel)
cosine = True # or False
if cosine: 
  schdeuler = "Cosine"
  cosine_total_step = EPOCH*len(merged_loader)
else:
  schdeuler = None
  cosine_total_step = None
trainer.setup(optimizer=f"SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)", 
              scheduler=None, scheduler_type=None,
              minimizer=f"UAM(rho={rho}, cosine_total_step={cosine_total_step}, gamma={gamma})", 
              n_epochs=EPOCH)

trainer.fit(train_loaders=merged_loader, n_epochs=5)
```

**Fisher Forgetting** - Use Fisher information matrix:

```python
from torchunlearn.unlearn.nontrainers.fisherforget import FisherForget

unlearner = FisherForget(rmodel)
unlearner.fit(train_loaders, 
              alphas=[1e-9, 1e-8, 1e-7, 1e-6], 
              repeat=3,
              save_path="./models/fisher")
```

**Influence Function** - Newton-based influence removal:

```python
from torchunlearn.unlearn.nontrainers.influence import Influence

unlearner = Influence(rmodel)
unlearner.fit(train_loaders, 
              alphas=[1e-9, 1e-8, 1e-7, 1e-6], 
              repeat=3,
              save_path="./models/influence")
```

### Evaluation

Track performance during unlearning:

```python
# Setup evaluation loaders
loaders_with_flags = {
    "(R)": train_loaders['Retain'],    # Should maintain high accuracy
    "(F)": train_loaders['Forget'],    # Should decrease accuracy (forgetting)
    "(Te)": test_loaders['Test'],      # Should maintain generalization
}

trainer.record_rob(loaders_with_flags, n_limit=1000)

# Train with automatic evaluation
trainer.fit(
    train_loaders=merged_loader, 
    n_epochs=5,
    save_path="./models/unlearned", 
    save_best={"Clean(R)": "HB", "Clean(F)": "LBO"},  # High retain, Low forget
    save_overwrite=True, 
    record_type="Epoch"
)
```

## :bulb: Understanding Machine Unlearning

**Why Machine Unlearning?**

Traditional approach to remove data influence requires retraining from scratch, which is:
- ⏱️ **Time-consuming** for large models
- 💰 **Expensive** in terms of computation
- 🔄 **Impractical** for frequent requests

Machine unlearning provides efficient alternatives that approximate the retrained model.

**Key Metrics**

- **Retain Accuracy**: Performance on data we want to keep (should stay high)
- **Forget Accuracy**: Performance on data we want to forget (should decrease)
- **Test Accuracy**: Generalization performance (should maintain)
- **Unlearning Gap**: `|Acc. of Retrained Model - Acc. of Unlearned Model|` (smaller is better)

## :book: Example Results

From `demo.ipynb` on CIFAR10 with 10% random forgetting:

```
Epoch   Cost     Clean(R)   Clean(F)   Clean(Te)   
=====================================================
1       0.0913   96.48%     91.02%     91.80%     
2       0.0524   96.39%     68.65%     91.02%     
3       0.0884   95.80%     54.39%     91.02%     
4       0.0525   96.58%     45.02%     90.04%     
5       0.1073   97.36%     33.01%     91.41%     
=====================================================
```

## :link: Related Projects

* **[MAIR](https://github.com/Harry24k/MAIR)**: *Adversarial Training Framework, NeurIPS'23.*
* **[Torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)**: *Adversarial Attack Library.*
* **[RobustBench](https://github.com/RobustBench/robustbench)**: *Adversarially Trained Models & Benchmarks.*

## :memo: Citation

If you use this package in your research, please cite:

```bibtex
@article{kim2025unlearning,
  title={Unlearning-Aware Minimization},
  author={Kim, Hoki and Kim, Keonwoo and Chae, Sungwon and Yoon, Sangwon},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
  volume={39},
  pages={--},
  year={2025}
}
```

## :memo: TODO List
[] Extend this package to support LLM unlearning.
[] Modify the logging/recording modules to ensure compatibility with TensorBoard.
[] ...

## :mortar_board: Getting Help

If you have questions or need help:

1. Check the [demo notebook](demo.ipynb) for examples
2. Review the [demo script](demo.py) for code usage
3. Open an issue on GitHub

## :handshake: Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
