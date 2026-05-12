"""torchunlearn.unlearn -- public API for all unlearning methods.

Training-based methods (gradient-descent / fine-tuning)
-------------------------------------------------------
Finetune      -- retain-set fine-tuning baseline
NegGrad       -- negative gradient on forget set (Golatkar et al., 2020)
RandomLabel   -- random label relabeling (Golatkar et al., 2020)
L1Sparse      -- L1-regularized retain fine-tuning (Jia et al., 2023)
Standard      -- generic trainer used for UAM (Kim et al., 2025)
SCRUB         -- KL-maximization/minimization (Kurmanji et al., NeurIPS 2023)
BadTeacher    -- competent/bad-teacher distillation (Chundawat et al., AAAI 2023)
BoundaryShrink -- nearest-class re-targeting (Chen et al., CVPR 2023)
SalUn         -- saliency-masked random-label (Fan et al., ICLR 2024)

Non-training methods (closed-form / weight-edit / architectural)
----------------------------------------------------------------
FisherForget  -- Fisher-information perturbation (Golatkar et al., 2020)
Influence     -- Newton/influence function removal (Izzo et al., 2021)
NegMerge      -- consensual weight negation (Kim et al., ICML 2025)
SISATrainingProtocol / SISAUnlearner / SISAAggregator
              -- sharded-sliced training protocol (Bourtoule et al., S&P 2021)
              rem_unlearn / expand_model_for_rem / drop_rem_expansion
                            -- redirection for erasing memory (Google DeepMind)

                            Engine
                            ------
                            Trainer       -- base training loop
                            RecordManager -- training-log / checkpoint manager
                            """

# --- Training-based ---
from .trainers.finetune import Finetune
from .trainers.neggrad import NegGrad
from .trainers.randomlabel import RandomLabel
from .trainers.l1sparse import L1Sparse
from .trainers.standard import Standard
from .trainers.scrub import SCRUB
from .trainers.badteacher import BadTeacher
from .trainers.boundaryshrink import BoundaryShrink
from .trainers.salun import SalUn

# --- Non-training ---
from .nontrainers.fisherforget import FisherForget
from .nontrainers.influence import Influence
from .nontrainers.negmerge import NegMerge
from .nontrainers.sisa import (
    SISATrainingProtocol,
    SISAUnlearner,
    SISAAggregator,
    aggregate_predictions,
)
from .nontrainers.rem import (
    rem_unlearn,
    expand_model_for_rem,
    drop_rem_expansion,
    set_rem_trainable,
    make_mask_assignment,
    IndexedDataset,
    REMConfig,
    accuracy_base_only,
    rem_forward,
    REMLinear,
    REMConv2d,
)

# --- Engine & utilities ---
from .trainer import Trainer
from .rm import RecordManager

__all__ = [
      # training-based
    "Finetune", "NegGrad", "RandomLabel", "L1Sparse", "Standard",
      "SCRUB", "BadTeacher", "BoundaryShrink", "SalUn",
      # non-training
      "FisherForget", "Influence", "NegMerge",
      "SISATrainingProtocol", "SISAUnlearner", "SISAAggregator", "aggregate_predictions",
      "rem_unlearn", "expand_model_for_rem", "drop_rem_expansion",
      "set_rem_trainable", "make_mask_assignment", "IndexedDataset",
      "REMConfig", "accuracy_base_only", "rem_forward", "REMLinear", "REMConv2d",
      # engine
      "Trainer", "RecordManager",
]
