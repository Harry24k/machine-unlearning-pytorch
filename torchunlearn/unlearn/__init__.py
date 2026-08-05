"""torchunlearn.unlearn -- public API for all unlearning methods.

Training-based methods (gradient-descent / fine-tuning)
-------------------------------------------------------
Finetune       -- retain-set fine-tuning baseline
NegGrad        -- negative gradient on forget set (Golatkar et al., CVPR 2020)
RandomLabel    -- random label relabeling (Golatkar et al., CVPR 2020)
L1Sparse       -- L1-regularized retain fine-tuning (Jia et al., NeurIPS 2023)
Standard       -- generic trainer; pair with the UAM minimizer
                  (Kim et al., NeurIPS 2025)
SCRUB          -- KL-maximization/minimization (Kurmanji et al., NeurIPS 2023)
BadTeacher     -- competent/bad-teacher distillation (Chundawat et al., AAAI 2023)
BoundaryShrink -- nearest-class re-targeting (Chen et al., CVPR 2023)
SalUn          -- saliency-masked random-label (Fan et al., ICLR 2024)
ARU            -- adversarial retain-free unlearning (Yoon et al., 2026)
AMUN           -- fine-tune on nearest adversarial examples
                  (Ebrahimpour-Boroojeny et al., ICML 2025)
SFRon          -- saliency forgetting in a remain-preserving manifold
                  (Huang et al., NeurIPS 2024 Spotlight)
MUMis          -- retain-free input-sensitivity suppression
                  (Cheng et al., ICLR 2026)
RFE            -- two-phase unlearning under retain-forget entanglement
                  (Cheng et al., ICLR 2026)

Non-training methods (closed-form / weight-edit / architectural)
----------------------------------------------------------------
FisherForget   -- Fisher-information perturbation (Golatkar et al., CVPR 2020)
Influence      -- Newton/influence function removal (Izzo et al., AISTATS 2021)
NegMerge       -- sign-consensual weight merging (Kim, Han & Choe, ICML 2025)
SISATrainingProtocol / SISAUnlearner / SISAAggregator
               -- sharded-sliced training protocol (Bourtoule et al., S&P 2021)
rem_unlearn / expand_model_for_rem / drop_rem_expansion
               -- redirection for erasing memory

Engine
------
Trainer        -- base training loop
RecordManager  -- training-log / checkpoint manager
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
from .trainers.aru import ARU
from .trainers.amun import AMUN
from .trainers.sfron import SFRon
from .trainers.mumis import MUMis
from .trainers.rfe import RFE

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
    "SCRUB", "BadTeacher", "BoundaryShrink", "SalUn", "ARU",
    "AMUN", "SFRon", "MUMis", "RFE",
    # non-training
    "FisherForget", "Influence", "NegMerge",
    "SISATrainingProtocol", "SISAUnlearner", "SISAAggregator",
    "aggregate_predictions",
    "rem_unlearn", "expand_model_for_rem", "drop_rem_expansion",
    "set_rem_trainable", "make_mask_assignment", "IndexedDataset",
    "REMConfig", "accuracy_base_only", "rem_forward", "REMLinear", "REMConv2d",
    # engine
    "Trainer", "RecordManager",
]
