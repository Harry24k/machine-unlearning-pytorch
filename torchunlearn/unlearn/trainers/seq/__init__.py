"""Sequence (text / image+text) unlearning trainers.  Same loss formulas as ``trainers/llm_pii.py``; the
batch contract is :class:`~torchunlearn.unlearn.seq_data.SeqCollator`'s, so one implementation serves plain LLMs
and vision-language models."""
from collections import OrderedDict

from .base import SeqUnlearner
from .finetune import SeqFinetune
from .gradascent import GradAscent
from .graddiff import GradDiff
from .npo import NPO
from .simnpo import SimNPO
from .dpo import DPO
from .altpo import AltPO
from .wga import WGA
from .satimp import SatImp
from .undial import UNDIAL
from .pdu import PDU
from .flat import FLAT
from .rmu import RMU
from .po import PO
from .klmin import KLMin
from .siu import SIU
from .grpo import GRPOUnlearner
from .asru import ASRU, ASRUSteer, is_refusal, REFUSAL_PATTERNS
from .lumoe import LUMoE, NameMatchRouter
from .safetymirage import SafetyMirageNPO, SafetyMirageRMU, build_vlguard_sets

SEQ_TRAINERS = OrderedDict([
    ("GradAscent", GradAscent), ("GradDiff", GradDiff), ("NPO", NPO), ("SimNPO", SimNPO), ("DPO", DPO),
    ("AltPO", AltPO), ("WGA", WGA), ("SatImp", SatImp), ("UNDIAL", UNDIAL), ("PDU", PDU), ("FLAT", FLAT), ("RMU", RMU),
    # multimodal-paper methods (ICLR / ICML / NeurIPS 2024-2026) and benchmark baselines
    ("PO", PO), ("KLMin", KLMin), ("SIU", SIU), ("ASRU", ASRU), ("SafetyMirageNPO", SafetyMirageNPO), ("SafetyMirageRMU", SafetyMirageRMU),
])

__all__ = ["SeqUnlearner", "SeqFinetune", "SEQ_TRAINERS", "GRPOUnlearner", "ASRUSteer", "LUMoE", "NameMatchRouter",
           "build_vlguard_sets", "is_refusal", "REFUSAL_PATTERNS"] + list(SEQ_TRAINERS)
