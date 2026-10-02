"""Dataset recipes: benchmark-specific loaders that turn each benchmark into SeqSample lists (+ meta for metrics).

    fiubench      FIUBench (ICLR 2025)                     gray311/FIUBench
    mllmu_bench   MLLMU-Bench / UMU-Bench (NeurIPS 2025)   MLLMMU/MLLMU-Bench, chengyewang/UMU-bench
    clear         CLEAR (TOFU-style multimodal)            therem/CLEAR
    mlubench      MLUBench (ICML 2026, lifelong)           lihe-maxsize/Lifelong_Unlearning_main (local JSON)
    mmubench      MMUBench (SIU, NeurIPS 2024) + SIU multifaceted fine-tuning data builder
    vlguard       VLGuard -> Safety Mirage forget / retain (trainers.seq.safetymirage.build_vlguard_sets)
    domains       domain-labelled image folders for ADU (NeurIPS 2025)
"""
from . import clear, domains, fiubench, mllmu_bench, mlubench, mmubench
from ..trainers.seq.safetymirage import build_vlguard_sets

__all__ = ["fiubench", "mllmu_bench", "clear", "mlubench", "mmubench", "domains", "build_vlguard_sets"]
