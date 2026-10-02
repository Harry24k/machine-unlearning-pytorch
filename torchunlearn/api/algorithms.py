"""Central registration of every unlearning algorithm.
Edit ONLY this file when adding/changing hparams or adding new algorithms."""

from .hparam_spec import HParam
from .registry import register_unlearner

# import every algorithm class
from ..unlearn.trainers.finetune import Finetune
from ..unlearn.trainers.neggrad import NegGrad
from ..unlearn.trainers.randomlabel import RandomLabel
from ..unlearn.trainers.l1sparse import L1Sparse
from ..unlearn.trainers.standard import Standard
from ..unlearn.trainers.scrub import SCRUB
from ..unlearn.trainers.badteacher import BadTeacher
from ..unlearn.trainers.boundaryshrink import BoundaryShrink
from ..unlearn.trainers.salun import SalUn
from ..unlearn.trainers.advtrainer import AdvTrainer
from ..unlearn.trainers.aru import ARU
from ..unlearn.nontrainers.fisherforget import FisherForget
from ..unlearn.nontrainers.influence import Influence
from ..unlearn.nontrainers.negmerge import NegMerge
# from ..unlearn.nontrainers.sisa import SISAUnlearner as _SISAUnlearnerImpl
# from ..unlearn.nontrainers.rem import REMConfig as _REMConfigImpl, rem_unlearn_and_drop as _rem_unlearn_and_drop


# class SISAUnlearner(_SISAUnlearnerImpl):
#     """Registry-facing wrapper. `.fit(...)` runs the SISA delete protocol."""

#     def fit(self, *args, **kwargs):
#         # Delegate to the SISA deletion routine the impl already exposes.
#         if hasattr(_SISAUnlearnerImpl, "unlearn"):
#             return _SISAUnlearnerImpl.unlearn(self, *args, **kwargs)
#         if hasattr(_SISAUnlearnerImpl, "delete"):
#             return _SISAUnlearnerImpl.delete(self, *args, **kwargs)
#         raise NotImplementedError(
#             "SISAUnlearner has no run entry point; expected `unlearn` or `delete`."
#         )


# class REMConfig(_REMConfigImpl):
#     """Registry-facing wrapper around the REM functional API."""

#     def fit(self, pretrained_model, train_loader, forget_loader,
#             dataset_size, forget_indices, **kwargs):
#         return _rem_unlearn_and_drop(
#             pretrained_model=pretrained_model,
#             train_loader=train_loader,
#             forget_loader=forget_loader,
#             dataset_size=dataset_size,
#             forget_indices=forget_indices,
#             cfg=self,
#             **kwargs,
#         )


# ----------------- TRAINERS -----------------
register_unlearner(
    "Finetune", kind="trainer", cls=Finetune,
    summary="Finetune on retain set only.",
    hparams=[
        HParam("optimizer", str, default="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)",
               category="setup"),
        HParam("n_epochs", int, default=5, range=(1, 200), category="setup"),
    ],
)

register_unlearner(
    "NegGrad", kind="trainer", cls=NegGrad,
    summary="Negative gradient ascent on the forget set.",
    hparams=[
        HParam("optimizer", str, default="SGD(lr=1e-3, momentum=0.9)", category="setup"),
        HParam("n_epochs", int, default=5, category="setup"),
        HParam("lambda_forget", float, default=1.0, range=(0.0, 10.0),
               description="Weight on the negative-gradient forget term."),
    ],
)

register_unlearner(
    "SCRUB", kind="trainer", cls=SCRUB,
    summary="Teacher-student distillation: KD on retain, ascent on forget.",
    reference="Kurmanji et al., NeurIPS 2023",
    hparams=[
        HParam("optimizer", str, default="SGD(lr=5e-3, momentum=0.9)", category="setup"),
        HParam("n_epochs", int, default=5, category="setup"),
        HParam("T", float, default=4.0, range=(0.1, 20.0),
               description="Distillation temperature."),
        HParam("alpha", float, default=1.0, description="Weight on retain KD loss."),
        HParam("gamma", float, default=0.99, range=(0.0, 1.0),
               description="Decay on forget-loss contribution."),
        HParam("msteps", int, default=3, range=(1, 50)),
    ],
)

register_unlearner(
    "BadTeacher", kind="trainer", cls=BadTeacher,
    summary="Competent-teacher / bad-teacher KD: match BT on forget, CT on retain.",
    reference="Chundawat et al., AAAI 2023",
    hparams=[
        HParam("optimizer", str, default="SGD(lr=1e-3, momentum=0.9)", category="setup"),
        HParam("n_epochs", int, default=5, category="setup"),
        HParam("retain_lambda", float, default=1.0, range=(0.0, 10.0),
               description="Weight on retain KL(CT||student)."),
        HParam("forget_lambda", float, default=1.0, range=(0.0, 10.0),
               description="Weight on forget KL(BT||student)."),
        HParam("temperature", float, default=1.0, range=(0.1, 20.0),
               description="Softmax temperature for KL distillation."),
    ],
)

register_unlearner(
    "BoundaryShrink", kind="trainer", cls=BoundaryShrink,
    summary="Adversarial nearest-class relabel; shift decision boundary on forget set.",
    reference="Chen et al., CVPR 2023",
    hparams=[
        HParam("optimizer", str, default="SGD(lr=1e-3, momentum=0.9)", category="setup"),
        HParam("n_epochs", int, default=5, category="setup"),
        HParam("omit_label", int, default=None,
               description="Class to forget (set for class-wise unlearning)."),
        HParam("eps", float, default=0.1, range=(0.0, 1.0),
               description="FGSM noise bound for cross-sample generation."),
        HParam("forget_lambda", float, default=1.0, range=(0.0, 10.0),
               description="Weight on Boundary Shrink loss."),
        HParam("retain_lambda", float, default=0.0, range=(0.0, 10.0),
               description="Weight on optional retain CE repair (0.0 = paper-faithful)."),
        HParam("force_incorrect", bool, default=True,
               description="Mask true class when picking the nearest-incorrect label."),
    ],
)

register_unlearner(
    "SalUn", kind="trainer", cls=SalUn,
    summary="Saliency-masked random-label unlearning; updates only salient weights.",
    reference="Fan et al., ICLR 2024",
    hparams=[
        HParam("optimizer", str, default="SGD(lr=1e-3, momentum=0.9)", category="setup"),
        HParam("n_epochs", int, default=5, category="setup"),
        HParam("mask_ratio", float, default=0.5, range=(0.0, 1.0),
               description="Top-|grad| fraction of weights kept trainable."),
        HParam("retain_lambda", float, default=1.0, range=(0.0, 10.0),
               description="Weight on CE loss over retain set."),
        HParam("forget_lambda", float, default=1.0, range=(0.0, 10.0),
               description="Weight on CE loss with random labels over forget set."),
    ],
)

# ... add the rest the same way: RandomLabel, L1Sparse, Standard, BadTeacher,
# BoundaryShrink, SalUn, AdvTrainer, ARU

# ----------------- NON-TRAINERS -----------------
register_unlearner(
    "FisherForget", kind="nontrainer", cls=FisherForget,
    summary="Closed-form Fisher Information unlearning via Hessian-scaled noise.",
    reference="Golatkar et al., CVPR 2020",
    hparams=[
        HParam("alphas", list, default=[1e-9, 1e-8, 1e-7, 1e-6],
               description="Noise scale candidates; best per-parameter auto-selected."),
        HParam("repeat", int, default=1, range=(1, 100)),
        HParam("omit_label", int, default=None,
               description="Class label to exclude from Hessian (None = use all)."),
    ],
)

register_unlearner(
    "Influence", kind="nontrainer", cls=Influence,
    summary="Influence-function-based parameter update to remove forget-set effect.",
    hparams=[
        HParam("damping", float, default=0.01, range=(0.0, 1.0)),
        HParam("scale", float, default=1.0),
        HParam("n_iters", int, default=100, range=(1, 10000)),
    ],
)

# register_unlearner(
#     "SISA", kind="nontrainer", cls=SISAUnlearner,
#     summary="Sharded, Isolated, Sliced, Aggregated training; targeted shard retraining on delete.",
#     reference="Bourtoule et al., IEEE S&P 2021",
#     hparams=[
#         HParam("batch_size", int, default=128, range=(1, 4096), category="setup"),
#         HParam("shuffle_batches", bool, default=True,
#                description="Shuffle mini-batches when retraining affected shards."),
#     ],
# )

# register_unlearner(
#     "REM", kind="nontrainer", cls=REMConfig,
#     summary="Redirection for Erasing Memory: removable theta2 branch + NPO repair, then drop.",
#     reference="Schoepf et al., ICLR 2026",
#     hparams=[
#         HParam("lr", float, default=5e-4, range=(1e-6, 1.0), category="setup"),
#         HParam("weight_decay", float, default=1e-4, range=(0.0, 1.0), category="setup"),
#         HParam("max_epochs", int, default=5, range=(1, 200), category="setup"),
#         HParam("beta", float, default=1.0, range=(0.0, 100.0),
#                description="NPO inverse-temperature on the reference-model log ratio."),
#         HParam("gamma", float, default=0.2, range=(0.0, 1.0),
#                description="Step-2 removal threshold on per-sample NPO value."),
#         HParam("lambda_remove_step3", float, default=1.0, range=(0.0, 10.0),
#                description="Weight on step-3 base-only removal loss on Df."),
#         HParam("max_removal_steps_per_epoch", int, default=20, range=(1, 1000),
#                description="Cap on step-2 removal iterations per epoch."),
#         HParam("active_ratio", float, default=0.2, range=(0.0, 1.0),
#                description="Fraction of theta2 mask channels active during repair."),
#     ],
# )


# ================= LLM / PII span-level unlearners (scope-aware T/S/C/G benchmark) =================
# see torchunlearn/unlearn/trainers/llm_pii.py and torchunlearn/unlearn/nontrainers/revs.py
from ..unlearn.trainers.llm_pii import (
    GradAscent as LLMGradAscent, GradDiff as LLMGradDiff, NPO as LLMNPO, SimNPO as LLMSimNPO,
    DPO as LLMDPO, AltPO as LLMAltPO, RMU as LLMRMU, WGA as LLMWGA, SatImp as LLMSatImp, CEU as LLMCEU,
    UNDIAL as LLMUNDIAL, PDU as LLMPDU, JensUn as LLMJensUn, FLAT as LLMFLAT,
)
from ..unlearn.nontrainers.revs import REVS as LLMREVS

_LLM_COMMON = [
    HParam("optimizer", str, default="AdamW(lr=1e-5)", category="setup"),
    HParam("n_epochs", int, default=5, category="setup"),
    HParam("clip_grad_norm", float, default=1.0, category="setup"),
    HParam("gamma", float, default=1.0, description="forget-term weight"),
    HParam("alpha", float, default=1.0, description="retain-term weight"),
    HParam("retain_loss_type", str, default="NLL", choices=["NLL", "KL", "JSD", "none", "EMBED_DIFF"]),
    HParam("retain_loss_on", str, default="span", choices=["span", "window"]),
    HParam("ref_mode", str, default="cache", choices=["cache", "model"]),
    HParam("trainable", str, default=None, description="regex over parameter names (None = all)"),
    HParam("grad_ckpt", bool, default=False),
    HParam("stop_lp", float, default=None, description="stop when T log-prob/token <= stop_lp"),
    HParam("eval_every", int, default=0),
]


def _llm(name, cls, summary, reference, extra=(), kind="trainer"):
    register_unlearner(name, kind=kind, cls=cls, summary=summary, reference=reference, modality="seq",
                       hparams=(list(extra) + _LLM_COMMON) if kind == "trainer" else list(extra))


_llm("LLM-GradAscent", LLMGradAscent, "[classic] gradient ascent on span tokens (loss = -CE).",
     "Jang et al., 2023 (open-unlearning GradAscent)")
_llm("LLM-GradDiff", LLMGradDiff, "[classic] -CE(forget) + alpha * retain (NLL | KL vs reference).",
     "Liu et al., 2022 (open-unlearning GradDiff)")
_llm("LLM-NPO", LLMNPO, "[classic] Negative Preference Optimization on seq-level log-ratio vs reference.",
     "Zhang et al., 2024 (open-unlearning NPO)", [HParam("beta", float, default=0.1)])
_llm("LLM-SimNPO", LLMSimNPO, "[classic] reference-free NPO on length-normalised NLL.",
     "Fan et al., 2024 (open-unlearning SimNPO)",
     [HParam("beta", float, default=4.5), HParam("delta", float, default=0.0)])
_llm("LLM-DPO", LLMDPO, "[classic] DPO: alternate answer ('[REDACTED]') preferred over the PII span.",
     "Rafailov et al., 2023 / Maini et al., 2024 IdkDPO (open-unlearning DPO)", [HParam("beta", float, default=0.1)])
_llm("LLM-AltPO", LLMAltPO, "[classic] DPO with M self-generated plausible alternates of the PII span as the win answer.",
     "Mekala et al., COLING 2025 (arXiv 2409.13474; open-unlearning community/methods/AltPO)",
     [HParam("beta", float, default=0.1)])
_llm("LLM-RMU", LLMRMU, "[classic] representation misdirection at layer L; trains down_proj of L-2..L.",
     "Li et al., 2024 (open-unlearning RMU)",
     [HParam("layer_id", int, default=7), HParam("steering_coeff", float, default=20.0),
      HParam("module_regex", str, default=None)])
_llm("LLM-WGA", LLMWGA, "[SOTA 2025] weighted GA, w = p^beta.", "Wang et al., 2025 (open-unlearning WGA)",
     [HParam("beta", float, default=1.0)])
_llm("LLM-SatImp", LLMSatImp, "[SOTA 2025] saturation x importance weighted GA.", "Wang et al., 2025 (open-unlearning SatImp)",
     [HParam("beta1", float, default=5.0), HParam("beta2", float, default=1.0)])
_llm("LLM-CEU", LLMCEU, "[SOTA 2025] cross-entropy unlearning towards the true-token-excluded distribution.",
     "Wang, 2025 (open-unlearning CEU)", [HParam("ignore_first_n", int, default=1)])
_llm("LLM-UNDIAL", LLMUNDIAL, "[SOTA 2025] self-distillation with penalised teacher logit on the true token.",
     "Dong et al., 2025 (open-unlearning UNDIAL)", [HParam("beta", float, default=10.0)])
_llm("LLM-PDU", LLMPDU, "[SOTA 2025] primal-dual unlearning, logit-margin forget loss.", "Entesari et al., NeurIPS 2025 (arXiv 2506.05314; open-unlearning PDU)",
     [HParam("primal_dual", bool, default=False), HParam("dual_step_size", float, default=1.0),
      HParam("retain_loss_eps", float, default=0.0),
      HParam("dual_update_upon", str, default="step", choices=["step", "epoch"]),
      HParam("dual_warmup_epochs", int, default=0)])
_llm("LLM-JensUn", LLMJensUn, "[SOTA 2025-09] Jensen-Shannon forget/retain with fixed target tokens.",
     "JensUn (2025)", [HParam("target_text", str, default="No way")])
_llm("LLM-FLAT", LLMFLAT, "[SOTA 2025] f-divergence contrastive loss between [REDACTED] and PII answer probabilities.",
     "Wang et al., ICLR 2025 (FLAT)", [HParam("div", str, default="Total-Variation")])
_llm("LLM-REVS", LLMREVS, "[PII-specific] rank editing in the vocabulary space (non-gradient neuron edit).",
     "Ashuach et al., ACL 2025 (REVS)", kind="nontrainer",
     extra=[HParam("n_neurons", int, default=30), HParam("max_tokens", int, default=2),
            HParam("token_method", str, default="rarest", choices=["rarest", "first", "frequent", "random"]),
            HParam("residual_bottom_rank_margin", int, default=10000), HParam("residual_top_rank_margin", int, default=20000),
            HParam("mlp_bottom_rank_margin", int, default=10000), HParam("mlp_top_rank_margin", int, default=10000),
            HParam("neuron_bottom_rank_margin", int, default=90000), HParam("neuron_top_rank_margin", int, default=100000),
            HParam("max_iter_mlp_rank", int, default=100), HParam("max_iter_neuron_rank", int, default=100),
            HParam("act_filter", str, default="top_100"), HParam("max_prompt_tokens", int, default=1024)])


# ================= Sequence (text LLM / image+text LVLM) unlearners: registry names "MM-*" =================
# see torchunlearn/unlearn/trainers/seq/ and docs/multimodal_unlearning.md
from ..unlearn.trainers.seq import (
    SeqFinetune, GradAscent as SeqGradAscent, GradDiff as SeqGradDiff, NPO as SeqNPO, SimNPO as SeqSimNPO,
    DPO as SeqDPO, AltPO as SeqAltPO, RMU as SeqRMU, WGA as SeqWGA, SatImp as SeqSatImp, UNDIAL as SeqUNDIAL,
    PDU as SeqPDU, FLAT as SeqFLAT,
)

_SEQ_COMMON = [
    HParam("optimizer", str, default="AdamW(lr=1e-5)", category="setup"),
    HParam("n_epochs", int, default=5, category="setup"),
    HParam("clip_grad_norm", float, default=1.0, category="setup"),
    HParam("gamma", float, default=1.0, description="forget-term weight"),
    HParam("alpha", float, default=1.0, description="retain-term weight"),
    HParam("retain_loss_type", str, default="NLL", choices=["NLL", "KL", "JSD", "none", "EMBED_DIFF"]),
    HParam("retain_loss_on", str, default="answer", choices=["answer", "full"]),
    HParam("ref_mode", str, default="cache", choices=["cache", "model"]),
    HParam("trainable", str, default=None,
           description="SeqRobModel spec: all | lm | vision | projector | lm+projector | lora:r=16,target=lm | re:<regex> (None = keep)"),
    HParam("grad_ckpt", bool, default=False),
    HParam("stop_lp", float, default=None, description="stop when stop_role log-prob/token <= stop_lp"),
    HParam("stop_role", str, default="Forget"),
    HParam("eval_every", int, default=0),
    HParam("traj_every", int, default=1, description="trajectory eval every k epochs (<=0: final eval only)"),
]


def _seq(name, cls, summary, reference, extra=()):
    register_unlearner(name, kind="trainer", cls=cls, summary=summary, reference=reference, modality="seq",
                       hparams=list(extra) + _SEQ_COMMON)


_seq("MM-Finetune", SeqFinetune, "[baseline] NLL fine-tuning: build the model to unlearn from, retrain on retain, or retain-only FT.",
     "-")
_seq("MM-GradAscent", SeqGradAscent, "[classic] gradient ascent on answer tokens (loss = -CE).",
     "Jang et al., 2023 (open-unlearning GradAscent)")
_seq("MM-GradDiff", SeqGradDiff, "[classic] -CE(forget) + alpha * retain (NLL | KL vs reference).",
     "Liu et al., 2022 (open-unlearning GradDiff)")
_seq("MM-NPO", SeqNPO, "[classic] Negative Preference Optimization on the seq-level log-ratio vs reference.",
     "Zhang et al., 2024 (open-unlearning NPO)", [HParam("beta", float, default=0.1)])
_seq("MM-SimNPO", SeqSimNPO, "[classic] reference-free NPO on length-normalised NLL.",
     "Fan et al., 2024 (open-unlearning SimNPO)", [HParam("beta", float, default=4.5), HParam("delta", float, default=0.0)])
_seq("MM-DPO", SeqDPO, "[classic] DPO: alternate answer (alt_text / alt_answers) preferred over the forget answer.",
     "Rafailov et al., 2023 / Maini et al., 2024 IdkDPO (open-unlearning DPO)", [HParam("beta", float, default=0.1)])
_seq("MM-AltPO", SeqAltPO, "[classic] DPO with M plausible alternates per forget sample as the win answer.",
     "Mekala et al., COLING 2025 (arXiv 2409.13474)", [HParam("beta", float, default=0.1)])
_seq("MM-RMU", SeqRMU, "[classic] representation misdirection at LM layer L; trains the MLP out-proj of L-2..L.",
     "Li et al., 2024 (open-unlearning RMU)",
     [HParam("layer_id", int, default=7), HParam("steering_coeff", float, default=20.0),
      HParam("module_regex", str, default=None)])
_seq("MM-WGA", SeqWGA, "[SOTA 2025] weighted GA, w = p^beta.", "Wang et al., 2025 (open-unlearning WGA)",
     [HParam("beta", float, default=1.0)])
_seq("MM-SatImp", SeqSatImp, "[SOTA 2025] saturation x importance weighted GA.", "Wang et al., 2025 (open-unlearning SatImp)",
     [HParam("beta1", float, default=5.0), HParam("beta2", float, default=1.0)])
_seq("MM-UNDIAL", SeqUNDIAL, "[SOTA 2025] self-distillation with a penalised teacher logit on the true token.",
     "Dong et al., 2025 (open-unlearning UNDIAL)", [HParam("beta", float, default=10.0)])
_seq("MM-PDU", SeqPDU, "[SOTA 2025] primal-dual unlearning, logit-margin forget loss.",
     "Entesari et al., NeurIPS 2025 (arXiv 2506.05314; open-unlearning PDU)",
     [HParam("primal_dual", bool, default=False), HParam("dual_step_size", float, default=1.0),
      HParam("retain_loss_eps", float, default=0.0),
      HParam("dual_update_upon", str, default="step", choices=["step", "epoch"]),
      HParam("dual_warmup_epochs", int, default=0)])
_seq("MM-FLAT", SeqFLAT, "[SOTA 2025] f-divergence contrastive loss between alternate and forget answer probabilities.",
     "Wang et al., ICLR 2025 (FLAT)", [HParam("div", str, default="Total-Variation")])


# ================= Multimodal (VLM) unlearning papers at ICLR / ICML / NeurIPS 2024-2026 =================
# see docs/multimodal_papers.md for the paper table and fidelity notes
from ..unlearn.trainers.seq import (PO as SeqPO, KLMin as SeqKLMin, SIU as SeqSIU, ASRU as SeqASRU, ASRUSteer as SeqASRUSteer,
                                    SafetyMirageNPO as SeqSMNPO, SafetyMirageRMU as SeqSMRMU)
from ..unlearn.nontrainers.slug import SLUG as ClipSLUG

_seq("MM-PO", SeqPO, "[baseline] Preference Optimisation: refusal answer on forget + NLL on retain (FIUBench / MLLMU-Bench / UMU-Bench / LUMoE expert).",
     "Ma et al., ICLR 2025 (FIUBench baseline)")
_seq("MM-KLMin", SeqKLMin, "[baseline] -CE(forget) + KL(ref || model) on retain (FIUBench / MLLMU-Bench / MLUBench baseline).",
     "Ma et al., ICLR 2025 (FIUBench baseline)")
_seq("MM-SIU", SeqSIU, "[NeurIPS 2024] Single Image Unlearning: CE on multifaceted data + dual-masked KL to the original model.",
     "Li et al., NeurIPS 2024 (arXiv 2405.12523)",
     [HParam("concept_names", list, default=None, description="names of the concept to forget (vocabulary mask K_V)"),
      HParam("ce_weight", float, default=0.9), HParam("dmk_weight", float, default=0.75)])
_seq("MM-ASRU", SeqASRU, "[ICML 2026] stage 2 of ASRU: GRPO with the rule-based refusal / boundary reward (run MM-ASRUSteer first).",
     "arXiv 2605.15687 (ICML 2026)",
     [HParam("group_size", int, default=4), HParam("max_new_tokens", int, default=32), HParam("temperature", float, default=1.0),
      HParam("kl_coef", float, default=0.1), HParam("norm_std", bool, default=True)])
register_unlearner("MM-ASRUSteer", kind="nontrainer", cls=SeqASRUSteer, modality="seq",
                   summary="[ICML 2026] stage 1 of ASRU: closed-form steering of one MLP down-projection along the knowledge-absence direction.",
                   reference="arXiv 2605.15687 (ICML 2026)",
                   hparams=[HParam("layer_id", int, default=17), HParam("lam", float, default=3.0), HParam("gamma", float, default=1e-2),
                            HParam("positions", str, default="answer", choices=["answer", "all"])])
_seq("MM-SafetyMirage-NPO", SeqSMNPO, "[ICLR 2026] Safety Mirage: NPO on unsafe multimodal queries + NLL on safe data (VLGuard recipe).",
     "Chen et al., ICLR 2026 (arXiv 2503.11832)", [HParam("beta", float, default=0.1)])
_seq("MM-SafetyMirage-RMU", SeqSMRMU, "[ICLR 2026] Safety Mirage: RMU on unsafe query + harmful response, EMBED_DIFF retain on safe data.",
     "Chen et al., ICLR 2026 (arXiv 2503.11832)",
     [HParam("layer_id", int, default=7), HParam("steering_coeff", float, default=20.0)])
register_unlearner("CLIP-SLUG", kind="nontrainer", cls=ClipSLUG, modality="clip",
                   summary="[ICML 2025] single-layer, single-gradient unlearning for CLIP (transplantable into LLaVA's vision tower).",
                   reference="Cai et al., ICML 2025 (arXiv 2407.11867)",
                   hparams=[HParam("forget_loss", str, default="cosine", choices=["cosine", "contrastive"]),
                            HParam("layer_regex", str, default=r"vision_model\.encoder\.layers\.\d+\.(self_attn|mlp)\..*weight")])
