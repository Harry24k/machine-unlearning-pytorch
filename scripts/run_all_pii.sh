#!/bin/bash
# Run every LLM/PII unlearning method with its recommended parameters and T/S/C/G evaluation.
#
#   scripts/run_all_pii.sh pilot                                   # Enron pilot (Qwen2-1.5B M_T), base env, fast
#   scripts/run_all_pii.sh llama31_8b  --scope subject --targets "N:allen-p/_sent_mail/586.#jim murnan"
#   scripts/run_all_pii.sh olmo2_7b    --scope fact    --targets Fd142db50f309 F29a0a7c75db2
#   scripts/run_all_pii.sh qwen35_9b   --scope subject_relation --targets "N:allen-p/_sent_mail/586.#jim murnan:EMAIL"   (overlay env)
#
#   METHODS="NPO RMU REVS" scripts/run_all_pii.sh pilot            # subset
#   GPU=0 EPOCHS=5 LR=1e-5 BS=4 OUT=out/pii_unlearn/<key> …        # env overrides
#
# Per-method parameters (defaults of the reference implementations; see torchunlearn/api/algorithms.py):
#   GradAscent  lr 1e-5                          | WGA     beta=1
#   GradDiff    gamma=1 alpha=1 retain NLL       | SatImp  beta1=5 beta2=1 gamma=0.1 alpha=1
#   NPO         beta=0.1 alpha=1                 | CEU     ignore_first_n=1 (no retain)
#   SimNPO      beta=4.5 delta=0 gamma=0.125     | UNDIAL  beta=10 alpha=0
#   DPO         beta=0.1, alt "[REDACTED]"       | PDU     gamma=1 alpha=1 (primal_dual off)
#   RMU         layer_id=7 steering_coeff=20 lr 5e-5, trains layers 5-7 down_proj
#   JensUn      target "No way", retain JSD alpha=1 | FLAT  Total-Variation, alt "[REDACTED]"
#   REVS        n_neurons=30 max_tokens=2 rarest, margins res 10000/20000 mlp 10000/10000 neuron 90000/100000
set -uo pipefail
cd "/home1/irteam/_[chaewon]/_[torchunlearn]"
KEY=${1:?key: pilot | olmo2_7b | llama31_8b | qwen35_9b}; shift
PII="/home1/irteam/_[chaewon]/_[26SS]PII"
GPU=${GPU:-1}; EPOCHS=${EPOCHS:-5}; LR=${LR:-1e-5}; BS=${BS:-4}
METHODS=${METHODS:-"GradAscent GradDiff NPO SimNPO DPO RMU WGA SatImp CEU UNDIAL PDU JensUn FLAT REVS"}
OUT=${OUT:-out/pii_unlearn/$KEY}
STOP=${STOP_LP:+--stop-lp $STOP_LP --eval-every ${EVAL_EVERY:-10}}
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

PY=python
if [ "$KEY" = "pilot" ]; then
  DATA="--pilot"
else
  DATA="--model $PII/out/ft_runs/$KEY/epoch2 --facts ${FACTS:-$PII/out/v3.3/facts.jsonl} --docs ${DOCS:-$PII/out/v3.3/docs.jsonl} $*"
fi
if [ "$KEY" = "qwen35_9b" ]; then   # overlay env + fla/tilelang nvcc vars (ft/run_ft.sh qwen35_9b block)
  PY=/home1/irteam/envs/gemma4-overlay/bin/python
  export PATH=/home1/irteam/envs/sm-base/shim:$PATH CC=/opt/conda/bin/x86_64-conda-linux-gnu-cc CXX=/opt/conda/bin/x86_64-conda-linux-gnu-c++
  BASEINC=/opt/conda/lib/python3.11/site-packages/nvidia/cu13/include
  export CUDA_HOME=/home1/irteam/envs/gemma4-overlay/lib/python3.11/site-packages/nvidia/cu13
  export NVCC_PREPEND_FLAGS="-I$BASEINC -I$BASEINC/cccl"; unset NVCC_APPEND_FLAGS || true
  export PATH=$CUDA_HOME/bin:$PATH
fi
mkdir -p "$OUT"; LOG="$OUT/run_all_$(date +%Y%m%d_%H%M%S).log"
run() {  # method [extra args...]
  local m=$1; shift
  echo "=== $m $(date '+%F %T')" | tee -a "$LOG"
  $PY scripts/run_pii_unlearn.py $DATA --method "$m" --epochs "$EPOCHS" --bs "$BS" --out "$OUT" $STOP "$@" 2>&1 | tee -a "$LOG" | grep "\[data\]\|\[done\]\|before\|after\|^role\|^[TSCG] |\|Error\|Traceback"
}
for m in $METHODS; do
  case $m in
    GradAscent) run $m --lr "$LR" ;;
    GradDiff)   run $m --lr "$LR" --hp gamma=1.0 alpha=1.0 retain_loss_type=NLL ;;
    NPO)        run $m --lr "$LR" --hp beta=0.1 alpha=1.0 ;;
    SimNPO)     run $m --lr "$LR" --hp beta=4.5 delta=0.0 gamma=0.125 alpha=1.0 ;;
    DPO)        run $m --lr "$LR" --hp beta=0.1 alpha=1.0 --alt "[REDACTED]" ;;
    RMU)        run $m --lr "${RMU_LR:-5e-5}" --hp layer_id="${RMU_LAYER:-7}" steering_coeff=20 alpha="${RMU_ALPHA:-1}" ;;
    WGA)        run $m --lr "$LR" --hp beta=1.0 alpha=1.0 ;;
    SatImp)     run $m --lr "$LR" --hp beta1=5.0 beta2=1.0 gamma=0.1 alpha=1.0 ;;
    CEU)        run $m --lr "$LR" --hp ignore_first_n=1 ;;
    UNDIAL)     run $m --lr "$LR" --hp beta=10.0 alpha=0.0 ;;
    PDU)        run $m --lr "$LR" --hp gamma=1.0 alpha=1.0 primal_dual=false ;;
    JensUn)     run $m --lr "$LR" --hp target_text="\"No way\"" alpha=1.0 retain_loss_type=JSD ;;
    FLAT)       run $m --lr "$LR" --hp div=Total-Variation --alt "[REDACTED]" ;;
    REVS)       run $m --hp n_neurons=30 max_tokens=2 token_method=rarest ;;
    *) echo "unknown method $m" ;;
  esac
done
echo "=== summary" | tee -a "$LOG"
$PY scripts/summarise_pii.py "$OUT" | tee -a "$LOG"
