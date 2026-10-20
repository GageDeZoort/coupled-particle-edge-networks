#!/bin/bash
# Hparam-xfer follow-up: constant-LR (no cosine) AdamW t-sweep + Adam baseline.
#
# Question: does AdamW / t-epoch WD help at D=256 independently of early cosine?
# Suspect: 5k flat → cosine@12k anneals too early; WD transfer may be confounded.
#
# Specs (same geometry as arm 1):
#   CAPEN k★=12 R★=0.1, L=4 D=256 H=8 (d_h=32), η₀=0.25
#   B_gpu=64 × 8 → B_eff=512; prescribed n_train=100M for t_epoch WD
#   Schedule: --scheduler none → constant peak LR η₀/√D for all steps
#   max_steps=10000
#
# Queue (sequential on 8×A100):
#   AdamW t∈{1,5,10} then Adam (no WD) η₀=0.25
#
#   bash scans/jets/run_hparam_xfer_constlr_lambda8.sh
#   ONLY=adamw T_EPOCHS="1 5" bash ...
#   ONLY=adam bash ...
#
set -uo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${HOME}/cpen/src:${HOME}/.local/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONNOUSERSITE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29630}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-600}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
FRESH="${FRESH:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

REPO="${HOME}/cpen"
RAW="${JETCLASS_RAW:-/lambda/nfs/jetclass-filesystem}"
ROOT="${RUN_ROOT:-${HOME}/runs/hparam_xfer}"
LOGDIR="${LOG_DIR:-${HOME}/logs/hparam_xfer}"
mkdir -p "$ROOT" "$LOGDIR"

STAR_R="${STAR_R:-0.1}"
KNN_K="${KNN_K:-12}"
BATCH="${BATCH:-64}"
N_GPUS="${N_GPUS:-8}"
ETA0="${ETA0:-0.25}"
WIDTH="${WIDTH:-256}"
HEAD="${HEAD:-8}"
DEPTH=4
SEED="${SEED:-0}"
N_TRAIN="${N_TRAIN:-100000000}"
N_VAL="${N_VAL:-25000}"
MAX_STEPS="${MAX_STEPS:-10000}"
T_EPOCHS="${T_EPOCHS:-1 5 10}"
ONLY="${ONLY:-all}"  # all | adamw | adam

R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
# Distinct tag from the 12k warmup_flat_cosine arm.
RUN_TAG_BASE="${RUN_TAG_BASE:-hxfer-constlr-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-k${KNN_K}-R${R_TAG}}"

cd "$REPO/scans/jets"
echo "$(date -Iseconds) START const-LR hxfer comparison"
echo "eta0=$ETA0 D=$WIDTH H=$HEAD k=$KNN_K R=$STAR_R B=${BATCH}x${N_GPUS}=$((BATCH*N_GPUS))"
echo "scheduler=none max_steps=$MAX_STEPS T_EPOCHS=$T_EPOCHS ONLY=$ONLY"
echo "run_tag_base=$RUN_TAG_BASE port=$MASTER_PORT"
nvidia-smi -L || true

n_vis=$(python3 -c "import torch; print(torch.cuda.device_count())")
if (( n_vis < N_GPUS )); then
  echo "Need $N_GPUS GPUs; visible=$n_vis" >&2
  exit 1
fi

run_one() {
  local OPT="$1"
  local T_EPOCH="${2:-}"
  local T_TAG RUN_TAG OUT EXISTING STEP rc
  if [[ "$OPT" == "adamw" ]]; then
    T_TAG=$(python3 -c "e=float('${T_EPOCH}'); print(f'{e:g}'.replace('.','p'))")
    RUN_TAG="${RUN_TAG_BASE}-adamw-t${T_TAG}"
    OUT="$LOGDIR/D${WIDTH}_eta${ETA_TAG}_constlr_adamw_t${T_TAG}_R${R_TAG}.out"
  else
    T_TAG="na"
    RUN_TAG="${RUN_TAG_BASE}-adam"
    OUT="$LOGDIR/D${WIDTH}_eta${ETA_TAG}_constlr_adam_R${R_TAG}.out"
  fi

  EXISTING=$(find "$ROOT/jetclass_output/${OPT}/sweep_lr" -maxdepth 1 -type d \
    -name "*_${WIDTH}_${ETA_TAG}_*${RUN_TAG}*" 2>/dev/null | head -1 || true)
  if [[ -n "$EXISTING" && -f "${EXISTING}.parquet" ]]; then
    STEP=$(python3 -c "
import pyarrow.parquet as pq
df=pq.read_table('${EXISTING}.parquet', columns=['global_step']).to_pandas()
print(int(df['global_step'].max()) if len(df) else 0)
" 2>/dev/null || echo 0)
    if (( STEP >= MAX_STEPS )); then
      echo "$(date -Iseconds) SKIP ${OPT} t=${T_EPOCH:-—} (step=$STEP >= $MAX_STEPS)"
      return 0
    fi
  fi
  if [[ -n "$EXISTING" && "$FRESH" == "1" ]]; then
    echo "$(date -Iseconds) FRESH=1 wipe $EXISTING"
    rm -rf "$EXISTING" "${EXISTING}.parquet"
  fi

  echo "$(date -Iseconds) RUN ${OPT} t=${T_EPOCH:-—} tag=$RUN_TAG → $OUT"
  set +e
  local -a CMD=(
    python3 test_cpen_jetclass.py
    --model capen-llama-att
    --operators incidence
    --jetclass-stream
    --jetclass-raw-root "$RAW"
    --jetclass-features kin7
    --jetclass-edge-features part-interaction
    --live-knn-k "$KNN_K"
    --star-radius "$STAR_R"
    --mode sweep_lr
    --optimizer "$OPT"
    --scheduler none
    --residual-structure transformer-like
    --etas "$ETA0"
    --epochs 1
    --max-steps "$MAX_STEPS"
    --n-train "$N_TRAIN"
    --n-test 0
    --data-root "$RAW"
    --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD"
    --dropout 0.0
    --root "$ROOT"
    --run-tag "$RUN_TAG"
    --batch-size "$BATCH"
    --n-gpus "$N_GPUS"
    --num-workers "$NUM_WORKERS"
    --normalization uniform
    --attention-normalization gamma
    --num-particles 80
    --n-val "$N_VAL"
    --seed "$SEED"
    --log-every-n-steps 100
    --layer-norm
    --check-val-every-n-epoch 1
    --val-check-interval 1000
    --checkpoint-every-n-steps 500
    --checkpoint-monitor val_loss
    --checkpoint-mode min
    --no-torch-compile
  )
  if [[ "$OPT" == "adamw" ]]; then
    CMD+=(--t-epoch "$T_EPOCH")
  fi
  "${CMD[@]}" >"$OUT" 2>&1
  rc=$?
  set -e
  echo "$(date -Iseconds) DONE ${OPT} t=${T_EPOCH:-—} rc=$rc"
  if (( rc != 0 )); then
    echo "FAIL ${OPT} t=${T_EPOCH:-—}" >&2
    exit "$rc"
  fi
}

if [[ "$ONLY" == "all" || "$ONLY" == "adamw" ]]; then
  for T_EPOCH in $T_EPOCHS; do
    run_one adamw "$T_EPOCH"
  done
fi
if [[ "$ONLY" == "all" || "$ONLY" == "adam" ]]; then
  run_one adam
fi

echo "$(date -Iseconds) ALL const-LR comparison cells done"
