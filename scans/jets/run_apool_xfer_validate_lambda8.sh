#!/bin/bash
# Attentional-pool μP transfer validation on Lambda 8×GPU.
#
# Twin of Della uniform-pool xfer validate
#   (scans/jets/scaling_law_L4_upool_xfer_validate_gputest.slurm)
# but **attentional** class-token pool (no --uniform-output-pool) and
# B_eff=384 split across all 8 GPUs: B_gpu=48 × 8.
#
#   L=4, d_h=32, k=10, R★=0.1
#   D ∈ {256, 512} × η₀ ∈ {0.05, 0.1, 0.25, 0.5, 1.0}
#   Adam η=η₀/√D, 1000 steps, n_train=1e6
#
# Flat TASK = width_idx*5 + eta_idx  (0–9).
#
#   # on Lambda host, from ~/cpen:
#   bash scans/jets/run_apool_xfer_validate_lambda8.sh
#   TASKS="0 1 2" bash scans/jets/run_apool_xfer_validate_lambda8.sh
#
# See paper/figures/fig_stability_transfer_validation.md (attn section).
#
set -uo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${HOME}/cpen/src:${HOME}/.local/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONNOUSERSITE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29710}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-600}"
# 8-way DDP on this host has hung with P2P/IB; keep conservative defaults.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
FRESH="${FRESH:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

REPO="${HOME}/cpen"
RAW="${JETCLASS_RAW:-/lambda/nfs/jetclass-filesystem}"
ROOT="${RUN_ROOT:-${HOME}/runs/apool_xfer}"
LOGDIR="${LOG_DIR:-${HOME}/logs/apool_xfer}"
mkdir -p "$ROOT" "$LOGDIR"

STAR_R="${STAR_R:-0.1}"
KNN_K="${KNN_K:-10}"
BATCH="${BATCH:-48}"
N_GPUS="${N_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-1000}"
EPOCHS="${EPOCHS:-2}"
N_TRAIN="${N_TRAIN:-1000000}"
N_VAL=25000
SEED=0
DEPTH=4
D_HEAD=32
ETA0_LIST=(0.05 0.1 0.25 0.5 1.0)
WIDTHS=(256 512)

R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG_BASE="${RUN_TAG_BASE:-apool-xfer-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}}"

cd "$REPO/scans/jets"
echo "$(date -Iseconds) START attn-pool xfer validate"
echo "B=${BATCH}x${N_GPUS}=$((BATCH*N_GPUS)) steps=$MAX_STEPS k=$KNN_K R=$STAR_R"
echo "run_tag=$RUN_TAG_BASE gpus=$CUDA_VISIBLE_DEVICES port=$MASTER_PORT"
nvidia-smi -L || true

n_vis=$(python3 -c "import torch; print(torch.cuda.device_count())")
if (( n_vis < N_GPUS )); then
  echo "Need $N_GPUS GPUs; visible=$n_vis" >&2
  exit 1
fi

if [[ -z "${TASKS:-}" ]]; then
  TASKS=$(seq 0 9)
fi

for TASK in $TASKS; do
  WIDTH_IDX=$((TASK / 5))
  ETA_IDX=$((TASK % 5))
  if (( WIDTH_IDX < 0 || WIDTH_IDX >= ${#WIDTHS[@]} || ETA_IDX < 0 || ETA_IDX >= ${#ETA0_LIST[@]} )); then
    echo "skip bad TASK=$TASK" >&2
    continue
  fi
  WIDTH=${WIDTHS[$WIDTH_IDX]}
  ETA0=${ETA0_LIST[$ETA_IDX]}
  HEAD=$((WIDTH / D_HEAD))
  if (( HEAD * D_HEAD != WIDTH )); then
    echo "D=$WIDTH not divisible by d_h=$D_HEAD" >&2
    exit 1
  fi
  ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
  RUN_TAG="${RUN_TAG_BASE}"
  OUT="$LOGDIR/D${WIDTH}_eta${ETA_TAG}_R${R_TAG}.out"

  EXISTING=$(find "$ROOT/jetclass_output/adam/sweep_lr" -maxdepth 1 -type d \
    -name "*_${WIDTH}_${ETA_TAG}_*${RUN_TAG}*" 2>/dev/null | head -1 || true)
  STEP=0
  if [[ -n "$EXISTING" && -f "${EXISTING}.parquet" ]]; then
    STEP=$(python3 -c "
import pyarrow.parquet as pq
df=pq.read_table('${EXISTING}.parquet', columns=['global_step']).to_pandas()
print(int(df['global_step'].max()) if len(df) else 0)
" 2>/dev/null || echo 0)
    if (( STEP >= MAX_STEPS )); then
      echo "$(date -Iseconds) SKIP D=$WIDTH eta=$ETA0 (step=$STEP >= $MAX_STEPS)"
      continue
    fi
  fi
  if [[ -n "$EXISTING" && "$FRESH" == "1" ]]; then
    echo "$(date -Iseconds) FRESH=1 wipe $EXISTING (step=$STEP)"
    rm -rf "$EXISTING" "${EXISTING}.parquet"
    EXISTING=""
  fi

  echo "$(date -Iseconds) RUN task=$TASK D=$WIDTH H=$HEAD eta=$ETA0 tag=$RUN_TAG → $OUT"
  set +e
  # Attentional pool = default (do NOT pass --uniform-output-pool).
  python3 test_cpen_jetclass.py \
    --model capen-llama-att \
    --operators incidence \
    --jetclass-stream \
    --jetclass-raw-root "$RAW" \
    --jetclass-features kin7 \
    --jetclass-edge-features part-interaction \
    --live-knn-k "$KNN_K" \
    --star-radius "$STAR_R" \
    --mode sweep_lr \
    --optimizer adam \
    --t-epoch 1 \
    --residual-structure transformer-like \
    --etas "$ETA0" \
    --epochs "$EPOCHS" \
    --max-steps "$MAX_STEPS" \
    --n-train "$N_TRAIN" \
    --n-test 0 \
    --data-root "$RAW" \
    --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
    --dropout 0.0 \
    --root "$ROOT" \
    --run-tag "$RUN_TAG" \
    --batch-size "$BATCH" \
    --n-gpus "$N_GPUS" \
    --num-workers "$NUM_WORKERS" \
    --normalization uniform \
    --attention-normalization gamma \
    --num-particles 80 \
    --n-val "$N_VAL" \
    --seed "$SEED" \
    --log-every-n-steps 50 \
    --layer-norm \
    --check-val-every-n-epoch 1 \
    --val-check-interval 200 \
    --checkpoint-every-n-steps 500 \
    --checkpoint-monitor val_loss \
    --checkpoint-mode min \
    --no-torch-compile \
    >"$OUT" 2>&1
  rc=$?
  set -e

  STEP_NOW=0
  PQ=$(find "$ROOT/jetclass_output/adam/sweep_lr" -maxdepth 1 -type f \
    -name "*_${WIDTH}_${ETA_TAG}_*${RUN_TAG}.parquet" 2>/dev/null | head -1 || true)
  if [[ -n "$PQ" ]]; then
    STEP_NOW=$(python3 -c "
import pyarrow.parquet as pq
df=pq.read_table('$PQ', columns=['global_step']).to_pandas()
print(int(df['global_step'].max()) if len(df) else 0)
" 2>/dev/null || echo 0)
  fi
  if (( rc == 0 || STEP_NOW >= MAX_STEPS )); then
    echo "$(date -Iseconds) OK D=$WIDTH eta=$ETA0 (rc=$rc step=$STEP_NOW)"
  else
    echo "$(date -Iseconds) FAIL D=$WIDTH eta=$ETA0 rc=$rc step=$STEP_NOW (see $OUT)"
    exit $rc
  fi
done

echo "$(date -Iseconds) ALL attn-pool xfer cells done"
