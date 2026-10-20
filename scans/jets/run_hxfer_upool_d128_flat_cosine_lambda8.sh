#!/bin/bash
# D=128 μP probe — uniform pool, Adam warmup_flat_cosine.
#
# Schedule (per user 2026-10-05):
#   10k steps flat at peak LR η=η₀/√D
#   then 10k steps cosine → 0.01 × peak LR
# Data: n_train=1_000_000 pool (historical "1M" hxfer recipe), enough
# epochs to cover max_steps without cutting early. Single width D=128.
#
#   η₀ ∈ {0.01, 0.05, 0.1, 0.5, 1.0}   → 5 cells, TASK 0–4
#   B=64×8=512, k★=6, R★=0.1, L=4, d_h=32, seed=0
#
#   bash scans/jets/run_hxfer_upool_d128_flat_cosine_lambda8.sh
#   TASKS="0 1" FRESH=1 bash scans/jets/run_hxfer_upool_d128_flat_cosine_lambda8.sh
#
set -uo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${HOME}/cpen/src:${HOME}/.local/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONNOUSERSITE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29760}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-600}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
FRESH="${FRESH:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"

REPO="${HOME}/cpen"
RAW="${JETCLASS_RAW:-/lambda/nfs/test-jetclass-filesystem}"
ROOT="${RUN_ROOT:-${HOME}/runs/hxfer_upool_d128_flatcos}"
LOGDIR="${LOG_DIR:-${HOME}/logs/hxfer_upool_d128_flatcos}"
mkdir -p "$ROOT" "$LOGDIR"

STAR_R="${STAR_R:-0.1}"
KNN_K="${KNN_K:-6}"
BATCH="${BATCH:-64}"
N_GPUS="${N_GPUS:-8}"
FLAT_STEPS="${FLAT_STEPS:-10000}"
COSINE_STEPS="${COSINE_STEPS:-10000}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MAX_STEPS="${MAX_STEPS:-$((FLAT_STEPS + COSINE_STEPS))}"
# 1M-jet pool; repeat as needed to reach max_steps (~10.2 epochs @ B_eff=512).
N_TRAIN="${N_TRAIN:-1000000}"
STEPS_PER_EPOCH=$(( (N_TRAIN + BATCH * N_GPUS - 1) / (BATCH * N_GPUS) ))
EPOCHS="${EPOCHS:-$(( (MAX_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))}"
N_VAL=25000
SEED=0
DEPTH=4
D_HEAD=32
LR_MIN_FRAC="${LR_MIN_FRAC:-0.01}"
_DEFAULT_ETA0=(0.01 0.05 0.1 0.5 1.0)
if [[ -n "${ETA0_LIST:-}" ]]; then
  # shellcheck disable=SC2206
  ETA0_LIST=(${ETA0_LIST})
else
  ETA0_LIST=("${_DEFAULT_ETA0[@]}")
fi
WIDTHS=(128)
N_ETA=${#ETA0_LIST[@]}

R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
# Keep short: warmup_flat_cosine already appends w/f/c/min to the run dir name.
# A long RUN_TAG previously hit ENAMETOOLONG on the parquet path.
RUN_TAG_BASE="${RUN_TAG_BASE:-hxfer-d128-fc-k${KNN_K}-R${R_TAG}}"

cd "$REPO/scans/jets"
echo "$(date -Iseconds) START hxfer upool D=128 flat→cosine"
echo "B=${BATCH}x${N_GPUS}=$((BATCH*N_GPUS)) flat=$FLAT_STEPS cosine=$COSINE_STEPS max_steps=$MAX_STEPS"
echo "n_train=$N_TRAIN steps/epoch≈$STEPS_PER_EPOCH epochs=$EPOCHS min_frac=$LR_MIN_FRAC"
echo "k=$KNN_K R=$STAR_R D∈{${WIDTHS[*]}} η₀∈{${ETA0_LIST[*]}} scheduler=warmup_flat_cosine UNIFORM-pool"
echo "raw=$RAW run_tag=$RUN_TAG_BASE gpus=$CUDA_VISIBLE_DEVICES port=$MASTER_PORT FRESH=$FRESH"
nvidia-smi -L || true

python3 - <<PY
from pathlib import Path
raw = Path("${RAW}")
train = sorted((raw/"train_100M").glob("*.root"))
n = len(train)
need_jets = int("${N_TRAIN}")
need_files = (need_jets + 100_000 - 1) // 100_000
print(f"[data] train ROOT files={n} (need ≥{need_files} for n_train={need_jets})")
if n < need_files:
    raise SystemExit(f"Insufficient train files under {raw}/train_100M: {n} < {need_files}")
PY

n_vis=$(python3 -c "import torch; print(torch.cuda.device_count())")
if (( n_vis < N_GPUS )); then
  echo "Need $N_GPUS GPUs; visible=$n_vis" >&2
  exit 1
fi

if [[ -z "${TASKS:-}" ]]; then
  TASKS=$(seq 0 $(( ${#WIDTHS[@]} * N_ETA - 1 )))
fi

for TASK in $TASKS; do
  WIDTH_IDX=$((TASK / N_ETA))
  ETA_IDX=$((TASK % N_ETA))
  if (( WIDTH_IDX < 0 || WIDTH_IDX >= ${#WIDTHS[@]} || ETA_IDX < 0 || ETA_IDX >= N_ETA )); then
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

  echo "$(date -Iseconds) RUN task=$TASK D=$WIDTH H=$HEAD eta=$ETA0 n_train=$N_TRAIN tag=$RUN_TAG → $OUT"
  set +e
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
    --scheduler warmup_flat_cosine \
    --warmup-steps "$WARMUP_STEPS" \
    --flat-steps "$FLAT_STEPS" \
    --cosine-steps "$COSINE_STEPS" \
    --lr-decay-min-frac "$LR_MIN_FRAC" \
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
    --uniform-output-pool \
    --num-particles 80 \
    --n-val "$N_VAL" \
    --seed "$SEED" \
    --log-every-n-steps 50 \
    --layer-norm \
    --check-val-every-n-epoch 1 \
    --val-check-interval 500 \
    --checkpoint-every-n-steps 1000 \
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

echo "$(date -Iseconds) ALL hxfer upool D=128 flat→cosine cells done"
