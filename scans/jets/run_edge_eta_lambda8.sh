#!/bin/bash
# Edge η₀×D transfer on Lambda 8×A100 — matched to M11-only panel (c).
#
# Specs (mirror scans/jets/scaling_law_L4_m11_eta_gputest.slurm):
#   model = capen-llama-att (M12+M12 + hyperedge M22; default CAPEN attn)
#   k★=12, R★=0.1 (locked operating point from panels a/b)
#   D ∈ {256,384,512}, η₀ ∈ {0.05,0.1,0.25,0.5,1.0}  → 15 cells
#   d_h=32 ⇒ H=D/32; Adam; 4000 steps; n_train=1e6; epochs=2
#   B_gpu=96 × 4 GPUs → B_global=384  (same as M11; 8-way DDP hung on this host)
#
# Sequential one-cell-at-a-time on GPUs 0–3 (avoids dual-DDP NCCL hangs).
#
#   bash scans/jets/run_edge_eta_lambda8.sh
#   TASKS="0 5 10" bash ...   # subset by flat index (width*5+eta)
#
set -uo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${HOME}/cpen/src:${HOME}/.local/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONNOUSERSITE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29510}"
# Fail fast on NCCL stalls instead of spinning 3 GPUs at 100% forever.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-600}"
# Dual-bank on this 8×A100 host: P2P/IB has correlated with 3×100%/1×0% stalls.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
# Set FRESH=1 to wipe any incomplete run dir before launch (avoids mid-epoch
# last.ckpt resume, which has stalled DDP on this host).
FRESH="${FRESH:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"

REPO="${HOME}/cpen"
RAW="${JETCLASS_RAW:-/lambda/nfs/jetclass-filesystem}"
ROOT="${RUN_ROOT:-${HOME}/runs/edge_eta}"
LOGDIR="${LOG_DIR:-${HOME}/logs/edge_eta}"
mkdir -p "$ROOT" "$LOGDIR"

STAR_R="${STAR_R:-0.1}"
KNN_K="${KNN_K:-12}"
BATCH="${BATCH:-96}"
N_GPUS="${N_GPUS:-4}"
MAX_STEPS="${MAX_STEPS:-4000}"
EPOCHS="${EPOCHS:-2}"
N_TRAIN="${N_TRAIN:-1000000}"
N_VAL=25000
SEED=0
DEPTH=4
D_HEAD=32
ETA0_LIST=(0.05 0.1 0.25 0.5 1.0)
WIDTHS=(256 384 512)
TAG=v6lam

R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
# Collector matches this substring (see EDGE_TAG in plot_graph_construction_nature.py).
# Override RUN_TAG_BASE when using an equivalent split (e.g. 128×3) that keeps B_global=384.
RUN_TAG_BASE="${RUN_TAG_BASE:-edge-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}}"

cd "$REPO/scans/jets"
echo "$(date -Iseconds) START edge η×D  STAR_R=$STAR_R k=$KNN_K B=${BATCH}x${N_GPUS}=$((BATCH*N_GPUS)) steps=$MAX_STEPS"
echo "run_tag_base=$RUN_TAG_BASE  gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

# Flat task indices 0..14 unless TASKS overridden.
if [[ -z "${TASKS:-}" ]]; then
  TASKS=$(seq 0 14)
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

  # Skip if parquet already reached target steps.
  EXISTING=$(find "$ROOT/jetclass_output/adam/sweep_lr" -maxdepth 1 -type d -name "*_${WIDTH}_${ETA_TAG}_*${RUN_TAG}*" 2>/dev/null | head -1 || true)
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

  # Incomplete run dir + last.ckpt → Lightning mid-epoch resume. On this 8×A100
  # host that has repeatedly left 3 GPUs at 100% / 1 at 0% (NCCL wait). Wipe
  # when FRESH=1, or always wipe last.ckpt so we start clean while keeping
  # parquet history only if FRESH=0 and dir exists… Prefer full wipe on FRESH.
  if [[ -n "$EXISTING" ]]; then
    if [[ "$FRESH" == "1" ]]; then
      echo "$(date -Iseconds) FRESH=1 wipe incomplete $EXISTING (step=${STEP:-?})"
      rm -rf "$EXISTING" "${EXISTING}.parquet"
      EXISTING=""
    elif [[ -f "$EXISTING/last.ckpt" || -f "$EXISTING/last-v1.ckpt" ]]; then
      echo "$(date -Iseconds) WARN incomplete run has last.ckpt; mid-epoch resume can DDP-stall. Re-run with FRESH=1 to wipe."
    fi
  fi

  echo "$(date -Iseconds) RUN task=$TASK D=$WIDTH H=$HEAD eta=$ETA0 tag=$RUN_TAG"
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
    --val-check-interval 500 \
    --checkpoint-every-n-steps 500 \
    --checkpoint-monitor val_loss \
    --checkpoint-mode min \
    --no-torch-compile \
    >"$OUT" 2>&1
  rc=$?
  set -e
  # Prefer parquet@MAX_STEPS as success (post-fit test may still trip).
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

echo "$(date -Iseconds) ALL edge η×D cells done (STAR_R=$STAR_R)"
