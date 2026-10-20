#!/bin/bash
# Graph construction panel (b): R★ sweep at fixed k★=6 on Lambda 8×A100.
#
# Matched to panel (a) Lambda recipe:
#   D=256, L=4, H=8, η₀=0.25, Adam, 1000 steps
#   B_gpu=64 × 8 → B_eff=512
#   k★=6 (plateau operating point from refined k scan)
#   R★ ∈ {0.05, 0.075, 0.1, 0.125, 0.15}
#   3 shards × 2 seeds = 6 / R★ → 30 cells
#   Attentional pool (no --uniform-output-pool); part-interaction edges.
#
#   See paper/figures/fig_graph_construction_k_R0p1.md
#
#   nohup bash scans/jets/run_graph_R_lambda8.sh >~/logs/graph_R/master.log 2>&1 &
#
set -uo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${HOME}/cpen/src:${HOME}/.local/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONNOUSERSITE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29730}"
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
ROOT="${RUN_ROOT:-${HOME}/runs/graph_R}"
LOGDIR="${LOG_DIR:-${HOME}/logs/graph_R}"
mkdir -p "$ROOT" "$LOGDIR"

KNN_K="${KNN_K:-6}"
BATCH="${BATCH:-64}"
N_GPUS="${N_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-1000}"
EPOCHS="${EPOCHS:-1}"
N_VAL=25000
DEPTH=4
WIDTH=256
HEAD=8
ETA0=0.25
if [[ -n "${RADII_CSV:-}" ]]; then
  IFS=',' read -r -a RADII <<< "$RADII_CSV"
else
  RADII=(0.05 0.075 0.1 0.125 0.15)
fi
N_SHARD=3
N_INIT=2
N_PER_R=$((N_SHARD * N_INIT))
N_TRAIN=$((MAX_STEPS * BATCH * N_GPUS))
TAG=v1lam

ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")

cd "$REPO/scans/jets"
echo "$(date -Iseconds) START graph-R k=$KNN_K B=${BATCH}x${N_GPUS}=$((BATCH*N_GPUS)) steps=$MAX_STEPS"
echo "D=$WIDTH H=$HEAD eta0=$ETA0 radii=${RADII[*]}  (attentional pool)"
echo "raw=$RAW root=$ROOT"
nvidia-smi -L || true

n_vis=$(python3 -c "import torch; print(torch.cuda.device_count())")
if (( n_vis < N_GPUS )); then
  echo "Need $N_GPUS GPUs; visible=$n_vis" >&2
  exit 1
fi

if [[ -z "${TASKS:-}" ]]; then
  TASKS=$(seq 0 $(( ${#RADII[@]} * N_PER_R - 1 )))
fi

for TASK in $TASKS; do
  if (( TASK < 0 || TASK >= ${#RADII[@]} * N_PER_R )); then
    echo "skip bad TASK=$TASK" >&2
    continue
  fi
  R_IDX=$((TASK / N_PER_R))
  REM=$((TASK % N_PER_R))
  SHARD=$((REM / N_INIT))
  SEED=$((REM % N_INIT))
  STAR_R=${RADII[$R_IDX]}
  TRAIN_SKIP=$((SHARD * N_TRAIN))
  R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
  RUN_TAG="graph-R-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-k${KNN_K}-R${R_TAG}-shard${SHARD}-seed${SEED}-${TAG}"
  OUT="$LOGDIR/R${R_TAG}_k${KNN_K}_shard${SHARD}_seed${SEED}.out"

  EXISTING=$(find "$ROOT/jetclass_output/adam/sweep_lr" -maxdepth 1 -type d \
    -name "*_${WIDTH}_${ETA_TAG}_*gstar${R_TAG}*knn${KNN_K}-*${RUN_TAG}*" 2>/dev/null | head -1 || true)
  STEP=0
  if [[ -n "$EXISTING" && -f "${EXISTING}.parquet" ]]; then
    STEP=$(python3 -c "
import pyarrow.parquet as pq
df=pq.read_table('${EXISTING}.parquet', columns=['global_step']).to_pandas()
print(int(df['global_step'].max()) if len(df) else 0)
" 2>/dev/null || echo 0)
    if (( STEP >= MAX_STEPS )); then
      echo "$(date -Iseconds) SKIP task=$TASK R=$STAR_R shard=$SHARD seed=$SEED (step=$STEP)"
      continue
    fi
  fi
  if [[ -n "$EXISTING" && "$FRESH" == "1" ]]; then
    echo "$(date -Iseconds) FRESH=1 wipe $EXISTING (step=$STEP)"
    rm -rf "$EXISTING" "${EXISTING}.parquet"
    EXISTING=""
  fi

  echo "$(date -Iseconds) RUN task=$TASK R=$STAR_R k=$KNN_K shard=$SHARD seed=$SEED skip=$TRAIN_SKIP → $OUT"
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
    --train-skip-jets "$TRAIN_SKIP" \
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

  STEP_NOW=0
  PQ=$(find "$ROOT/jetclass_output/adam/sweep_lr" -maxdepth 1 -type f \
    -name "*_${WIDTH}_${ETA_TAG}_*knn${KNN_K}-*${RUN_TAG}.parquet" 2>/dev/null | head -1 || true)
  if [[ -n "$PQ" ]]; then
    STEP_NOW=$(python3 -c "
import pyarrow.parquet as pq
df=pq.read_table('$PQ', columns=['global_step']).to_pandas()
print(int(df['global_step'].max()) if len(df) else 0)
" 2>/dev/null || echo 0)
  fi
  if (( rc == 0 || STEP_NOW >= MAX_STEPS )); then
    echo "$(date -Iseconds) OK task=$TASK R=$STAR_R shard=$SHARD seed=$SEED (rc=$rc step=$STEP_NOW)"
  else
    echo "$(date -Iseconds) FAIL task=$TASK R=$STAR_R shard=$SHARD seed=$SEED rc=$rc step=$STEP_NOW (see $OUT)"
    exit $rc
  fi
done

echo "$(date -Iseconds) ALL graph-R cells done"
