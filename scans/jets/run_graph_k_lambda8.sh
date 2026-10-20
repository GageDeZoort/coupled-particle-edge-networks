#!/bin/bash
# Graph construction panel (a): kNN-k sweep at R★=0.1 on Lambda 8×A100.
#
#   D=256, L=4, H=8, η₀=0.25, Adam, 1000 steps
#   B_gpu=64 × 8 → B_eff=512
#   Default KS (refined 2026-10-02): {1,3,5,7} fill-in after even-k plateau
#     (k∈{2,4,6,8,10} already complete; k∈{12,14,16} skipped).
#   Override: KS_CSV=2,4,6 bash ...
#   3 shards × 2 seeds per k. Sequential on all 8 GPUs.
#
#   See paper/figures/fig_graph_construction_k_R0p1.md
#
set -uo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${HOME}/cpen/src:${HOME}/.local/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONNOUSERSITE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29720}"
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
ROOT="${RUN_ROOT:-${HOME}/runs/graph_k}"
LOGDIR="${LOG_DIR:-${HOME}/logs/graph_k}"
mkdir -p "$ROOT" "$LOGDIR"

STAR_R="${STAR_R:-0.1}"
BATCH="${BATCH:-64}"
N_GPUS="${N_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-1000}"
EPOCHS="${EPOCHS:-1}"
N_VAL=25000
DEPTH=4
WIDTH=256
HEAD=8
ETA0=0.25
# Refined odd-k fill (even plateau already done). Override with KS_CSV.
if [[ -n "${KS_CSV:-}" ]]; then
  IFS=',' read -r -a KS <<< "$KS_CSV"
else
  KS=(1 3 5 7)
fi
N_SHARD=3
N_INIT=2
N_PER_K=$((N_SHARD * N_INIT))
N_TRAIN=$((MAX_STEPS * BATCH * N_GPUS))
TAG=v1lam

R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")

cd "$REPO/scans/jets"
echo "$(date -Iseconds) START graph-k R=$STAR_R B=${BATCH}x${N_GPUS}=$((BATCH*N_GPUS)) steps=$MAX_STEPS n_train=$N_TRAIN"
echo "D=$WIDTH H=$HEAD eta0=$ETA0 ks=${KS[*]}  (attentional pool; NO --uniform-output-pool)"
echo "edge_features=part-interaction  operators=incidence"
echo "raw=$RAW root=$ROOT gpus=$CUDA_VISIBLE_DEVICES port=$MASTER_PORT"
nvidia-smi -L || true

n_vis=$(python3 -c "import torch; print(torch.cuda.device_count())")
if (( n_vis < N_GPUS )); then
  echo "Need $N_GPUS GPUs; visible=$n_vis" >&2
  exit 1
fi

if [[ -z "${TASKS:-}" ]]; then
  TASKS=$(seq 0 $(( ${#KS[@]} * N_PER_K - 1 )))
fi

for TASK in $TASKS; do
  if (( TASK < 0 || TASK >= ${#KS[@]} * N_PER_K )); then
    echo "skip bad TASK=$TASK" >&2
    continue
  fi
  K_IDX=$((TASK / N_PER_K))
  REM=$((TASK % N_PER_K))
  SHARD=$((REM / N_INIT))
  SEED=$((REM % N_INIT))
  KNN_K=${KS[$K_IDX]}
  TRAIN_SKIP=$((SHARD * N_TRAIN))
  RUN_TAG="graph-k-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-R${R_TAG}-shard${SHARD}-seed${SEED}-${TAG}"
  OUT="$LOGDIR/k${KNN_K}_shard${SHARD}_seed${SEED}_R${R_TAG}.out"

  EXISTING=$(find "$ROOT/jetclass_output/adam/sweep_lr" -maxdepth 1 -type d \
    -name "*_${WIDTH}_${ETA_TAG}_*knn${KNN_K}-*${RUN_TAG}*" 2>/dev/null | head -1 || true)
  STEP=0
  if [[ -n "$EXISTING" && -f "${EXISTING}.parquet" ]]; then
    STEP=$(python3 -c "
import pyarrow.parquet as pq
df=pq.read_table('${EXISTING}.parquet', columns=['global_step']).to_pandas()
print(int(df['global_step'].max()) if len(df) else 0)
" 2>/dev/null || echo 0)
    if (( STEP >= MAX_STEPS )); then
      echo "$(date -Iseconds) SKIP task=$TASK k=$KNN_K shard=$SHARD seed=$SEED (step=$STEP)"
      continue
    fi
  fi
  if [[ -n "$EXISTING" && "$FRESH" == "1" ]]; then
    echo "$(date -Iseconds) FRESH=1 wipe $EXISTING (step=$STEP)"
    rm -rf "$EXISTING" "${EXISTING}.parquet"
    EXISTING=""
  fi

  echo "$(date -Iseconds) RUN task=$TASK k=$KNN_K shard=$SHARD seed=$SEED skip=$TRAIN_SKIP tag=$RUN_TAG → $OUT"
  set +e
  # Attentional class-token pool = default (do NOT pass --uniform-output-pool).
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
    echo "$(date -Iseconds) OK task=$TASK k=$KNN_K shard=$SHARD seed=$SEED (rc=$rc step=$STEP_NOW)"
  else
    echo "$(date -Iseconds) FAIL task=$TASK k=$KNN_K shard=$SHARD seed=$SEED rc=$rc step=$STEP_NOW (see $OUT)"
    exit $rc
  fi
done

echo "$(date -Iseconds) ALL graph-k cells done (ks=${KS[*]})"
