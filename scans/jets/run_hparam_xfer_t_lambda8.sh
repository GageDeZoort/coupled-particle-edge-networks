#!/bin/bash
# Hyperparameter-transfer proof — arm 1: t-sweep at fixed η₀ on Lambda 8×A100.
#
# Specs (see paper/figures/fig_hparam_transfer_proof.md):
#   CAPEN k★=12 R★=0.1, L=4 D=256 H=8 (d_h=32), AdamW, η₀=0.25
#   B_gpu=64 × 8 GPUs → B_eff=512
#   Prescribed n_train = 100M for t_epoch / WD always:
#     λ = √D · B_eff / (t_epoch · η₀ · n_train),  η = η₀/√D
#     (physical Lambda stage may be a subset; WD still uses 100M)
#   Schedule: 5k constant (peak) LR → cosine over remainder of one 100M epoch
#     cosine_steps = ceil(100M/512) − 5k = 190313 (floor 0.01×peak if fully run)
#   Plot / wall-clock cutoff: max_steps=12k (5k flat + first 7k of cosine)
#   Default queue: t_epoch ∈ {5,10,20}. With ADAPTIVE_AFTER_T10=1 (default):
#     if final val ROC(t=10) ≤ ROC(t=5), skip remaining (e.g. t=20) and run t=1 next.
#
#   T_EPOCHS=5 bash scans/jets/run_hparam_xfer_t_lambda8.sh
#   T_EPOCHS="5 10 20" bash ...
#   ADAPTIVE_AFTER_T10=0 T_EPOCHS="5 10 20" bash ...   # force full list
#
set -uo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${HOME}/cpen/src:${HOME}/.local/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONNOUSERSITE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29610}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-600}"
# 8-way DDP on this host: disable P2P/IB (correlated with 3×100%/1×0% stalls).
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
# Prescribed dataset size D for t_epoch ↔ λ₀ (full JetClass train), not the
# physical subset on disk.
N_TRAIN="${N_TRAIN:-100000000}"
N_VAL="${N_VAL:-25000}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
FLAT_STEPS="${FLAT_STEPS:-5000}"
# Total run budget (flat + first slice of cosine).
MAX_STEPS="${MAX_STEPS:-12000}"
LR_MIN_FRAC="${LR_MIN_FRAC:-0.01}"
# Full one-epoch step count at B_eff; cosine schedule covers the remainder
# after the constant-LR prefix so LR would hit 0.01×peak at end of that epoch.
EPOCH_STEPS="${EPOCH_STEPS:-$(python3 -c "import math; print(math.ceil(${N_TRAIN}/(${BATCH}*${N_GPUS})))")}"
COSINE_STEPS="${COSINE_STEPS:-$((EPOCH_STEPS - WARMUP_STEPS - FLAT_STEPS))}"
ANNEAL_RUN_STEPS=$((MAX_STEPS - WARMUP_STEPS - FLAT_STEPS))
if (( ANNEAL_RUN_STEPS < 1 )); then
  echo "MAX_STEPS=$MAX_STEPS too small for flat prefix $FLAT_STEPS" >&2
  exit 1
fi
if (( COSINE_STEPS < ANNEAL_RUN_STEPS )); then
  echo "COSINE_STEPS=$COSINE_STEPS < anneal_run=$ANNEAL_RUN_STEPS" >&2
  exit 1
fi
T_EPOCHS="${T_EPOCHS:-5 10 20}"
ADAPTIVE_AFTER_T10="${ADAPTIVE_AFTER_T10:-1}"
FALLBACK_T_IF_T10_FLAT="${FALLBACK_T_IF_T10_FLAT:-1}"

R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG_BASE="${RUN_TAG_BASE:-hxfer-t-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-k${KNN_K}-R${R_TAG}}"

final_val_roc() {
  # Best-effort: last val_roc_auc= in the run log (final checkpoint line).
  local out="$1"
  python3 -c "
import re, sys
p=sys.argv[1]
try:
  txt=open(p).read()
except FileNotFoundError:
  print('nan'); raise SystemExit
hits=re.findall(r'val_roc_auc=([0-9.eE+-]+)', txt)
print(hits[-1] if hits else 'nan')
" "$out" 2>/dev/null || echo nan
}

cd "$REPO/scans/jets"
echo "$(date -Iseconds) START hparam-xfer t-sweep"
echo "eta0=$ETA0 D=$WIDTH H=$HEAD k=$KNN_K R=$STAR_R B=${BATCH}x${N_GPUS}=$((BATCH*N_GPUS))"
echo "epoch_steps=$EPOCH_STEPS warmup=$WARMUP_STEPS flat=$FLAT_STEPS cosine_schedule=$COSINE_STEPS anneal_run=$ANNEAL_RUN_STEPS max_steps=$MAX_STEPS"
echo "T_EPOCHS=$T_EPOCHS  ADAPTIVE_AFTER_T10=$ADAPTIVE_AFTER_T10 fallback_t=$FALLBACK_T_IF_T10_FLAT"
echo "run_tag_base=$RUN_TAG_BASE  gpus=$CUDA_VISIBLE_DEVICES port=$MASTER_PORT"
nvidia-smi -L || true

n_vis=$(python3 -c "import torch; print(torch.cuda.device_count())")
if (( n_vis < N_GPUS )); then
  echo "Need $N_GPUS GPUs; visible=$n_vis" >&2
  exit 1
fi

# Mutable queue so ADAPTIVE_AFTER_T10 can rewrite remaining cells.
read -r -a QUEUE <<< "$T_EPOCHS"
DONE_TS=()
qi=0
while (( qi < ${#QUEUE[@]} )); do
  T_EPOCH="${QUEUE[$qi]}"
  T_TAG=$(python3 -c "e=float('${T_EPOCH}'); print(f'{e:g}'.replace('.','p'))")
  RUN_TAG="${RUN_TAG_BASE}-t${T_TAG}"
  OUT="$LOGDIR/D${WIDTH}_eta${ETA_TAG}_t${T_TAG}_R${R_TAG}.out"

  EXISTING=$(find "$ROOT/jetclass_output/adamw/sweep_lr" -maxdepth 1 -type d \
    -name "*_${WIDTH}_${ETA_TAG}_t${T_TAG}_*${RUN_TAG}*" 2>/dev/null | head -1 || true)
  if [[ -n "$EXISTING" && -f "${EXISTING}.parquet" ]]; then
    STEP=$(python3 -c "
import pyarrow.parquet as pq
df=pq.read_table('${EXISTING}.parquet', columns=['global_step']).to_pandas()
print(int(df['global_step'].max()) if len(df) else 0)
" 2>/dev/null || echo 0)
    if (( STEP >= MAX_STEPS )); then
      echo "$(date -Iseconds) SKIP t=$T_EPOCH (step=$STEP >= $MAX_STEPS)"
      DONE_TS+=("$T_EPOCH")
      qi=$((qi + 1))
      continue
    fi
  fi
  if [[ -n "$EXISTING" && "$FRESH" == "1" ]]; then
    echo "$(date -Iseconds) FRESH=1 wipe $EXISTING"
    rm -rf "$EXISTING" "${EXISTING}.parquet"
  fi

  echo "$(date -Iseconds) RUN t=$T_EPOCH tag=$RUN_TAG → $OUT"
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
    --optimizer adamw \
    --t-epoch "$T_EPOCH" \
    --scheduler warmup_flat_cosine \
    --warmup-steps "$WARMUP_STEPS" \
    --flat-steps "$FLAT_STEPS" \
    --cosine-steps "$COSINE_STEPS" \
    --lr-decay-min-frac "$LR_MIN_FRAC" \
    --residual-structure transformer-like \
    --etas "$ETA0" \
    --epochs 1 \
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
    --log-every-n-steps 100 \
    --layer-norm \
    --check-val-every-n-epoch 1 \
    --val-check-interval 1000 \
    --checkpoint-every-n-steps 500 \
    --checkpoint-monitor val_loss \
    --checkpoint-mode min \
    --no-torch-compile \
    >"$OUT" 2>&1
  rc=$?
  set -e
  echo "$(date -Iseconds) DONE t=$T_EPOCH rc=$rc"
  if (( rc != 0 )); then
    echo "FAIL t=$T_EPOCH — stopping sweep" >&2
    exit "$rc"
  fi
  DONE_TS+=("$T_EPOCH")

  # After t=10: if it did not beat t=5, drop remaining queue cells and run fallback t.
  if [[ "$ADAPTIVE_AFTER_T10" == "1" ]]; then
    t_cmp=$(python3 -c "e=float('${T_EPOCH}'); print(f'{e:g}')")
    if [[ "$t_cmp" == "10" ]]; then
      OUT5="$LOGDIR/D${WIDTH}_eta${ETA_TAG}_t5_R${R_TAG}.out"
      OUT10="$OUT"
      roc5=$(final_val_roc "$OUT5")
      roc10=$(final_val_roc "$OUT10")
      echo "$(date -Iseconds) ADAPTIVE check: ROC(t=5)=$roc5  ROC(t=10)=$roc10"
      decide=$(python3 -c "
r5=float('$roc5'); r10=float('$roc10')
import math
if math.isnan(r5) or math.isnan(r10):
  print('keep')  # don't rewrite if metrics missing
elif r10 > r5:
  print('keep')
else:
  print('fallback')
")
      if [[ "$decide" == "fallback" ]]; then
        fb="$FALLBACK_T_IF_T10_FLAT"
        echo "$(date -Iseconds) t=10 did not improve on t=5 → skip remaining ${QUEUE[*]:$((qi+1))} ; next t=$fb"
        QUEUE=("$fb")
        qi=0
        continue
      else
        echo "$(date -Iseconds) t=10 improved (or metrics missing) → continue remaining queue"
      fi
    fi
  fi
  qi=$((qi + 1))
done

echo "$(date -Iseconds) ALL t-cells done: ${DONE_TS[*]} (last queue head had ${QUEUE[*]})"
