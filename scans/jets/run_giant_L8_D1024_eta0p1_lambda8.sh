#!/bin/bash
# Giant CAPEN Adam — L=8 D=1024 d_h=32 on Lambda 8×A100.
#
# Pretrain home (with L4 full-epoch continue on Della):
#   /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/
#   Lambda live root: ~/runs/giant_L8D1024  (mirrored → pretrain/lambda_L8D1024)
#
# η₀=0.1 (frozen from L4/D256 + L6/D512 40k edge η grids).
# Geometry: k★=7, R★=0.1, all-seed star kin7 ∥ ΔR-kNN noloop.
#
#   L=8, D=1024, d_h=32 → H=32
#   Adam, --scheduler none (const peak LR η=η₀/√D)
#   B_gpu=32 × 8 GPUs → B_eff=256  (40GB A100s OOM at B=64)
#   Unique 40k steps (same protocol as Della η grids):
#     n_train=10_240_000, max_steps=40000
#   n_val=0 — train only; eval from last.ckpt later
#
# Data: class-balanced stream via plan_shards (same as Della). Requires
# ≥205 train ROOT files at launch (plan fixed at dataset init; 100k jets/file).
# 10×21 staged for headroom.
#
#   bash scans/jets/run_giant_L8_D1024_eta0p1_lambda8.sh
#
set -uo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${HOME}/cpen/src:${HOME}/.local/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
unset PYTHONNOUSERSITE
# Prefer smaller CUDA caching fragments on 40GB A100s.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29810}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-600}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
FRESH="${FRESH:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"

REPO="${HOME}/cpen"
RAW="${JETCLASS_RAW:-/home/ubuntu/data/jetclass}"
ROOT="${RUN_ROOT:-${HOME}/runs/giant_L8D1024}"
LOGDIR="${LOG_DIR:-${HOME}/logs/giant_L8D1024}"
mkdir -p "$ROOT" "$LOGDIR"

STAR_R="${STAR_R:-0.1}"
KNN_K="${KNN_K:-7}"
# 40GB A100s: B=64 OOMs (~39GB/GPU on first step). B=32 → B_eff=256.
BATCH="${BATCH:-32}"
N_GPUS="${N_GPUS:-8}"
# Unique 40k @ B_eff=256 (same unique-jet count as Della L4/L6 η grids).
N_TRAIN="${N_TRAIN:-$((40000 * BATCH * N_GPUS))}"
MAX_STEPS="${MAX_STEPS:-40000}"
EPOCHS="${EPOCHS:-1}"
N_VAL="${N_VAL:-0}"
NEED_PER_CLASS="${NEED_PER_CLASS:-21}"
SEED=0
DEPTH=8
WIDTH=1024
D_HEAD=32
ETA0="${ETA0:-0.1}"
HEAD=$((WIDTH / D_HEAD))
if (( HEAD * D_HEAD != WIDTH )); then
  echo "D=$WIDTH not divisible by d_h=$D_HEAD" >&2
  exit 1
fi

R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="${RUN_TAG:-giant-L${DEPTH}D${WIDTH}-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}-eta${ETA_TAG}-s40k}"
OUT="$LOGDIR/L${DEPTH}_D${WIDTH}_eta${ETA_TAG}_R${R_TAG}.out"

cd "$REPO/scans/jets"
echo "$(date -Iseconds) START giant L=$DEPTH D=$WIDTH d_h=$D_HEAD H=$HEAD eta0=$ETA0"
echo "B=${BATCH}x${N_GPUS}=$((BATCH*N_GPUS)) steps=$MAX_STEPS epochs=$EPOCHS n_train=$N_TRAIN n_val=$N_VAL"
echo "k=$KNN_K R=$STAR_R scheduler=none attn-pool noloop unique-stream s40k"
echo "raw=$RAW run_tag=$RUN_TAG gpus=$CUDA_VISIBLE_DEVICES port=$MASTER_PORT FRESH=$FRESH"
nvidia-smi -L || true

python3 - <<PY
from pathlib import Path
import json
from collections import Counter
from datetime import datetime, timezone
raw = Path("${RAW}")
run_tag = "${RUN_TAG}"
need_per = int("${NEED_PER_CLASS}")
train = sorted((raw/"train_100M").glob("*.root"))
n = len(train)
need_jets = int("${N_TRAIN}")
need_files = (need_jets + 100_000 - 1) // 100_000
by = Counter(p.name.rsplit("_", 1)[0] for p in train)
print(f"[data] train ROOT files={n} (need ≥{need_files} for n_train={need_jets})")
print(f"[data] per-class counts: {dict(sorted(by.items()))}")
if n < need_files:
    raise SystemExit(f"Insufficient train files under {raw}/train_100M: {n} < {need_files}")
if len(by) < 10 or min(by.values()) < need_per:
    raise SystemExit(
        f"Need ≥{need_per} files in each of 10 classes for unique-40k mix; got {dict(by)}"
    )
print("[data] train-only (n_val=0); class-interleaved plan_shards as on Della")
man = raw / "manifests"
man.mkdir(parents=True, exist_ok=True)
names = [p.name for p in train]
used_path = man / f"{run_tag}_used_train.txt"
used_path.write_text(
    "# Unique 40k — files present at launch\n" + "\n".join(names) + "\n"
)
ledger = man / "usage_ledger.jsonl"
rec = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "event": "train_launch",
    "run_tag": run_tag,
    "n_train_files": n,
    "n_train": need_jets,
    "max_steps": int("${MAX_STEPS}"),
    "batch_eff": int("${BATCH}") * int("${N_GPUS}"),
    "d_head": int("${D_HEAD}"),
    "per_class": dict(by),
    "files": names,
}
with ledger.open("a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"[data] wrote {used_path} ({n} files)")
PY

n_vis=$(python3 -c "import torch; print(torch.cuda.device_count())")
if (( n_vis < N_GPUS )); then
  echo "Need $N_GPUS GPUs; visible=$n_vis" >&2
  exit 1
fi

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
    echo "$(date -Iseconds) SKIP already done (step=$STEP >= $MAX_STEPS)"
    exit 0
  fi
fi
if [[ -n "$EXISTING" && "$FRESH" == "1" ]]; then
  echo "$(date -Iseconds) FRESH=1 wipe $EXISTING (step=$STEP)"
  rm -rf "$EXISTING" "${EXISTING}.parquet"
  EXISTING=""
fi

echo "$(date -Iseconds) RUN → $OUT"
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
  --live-star-kin7 \
  --mode sweep_lr \
  --optimizer adam \
  --scheduler none \
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
  --limit-val-batches 0 \
  --seed "$SEED" \
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --checkpoint-every-n-steps 1000 \
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
  echo "$(date -Iseconds) OK (rc=$rc step=$STEP_NOW)"
  exit 0
fi
echo "$(date -Iseconds) FAIL rc=$rc step=$STEP_NOW (see $OUT)"
exit $rc
