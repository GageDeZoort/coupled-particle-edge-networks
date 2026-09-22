#!/usr/bin/env bash
# Login-node watchdog: resubmit 10M att grid and 5M (4,1024,32) until epoch-2 val exists.
# Does not use --export=ALL (DEPTH/WIDTH must not leak into the 10M jobs).
set -euo pipefail

SLURM=/home/jdezoort/coupled-particle-edge-networks/scans/jets/test_capen_llama_att_knn_hyper.slurm
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs
LOGDIR=$ROOT
PARQ=$ROOT/jetclass_output/adam/sweep_lr
OUT=$ROOT/jc_att_watchdog.log
PYTHON="${PYTHON:-$HOME/.conda/envs/mamba-env/bin/python}"

sbatch_10m() {
  local task=$1
  sbatch --array="$task" --job-name=jc-att-10M \
    --export=N_TRAIN=10000000,EPOCHS=3,USE_ROPE=0,BATCH_SIZE=512 \
    --output=$LOGDIR/jc_att_10M_%A_%a.out \
    --error=$LOGDIR/jc_att_10M_%A_%a.err \
    "$SLURM"
}

sbatch_w1024() {
  local task=$1
  sbatch --array="$task" --job-name=jc-att-5M-w1024 \
    --export=N_TRAIN=5000000,EPOCHS=3,USE_ROPE=0,BATCH_SIZE=128,DEPTH=4,WIDTH=1024,HEADS=32 \
    --output=$LOGDIR/jc_att_5M_w1024_%A_%a.out \
    --error=$LOGDIR/jc_att_5M_w1024_%A_%a.err \
    "$SLURM"
}

queued_tasks() {
  local name=$1
  squeue -u "$USER" -n "$name" -h -o '%i' 2>/dev/null | awk -F_ '{
    id=$2
    if (id ~ /^\[/) {
      gsub(/[\[\]]/, "", id)
      n=split(id, a, /[-,]/)
      # only handle lo-hi ranges like 3-13 or 0-6
      if (id ~ /-/) {
        split(id, b, /-/)
        for (i=b[1]; i<=b[2]; i++) print i
      } else print id
    } else if (NF>=2) print $NF
  }' | sort -n | uniq
}

epoch2_done() {
  local glob=$1
  "$PYTHON" - "$PARQ" "$glob" <<'PY'
import sys
from pathlib import Path
root, glob = Path(sys.argv[1]), sys.argv[2]
import pandas as pd
hits = [p for p in root.glob(glob) if "rope" not in p.name]
if not hits:
    raise SystemExit(1)
df = pd.read_parquet(hits[0])
if "epoch" not in df.columns or "val_roc_auc" not in df.columns:
    raise SystemExit(1)
sub = df.dropna(subset=["val_roc_auc"])
if sub.empty:
    raise SystemExit(1)
raise SystemExit(0 if (sub["epoch"] >= 2).any() else 1)
PY
}

STEMS_10M=(
  "capen-llama-att_3_256_0p01_*ntr10M*.parquet"
  "capen-llama-att_3_256_0p05_*ntr10M*.parquet"
  "capen-llama-att_3_256_0p1_*ntr10M*.parquet"
  "capen-llama-att_3_256_0p25_*ntr10M*.parquet"
  "capen-llama-att_3_256_0p5_*ntr10M*.parquet"
  "capen-llama-att_3_256_1_*ntr10M*.parquet"
  "capen-llama-att_3_256_2p5_*ntr10M*.parquet"
  "capen-llama-att_6_512_0p01_*ntr10M*.parquet"
  "capen-llama-att_6_512_0p05_*ntr10M*.parquet"
  "capen-llama-att_6_512_0p1_*ntr10M*.parquet"
  "capen-llama-att_6_512_0p25_*ntr10M*.parquet"
  "capen-llama-att_6_512_0p5_*ntr10M*.parquet"
  "capen-llama-att_6_512_1_*ntr10M*.parquet"
  "capen-llama-att_6_512_2p5_*ntr10M*.parquet"
)
STEMS_W=(
  "capen-llama-att_4_1024_0p01_*ntr5M*.parquet"
  "capen-llama-att_4_1024_0p05_*ntr5M*.parquet"
  "capen-llama-att_4_1024_0p1_*ntr5M*.parquet"
  "capen-llama-att_4_1024_0p25_*ntr5M*.parquet"
  "capen-llama-att_4_1024_0p5_*ntr5M*.parquet"
  "capen-llama-att_4_1024_1_*ntr5M*.parquet"
  "capen-llama-att_4_1024_2p5_*ntr5M*.parquet"
)

tick() {
  local q10 qw
  q10=$(queued_tasks jc-att-10M | tr '\n' ' ')
  qw=$(queued_tasks jc-att-5M-w1024 | tr '\n' ' ')
  echo "$(date -Iseconds) queued 10M=[$q10] w1024=[$qw]" >>"$OUT"

  local i stem
  for i in $(seq 0 13); do
    if echo " $q10 " | grep -q " $i "; then
      continue
    fi
    stem=${STEMS_10M[$i]}
    if epoch2_done "$stem"; then
      continue
    fi
    echo "$(date -Iseconds) sbatch 10M task $i" >>"$OUT"
    sbatch_10m "$i" >>"$OUT" 2>&1 || true
  done
  for i in $(seq 0 6); do
    if echo " $qw " | grep -q " $i "; then
      continue
    fi
    stem=${STEMS_W[$i]}
    if epoch2_done "$stem"; then
      continue
    fi
    echo "$(date -Iseconds) sbatch w1024 task $i" >>"$OUT"
    sbatch_w1024 "$i" >>"$OUT" 2>&1 || true
  done
}

mkdir -p "$(dirname "$OUT")"
echo "$(date -Iseconds) watchdog start pid=$$" >>"$OUT"
while true; do
  tick || echo "$(date -Iseconds) tick error $?" >>"$OUT"
  # stop if everything is done
  all=1
  for i in $(seq 0 13); do
    epoch2_done "${STEMS_10M[$i]}" || all=0
  done
  for i in $(seq 0 6); do
    epoch2_done "${STEMS_W[$i]}" || all=0
  done
  if [[ $all -eq 1 ]]; then
    echo "$(date -Iseconds) all epoch-2 vals present; exiting" >>"$OUT"
    exit 0
  fi
  sleep 180
done
