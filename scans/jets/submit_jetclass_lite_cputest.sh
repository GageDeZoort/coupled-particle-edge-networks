#!/bin/bash
# Submit a dependency chain of ~1h JetClassLite rebuild jobs on Della CPU QOS
# "test" (--qos=test). There is no "cputest" partition.
#
# From repo root:
#   bash scans/jets/submit_jetclass_lite_cputest.sh
#   STAR_RADIUS=0.1 bash scans/jets/submit_jetclass_lite_cputest.sh
#
# Pipeline:
#   1) 10x train lite class shards (parallelism limited by QOS MaxJobsPU=2)
#   2) merge train lite
#   3) lite val + lite test
#   4) graphs val + graphs test
#   5) graphs train (high mem)
#
# Cancel the monolithic pending job first if it is still queued:
#   scancel 12117879

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/build_jetclass_lite_cputest.slurm"

STAR_RADIUS="${STAR_RADIUS:-0.15}"
REBUILD="${REBUILD:-1}"
COMMON="ALL,STAR_RADIUS=${STAR_RADIUS},REBUILD=${REBUILD}"

echo "Submitting JetClassLite cputest chain (star-R=${STAR_RADIUS}, rebuild=${REBUILD})"

# --- train lite: 10 class shards ---
shard_ids=()
for c in $(seq 0 9); do
  jid=$(sbatch --parsable \
    --job-name="jc-lite-c${c}" \
    --mem=128G \
    --export="${COMMON},STAGE=lite-shard,SPLITS=train,CLASS=${c}" \
    "${SLURM_SCRIPT}")
  shard_ids+=("${jid}")
  echo "  lite-shard train class ${c} -> ${jid}"
done
shard_dep=$(IFS=:; echo "${shard_ids[*]}")

# --- merge train lite (after all shards) ---
merge_jid=$(sbatch --parsable \
  --dependency="afterok:${shard_dep}" \
  --job-name="jc-lite-merge" \
  --mem=400G \
  --export="${COMMON},STAGE=lite-merge,SPLITS=train" \
  "${SLURM_SCRIPT}")
echo "  lite-merge train -> ${merge_jid} (after ${shard_dep})"

# --- lite val / test (independent of train shards; can run in parallel with them) ---
val_lite_jid=$(sbatch --parsable \
  --job-name="jc-lite-val" \
  --mem=128G \
  --export="${COMMON},STAGE=lite,SPLITS=val" \
  "${SLURM_SCRIPT}")
echo "  lite val -> ${val_lite_jid}"

test_lite_jid=$(sbatch --parsable \
  --job-name="jc-lite-test" \
  --mem=256G \
  --export="${COMMON},STAGE=lite,SPLITS=test" \
  "${SLURM_SCRIPT}")
echo "  lite test -> ${test_lite_jid}"

# --- graphs: val/test after their lite archives; train after merge ---
val_g_jid=$(sbatch --parsable \
  --dependency="afterok:${val_lite_jid}" \
  --job-name="jc-graph-val" \
  --mem=256G \
  --export="${COMMON},STAGE=graphs,SPLITS=val" \
  "${SLURM_SCRIPT}")
echo "  graphs val -> ${val_g_jid}"

test_g_jid=$(sbatch --parsable \
  --dependency="afterok:${test_lite_jid}" \
  --job-name="jc-graph-test" \
  --mem=500G \
  --export="${COMMON},STAGE=graphs,SPLITS=test" \
  "${SLURM_SCRIPT}")
echo "  graphs test -> ${test_g_jid}"

train_g_jid=$(sbatch --parsable \
  --dependency="afterok:${merge_jid}" \
  --job-name="jc-graph-train" \
  --mem=1200G \
  --cpus-per-task=16 \
  --export="${COMMON},STAGE=graphs,SPLITS=train" \
  "${SLURM_SCRIPT}")
echo "  graphs train -> ${train_g_jid}"

echo
echo "Track with: squeue -u \$USER"
echo "Note: QOS test allows only ~2 running jobs; the rest stay Pending until slots free."
