#!/bin/bash
# Submit a dependency chain of ~1h TopTagging kNN rebuild jobs on Della CPU
# QOS "test" (--qos=test). There is no "cputest" partition.
#
# From repo root:
#   bash scans/jets/submit_knn_graphs_cputest.sh
#   GRAPH_CONSTRUCTION=4-NN NUM_PARTICLES=128 bash scans/jets/submit_knn_graphs_cputest.sh
#
# Defaults sized so each shard materializes in well under 1h
# (n=128, k=4; ~2–3k jets/s in practice):
#   train 1.211M → 10 shards (~121k each)
#   val   0.403M →  3 shards (~134k each)
#   test  0.404M →  3 shards (~135k each)
#
# Pipeline:
#   1) train/val/test shards (parallel; QOS MaxJobsPU usually limits concurrency)
#   2) merge each split after its shards (train merge wants high mem)
#
# Cancel a stuck monolithic build first if needed:
#   scancel <jobid>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/build_knn_graphs_cputest.slurm"

GRAPH_CONSTRUCTION="${GRAPH_CONSTRUCTION:-4-NN}"
NUM_PARTICLES="${NUM_PARTICLES:-128}"
REBUILD="${REBUILD:-1}"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets/toptagging}"

TRAIN_SHARDS="${TRAIN_SHARDS:-10}"
VAL_SHARDS="${VAL_SHARDS:-3}"
TEST_SHARDS="${TEST_SHARDS:-3}"

COMMON="ALL,DATA_ROOT=${DATA_ROOT},GRAPH_CONSTRUCTION=${GRAPH_CONSTRUCTION},NUM_PARTICLES=${NUM_PARTICLES},REBUILD=${REBUILD}"

echo "Submitting TopTagging kNN cputest chain"
echo "  graph=${GRAPH_CONSTRUCTION} n=${NUM_PARTICLES} rebuild=${REBUILD}"
echo "  shards: train=${TRAIN_SHARDS} val=${VAL_SHARDS} test=${TEST_SHARDS}"

submit_shards() {
  local split="$1"
  local n_shards="$2"
  local mem="$3"
  local ids=()
  local i jid
  for i in $(seq 0 $((n_shards - 1))); do
    jid=$(sbatch --parsable \
      --job-name="knn-${split}-s${i}" \
      --mem="${mem}" \
      --export="${COMMON},STAGE=shard,SPLIT=${split},SHARD_INDEX=${i},N_SHARDS=${n_shards}" \
      "${SLURM_SCRIPT}")
    ids+=("${jid}")
    echo "  shard ${split} ${i}/${n_shards} -> ${jid}" >&2
  done
  (IFS=:; echo "${ids[*]}")
}

train_dep=$(submit_shards train "${TRAIN_SHARDS}" "256G")
val_dep=$(submit_shards val "${VAL_SHARDS}" "192G")
test_dep=$(submit_shards test "${TEST_SHARDS}" "192G")

# Merges: train needs enough RAM to concat ~full incidence payload.
train_merge=$(sbatch --parsable \
  --dependency="afterok:${train_dep}" \
  --job-name="knn-train-merge" \
  --mem=800G \
  --cpus-per-task=8 \
  --export="${COMMON},STAGE=merge,SPLIT=train,N_SHARDS=${TRAIN_SHARDS}" \
  "${SLURM_SCRIPT}")
echo "  merge train -> ${train_merge} (after ${train_dep})"

val_merge=$(sbatch --parsable \
  --dependency="afterok:${val_dep}" \
  --job-name="knn-val-merge" \
  --mem=400G \
  --export="${COMMON},STAGE=merge,SPLIT=val,N_SHARDS=${VAL_SHARDS}" \
  "${SLURM_SCRIPT}")
echo "  merge val -> ${val_merge} (after ${val_dep})"

test_merge=$(sbatch --parsable \
  --dependency="afterok:${test_dep}" \
  --job-name="knn-test-merge" \
  --mem=400G \
  --export="${COMMON},STAGE=merge,SPLIT=test,N_SHARDS=${TEST_SHARDS}" \
  "${SLURM_SCRIPT}")
echo "  merge test -> ${test_merge} (after ${test_dep})"

echo
echo "Canonical caches will land under:"
TAG=$(echo "${GRAPH_CONSTRUCTION}" | tr '[:upper:]' '[:lower:]' | tr -d '-')
echo "  ${DATA_ROOT}/processed/${TAG}/n${NUM_PARTICLES}/{train,val,test}.pt"
echo "Track with: squeue -u \$USER"
echo "Note: QOS test often allows only ~2 running jobs; the rest stay Pending."
