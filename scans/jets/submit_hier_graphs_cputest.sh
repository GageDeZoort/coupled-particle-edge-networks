#!/bin/bash
# Submit a dependency chain of ~1h TopTagging hierarchical rebuild jobs on Della
# CPU QOS "test" (--qos=test). There is no "cputest" partition.
#
# From repo root:
#   bash scans/jets/submit_hier_graphs_cputest.sh
#   K=8 EPS=0.08 NUM_PARTICLES=128 bash scans/jets/submit_hier_graphs_cputest.sh
#
# Defaults sized conservatively for DBSCAN + virtual structure (slower than plain
# kNN). Increase TRAIN_SHARDS if shards approach the 59m wallclock:
#   train 1.211M → 20 shards (~60k each)
#   val   0.403M →  6 shards (~67k each)
#   test  0.404M →  6 shards (~67k each)
#
# Pipeline:
#   1) train/val/test shards (parallel; QOS MaxJobsPU usually limits concurrency)
#   2) merge each split after its shards (train merge wants high mem)
#
# STAGE/SPLIT/shard index are passed as *positional* sbatch args (not --export),
# which is reliable when the site default is ExportEnv=NONE.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/build_hier_graphs_cputest.slurm"

ACCOUNT="${ACCOUNT:-bhanin}"
K="${K:-8}"
EPS="${EPS:-0.08}"
MIN_SAMPLES="${MIN_SAMPLES:-2}"
N_VIRTUAL_NODES="${N_VIRTUAL_NODES:-1}"
N_VIRTUAL_EDGES="${N_VIRTUAL_EDGES:-1}"
MAX_DBSCAN_EDGES="${MAX_DBSCAN_EDGES:-32}"
NUM_PARTICLES="${NUM_PARTICLES:-128}"
REBUILD="${REBUILD:-1}"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets/toptagging}"
BUILD_BATCH_SIZE="${BUILD_BATCH_SIZE:-64}"

TRAIN_SHARDS="${TRAIN_SHARDS:-20}"
VAL_SHARDS="${VAL_SHARDS:-6}"
TEST_SHARDS="${TEST_SHARDS:-6}"

# Construction knobs still go through --export (optional); STAGE/SPLIT via argv.
COMMON="ALL,DATA_ROOT=${DATA_ROOT},K=${K},EPS=${EPS},MIN_SAMPLES=${MIN_SAMPLES},N_VIRTUAL_NODES=${N_VIRTUAL_NODES},N_VIRTUAL_EDGES=${N_VIRTUAL_EDGES},MAX_DBSCAN_EDGES=${MAX_DBSCAN_EDGES},NUM_PARTICLES=${NUM_PARTICLES},REBUILD=${REBUILD},BUILD_BATCH_SIZE=${BUILD_BATCH_SIZE}"

EPS_TAG=$(echo "${EPS}" | tr '.' 'p')
TAG="hier_k${K}_eps${EPS_TAG}_ms${MIN_SAMPLES}_vn${N_VIRTUAL_NODES}_ve${N_VIRTUAL_EDGES}_mdb${MAX_DBSCAN_EDGES}"

echo "Submitting TopTagging hierarchical cputest chain"
echo "  account=${ACCOUNT} tag=${TAG} n=${NUM_PARTICLES} rebuild=${REBUILD}"
echo "  shards: train=${TRAIN_SHARDS} val=${VAL_SHARDS} test=${TEST_SHARDS}"

submit_shards() {
  local split="$1"
  local n_shards="$2"
  local mem="$3"
  local ids=()
  local i jid
  for i in $(seq 0 $((n_shards - 1))); do
    jid=$(sbatch --parsable \
      --account="${ACCOUNT}" \
      --partition=cpu \
      --qos=test \
      --job-name="hier-${split}-s${i}" \
      --mem="${mem}" \
      --export="${COMMON}" \
      "${SLURM_SCRIPT}" shard "${split}" "${i}" "${n_shards}")
    ids+=("${jid}")
    echo "  shard ${split} ${i}/${n_shards} -> ${jid}" >&2
  done
  (IFS=:; echo "${ids[*]}")
}

train_dep=$(submit_shards train "${TRAIN_SHARDS}" "256G")
val_dep=$(submit_shards val "${VAL_SHARDS}" "192G")
test_dep=$(submit_shards test "${TEST_SHARDS}" "192G")

train_merge=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --partition=cpu \
  --qos=test \
  --dependency="afterok:${train_dep}" \
  --job-name="hier-train-merge" \
  --mem=900G \
  --cpus-per-task=8 \
  --export="${COMMON}" \
  "${SLURM_SCRIPT}" merge train - "${TRAIN_SHARDS}")
echo "  merge train -> ${train_merge} (after ${train_dep})"

val_merge=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --partition=cpu \
  --qos=test \
  --dependency="afterok:${val_dep}" \
  --job-name="hier-val-merge" \
  --mem=450G \
  --export="${COMMON}" \
  "${SLURM_SCRIPT}" merge val - "${VAL_SHARDS}")
echo "  merge val -> ${val_merge} (after ${val_dep})"

test_merge=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --partition=cpu \
  --qos=test \
  --dependency="afterok:${test_dep}" \
  --job-name="hier-test-merge" \
  --mem=450G \
  --export="${COMMON}" \
  "${SLURM_SCRIPT}" merge test - "${TEST_SHARDS}")
echo "  merge test -> ${test_merge} (after ${test_dep})"

echo
echo "Canonical caches will land under:"
echo "  ${DATA_ROOT}/processed/${TAG}/n${NUM_PARTICLES}/{train,val,test}.pt"
echo "Track with: squeue -u \$USER"
echo "Note: QOS test often allows only ~2 running jobs; the rest stay Pending."
echo "If shards approach 59m, resubmit with TRAIN_SHARDS=30 (etc.)."
