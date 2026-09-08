#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --job-name=hier-cputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_hier_cputest_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_hier_cputest_%j.err

# One ~1h TopTagging hierarchical cache piece for Della CPU QOS "test"
# (there is no "cputest" partition — use --qos=test).
#
# Prefer the helper (passes STAGE/SPLIT as positional args — reliable on Della):
#   bash scans/jets/submit_hier_graphs_cputest.sh
#
# Manual examples:
#   sbatch scans/jets/build_hier_graphs_cputest.slurm shard train 0 20
#   sbatch --mem=900G scans/jets/build_hier_graphs_cputest.slurm merge train - 20
#
# Positional args (preferred; survive ExportEnv=NONE):
#   $1 STAGE       = shard | merge
#   $2 SPLIT       = train | val | test
#   $3 SHARD_INDEX = 0..N_SHARDS-1  (use "-" for merge)
#   $4 N_SHARDS    = total shards for this split
#
# Optional env knobs: K EPS MIN_SAMPLES N_VIRTUAL_* MAX_DBSCAN_EDGES
#   NUM_PARTICLES REBUILD BUILD_BATCH_SIZE DATA_ROOT

set -euo pipefail

# Positional args win over env (env alone is often stripped by Slurm).
if [[ "${1:-}" == "shard" || "${1:-}" == "merge" ]]; then
  STAGE="$1"
  SPLIT="${2:?SPLIT required as arg 2 (train|val|test)}"
  if [[ "${3:-}" != "" && "${3:-}" != "-" ]]; then
    SHARD_INDEX="$3"
  else
    SHARD_INDEX="${SHARD_INDEX:-}"
  fi
  N_SHARDS="${4:?N_SHARDS required as arg 4}"
else
  STAGE="${STAGE:-}"
  SPLIT="${SPLIT:-}"
  SHARD_INDEX="${SHARD_INDEX:-}"
  N_SHARDS="${N_SHARDS:-}"
fi

if [[ -z "${STAGE}" || -z "${SPLIT}" || -z "${N_SHARDS}" ]]; then
  echo "ERROR: STAGE/SPLIT/N_SHARDS missing." >&2
  echo "Use: bash scans/jets/submit_hier_graphs_cputest.sh" >&2
  echo "Or:  sbatch $0 shard train 0 20" >&2
  echo "     sbatch --mem=900G $0 merge train - 20" >&2
  exit 1
fi
if [[ "${STAGE}" != "shard" && "${STAGE}" != "merge" ]]; then
  echo "ERROR: STAGE must be shard|merge; got '${STAGE}'" >&2
  exit 1
fi

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets/toptagging}"
K="${K:-8}"
EPS="${EPS:-0.08}"
MIN_SAMPLES="${MIN_SAMPLES:-2}"
N_VIRTUAL_NODES="${N_VIRTUAL_NODES:-1}"
N_VIRTUAL_EDGES="${N_VIRTUAL_EDGES:-1}"
MAX_DBSCAN_EDGES="${MAX_DBSCAN_EDGES:-32}"
NUM_PARTICLES="${NUM_PARTICLES:-128}"
DATA_SEED="${DATA_SEED:-42}"
REBUILD="${REBUILD:-1}"
BUILD_BATCH_SIZE="${BUILD_BATCH_SIZE:-64}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONNOUSERSITE=1

mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs
cd "${REPO_ROOT}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "stage=${STAGE} split=${SPLIT} shard=${SHARD_INDEX:-na}/${N_SHARDS}"
echo "k=${K} eps=${EPS} ms=${MIN_SAMPLES} vn=${N_VIRTUAL_NODES} ve=${N_VIRTUAL_EDGES} mdb=${MAX_DBSCAN_EDGES}"
echo "n=${NUM_PARTICLES} rebuild=${REBUILD} batch=${BUILD_BATCH_SIZE}"
echo "mem=${SLURM_MEM_PER_NODE:-?} cpus=${SLURM_CPUS_PER_TASK}"

ARGS=(
  scans/jets/build_toptagging_hier_cputest.py
  --data-root "${DATA_ROOT}"
  --stage "${STAGE}"
  --split "${SPLIT}"
  --k "${K}"
  --eps "${EPS}"
  --min-samples "${MIN_SAMPLES}"
  --n-virtual-nodes "${N_VIRTUAL_NODES}"
  --n-virtual-edges "${N_VIRTUAL_EDGES}"
  --max-dbscan-edges "${MAX_DBSCAN_EDGES}"
  --num-particles "${NUM_PARTICLES}"
  --data-seed "${DATA_SEED}"
  --build-batch-size "${BUILD_BATCH_SIZE}"
  --n-shards "${N_SHARDS}"
)
if [[ "${REBUILD}" == "1" ]]; then
  ARGS+=(--rebuild)
fi
if [[ "${STAGE}" == "shard" ]]; then
  if [[ -z "${SHARD_INDEX}" ]]; then
    echo "ERROR: SHARD_INDEX required for STAGE=shard (arg 3)"
    exit 1
  fi
  ARGS+=(--shard-index "${SHARD_INDEX}")
fi

python "${ARGS[@]}"
echo "Done stage=${STAGE} split=${SPLIT}."
