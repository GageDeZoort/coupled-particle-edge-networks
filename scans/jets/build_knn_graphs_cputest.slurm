#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --job-name=knn-cputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_knn_cputest_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_knn_cputest_%j.err

# One ~1h TopTagging kNN cache piece for Della CPU QOS "test"
# (there is no "cputest" partition — use --qos=test).
#
# Prefer the helper:
#   bash scans/jets/submit_knn_graphs_cputest.sh
#
# Manual examples:
#   sbatch --export=ALL,STAGE=shard,SPLIT=train,SHARD_INDEX=0,N_SHARDS=10,REBUILD=1 \
#     scans/jets/build_knn_graphs_cputest.slurm
#   sbatch --mem=800G --export=ALL,STAGE=merge,SPLIT=train,N_SHARDS=10,REBUILD=1 \
#     scans/jets/build_knn_graphs_cputest.slurm
#
# Env knobs:
#   STAGE          = shard | merge
#   SPLIT          = train | val | test
#   SHARD_INDEX    = 0..N_SHARDS-1   (required for shard)
#   N_SHARDS       = total shards for this split
#   GRAPH_CONSTRUCTION (default 8-NN)
#   NUM_PARTICLES  (default 128)
#   REBUILD        (default 1)

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets/toptagging}"
GRAPH_CONSTRUCTION="${GRAPH_CONSTRUCTION:-4-NN}"
NUM_PARTICLES="${NUM_PARTICLES:-128}"
DATA_SEED="${DATA_SEED:-42}"
REBUILD="${REBUILD:-1}"
STAGE="${STAGE:?STAGE required (shard|merge)}"
SPLIT="${SPLIT:?SPLIT required (train|val|test)}"
N_SHARDS="${N_SHARDS:?N_SHARDS required}"
SHARD_INDEX="${SHARD_INDEX:-}"

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
echo "graph=${GRAPH_CONSTRUCTION} n=${NUM_PARTICLES} rebuild=${REBUILD}"
echo "mem=${SLURM_MEM_PER_NODE:-?} cpus=${SLURM_CPUS_PER_TASK}"

ARGS=(
  scans/jets/build_toptagging_knn_cputest.py
  --data-root "${DATA_ROOT}"
  --stage "${STAGE}"
  --split "${SPLIT}"
  --graph-construction "${GRAPH_CONSTRUCTION}"
  --num-particles "${NUM_PARTICLES}"
  --data-seed "${DATA_SEED}"
  --n-shards "${N_SHARDS}"
)
if [[ "${REBUILD}" == "1" ]]; then
  ARGS+=(--rebuild)
fi
if [[ "${STAGE}" == "shard" ]]; then
  if [[ -z "${SHARD_INDEX}" ]]; then
    echo "ERROR: SHARD_INDEX required for STAGE=shard"
    exit 1
  fi
  ARGS+=(--shard-index "${SHARD_INDEX}")
fi

python "${ARGS[@]}"
echo "Done stage=${STAGE} split=${SPLIT}."
