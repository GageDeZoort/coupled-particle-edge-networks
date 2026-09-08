#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=build-knn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# Full train (~1.2M jets, n=128, k=8) is RAM-heavy while stacking before torch.save.
#SBATCH --mem=1500G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_knn_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_knn_%j.err

# Pre-build TopTagging directed kNN graph caches (plain kNN, no pT cut).
# Default recipe: k=8, n=128
#   → {data-root}/processed/8nn/n128/{train,val,test}.pt
#
# Prefer the sharded ~1h cputest chain (recommended for full train):
#   bash scans/jets/submit_knn_graphs_cputest.sh
#
# This monolithic job is for small smokes / overrides only:
#   sbatch --export=ALL,N_TRAIN=50000 scans/jets/build_knn_graphs.slurm
#
# Overrides:
#   sbatch --export=ALL,GRAPH_CONSTRUCTION=8-NN,NUM_PARTICLES=128 \
#     scans/jets/build_knn_graphs.slurm

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

N_TRAIN="${N_TRAIN:-}"
N_VAL="${N_VAL:-}"
N_TEST="${N_TEST:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONNOUSERSITE=1

mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs

cd "${REPO_ROOT}"
echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Mem: ${SLURM_MEM_PER_NODE:-?} MB"
echo "data-root: ${DATA_ROOT}"
echo "graph-construction: ${GRAPH_CONSTRUCTION}"
echo "num-particles: ${NUM_PARTICLES}"
echo "rebuild: ${REBUILD}"

ARGS=(
  scans/jets/build_toptagging_graphs.py
  --data-root "${DATA_ROOT}"
  --graph-construction "${GRAPH_CONSTRUCTION}"
  --num-particles "${NUM_PARTICLES}"
  --data-seed "${DATA_SEED}"
)

if [[ "${REBUILD}" == "1" ]]; then
  ARGS+=(--rebuild)
fi
if [[ -n "${N_TRAIN}" ]]; then
  ARGS+=(--n-train "${N_TRAIN}")
fi
if [[ -n "${N_VAL}" ]]; then
  ARGS+=(--n-val "${N_VAL}")
fi
if [[ -n "${N_TEST}" ]]; then
  ARGS+=(--n-test "${N_TEST}")
fi

python "${ARGS[@]}"
echo "Done. Caches under ${DATA_ROOT}/processed/"
