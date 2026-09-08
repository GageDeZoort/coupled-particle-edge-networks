#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=build-graphs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=768G
#SBATCH --time=0:59:00
#SBATCH --output=build_graphs_%j.out
#SBATCH --error=build_graphs_%j.err

# Pre-build TopTagging kNN graph caches on a high-memory CPU node.
#
# Full train split (1.21M jets, n=100, k=8) needs ~400 GB RAM peak while
# tensors are stacked before torch.save. Login nodes will OOM-kill this job.
#
# Submit from repo root:
#   sbatch scans/jets/build_graphs.slurm
#
# Or override settings at submit time, e.g.:
#   sbatch --mem=512G scans/jets/build_graphs.slurm
#   sbatch --export=ALL,NUM_PARTICLES=200 scans/jets/build_graphs.slurm

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets/toptagging}"
GRAPH_CONSTRUCTION="${GRAPH_CONSTRUCTION:-8-NN}"
NUM_PARTICLES="${NUM_PARTICLES:-100}"
DATA_SEED="${DATA_SEED:-42}"

# Optional subsample limits (leave empty for full splits).
N_TRAIN="${N_TRAIN:-}"
N_VAL="${N_VAL:-}"
N_TEST="${N_TEST:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_ROOT}"
echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "data-root: ${DATA_ROOT}"
echo "graph-construction: ${GRAPH_CONSTRUCTION}"
echo "num-particles: ${NUM_PARTICLES}"

ARGS=(
  scans/jets/build_toptagging_graphs.py
  --data-root "${DATA_ROOT}"
  --graph-construction "${GRAPH_CONSTRUCTION}"
  --num-particles "${NUM_PARTICLES}"
  --data-seed "${DATA_SEED}"
)

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
