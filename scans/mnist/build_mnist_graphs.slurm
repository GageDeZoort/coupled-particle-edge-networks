#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=build-mnist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_mnist_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_mnist_%j.err

# Incidence-v1 caches. Train/val are a 90/10 split of the official PyG train
# set (seed 42), same as hp-transfer-gts. Test is the official test set.
#
#   {data-root}/processed/mnist-sp/incidence-v1/{train,val,test}.pt
#
# Submit from repo root after download:
#   sbatch scans/mnist/build_mnist_graphs.slurm

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets}"
SPLITS="${SPLITS:-train,val,test}"
REBUILD="${REBUILD:-1}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONNOUSERSITE=1
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs

cd "${REPO_ROOT}"
echo "PWD=$(pwd)  host=$(hostname)  job=${SLURM_JOB_ID:-unset}"
echo "data-root=${DATA_ROOT}  splits=${SPLITS}  rebuild=${REBUILD}"

ARGS=(scans/mnist/build_mnist_graphs.py --data-root "${DATA_ROOT}" --splits "${SPLITS}")
if [[ "${REBUILD}" == "1" ]]; then
  ARGS+=(--rebuild)
fi
python "${ARGS[@]}"
echo "Done. Caches under ${DATA_ROOT}/processed/mnist-sp/incidence-v1/"
