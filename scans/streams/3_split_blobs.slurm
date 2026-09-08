#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=s3-blobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:59:00
#SBATCH --array=0-18
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/streams/s3_blobs_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/streams/s3_blobs_%A_%a.err

# Stage 3: drop w>1.65 Gaussians, sky k-means split of blobs with N>8000.
#   sbatch scans/streams/3_split_blobs.slurm
#   sbatch --export=ALL,SPLIT=val --array=0-9 scans/streams/3_split_blobs.slurm

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
SPLIT="${SPLIT:-train}"
OUT_ROOT="${OUT_ROOT:-/scratch/gpfs/BHANIN/jgdezoort/streams/galaxies}"
REBUILD="${REBUILD:-0}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1

mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/streams
cd "${REPO_ROOT}"

echo "PWD=$(pwd)  Host=$(hostname)  Job=${SLURM_JOB_ID}  array=${SLURM_ARRAY_TASK_ID}  SPLIT=${SPLIT}"

ARGS=(
  scans/streams/3_split_blobs.py
  --split "${SPLIT}"
  --array-task "${SLURM_ARRAY_TASK_ID}"
  --out-root "${OUT_ROOT}"
)
if [[ "${REBUILD}" == "1" ]]; then
  ARGS+=(--rebuild)
fi

python "${ARGS[@]}"
echo "Done stage 3 ${SPLIT} task ${SLURM_ARRAY_TASK_ID}"
