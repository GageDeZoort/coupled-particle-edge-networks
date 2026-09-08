#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=s1-phys
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:59:00
#SBATCH --array=0-18
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/streams/s1_phys_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/streams/s1_phys_%A_%a.err

# Stage 1: frozen CMD slab + π<1 mas on cells from stage 0.
#   sbatch scans/streams/1_physical_cuts.slurm
#   sbatch --export=ALL,SPLIT=val --array=0-9 scans/streams/1_physical_cuts.slurm

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
  scans/streams/1_physical_cuts.py
  --split "${SPLIT}"
  --array-task "${SLURM_ARRAY_TASK_ID}"
  --out-root "${OUT_ROOT}"
)
if [[ "${REBUILD}" == "1" ]]; then
  ARGS+=(--rebuild)
fi

python "${ARGS[@]}"
echo "Done stage 1 ${SPLIT} task ${SLURM_ARRAY_TASK_ID}"
