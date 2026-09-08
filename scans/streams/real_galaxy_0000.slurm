#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=real-0000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=5:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/streams/real0000_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/streams/real0000_%j.err

# Mock-less train/0000: keep S5 members, drop injected mocks, same blob pipeline.
#   sbatch scans/streams/real_galaxy_0000.slurm
#   REBUILD=1 sbatch scans/streams/real_galaxy_0000.slurm

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
OUT_ROOT="${OUT_ROOT:-/scratch/gpfs/BHANIN/jgdezoort/streams/galaxies_real}"
REBUILD="${REBUILD:-0}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1

mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/streams
cd "${REPO_ROOT}"

echo "PWD=$(pwd)  Host=$(hostname)  Job=${SLURM_JOB_ID}"
echo "OUT_ROOT=${OUT_ROOT}  REBUILD=${REBUILD}"

ARGS=(
  scans/streams/run_real_pipeline.py
  --split train
  --galaxy-id 0000
  --include-tune-galaxy
  --out-root "${OUT_ROOT}"
)
if [[ "${REBUILD}" == "1" ]]; then
  ARGS+=(--rebuild)
fi

python "${ARGS[@]}"
echo "Done real pipeline train/0000"
