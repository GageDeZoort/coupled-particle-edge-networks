#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=s0-cells
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0:59:00
#SBATCH --array=0-18

# Stage 0: galaxy parquet → DES 5° cells.
# train (skip 0000): 19 galaxies → --array=0-18
# val: 10 → --array=0-9
# test: 10 → --array=0-9
#
#   sbatch scans/streams/0_preprocess_stream_df.slurm
#   sbatch --export=ALL,SPLIT=val --array=0-9 scans/streams/0_preprocess_stream_df.slurm
#   python scans/streams/0_preprocess_stream_df.py --split train --list

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

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}  array ${SLURM_ARRAY_TASK_ID}"
echo "SPLIT=${SPLIT}"

ARGS=(
  scans/streams/0_preprocess_stream_df.py
  --split "${SPLIT}"
  --array-task "${SLURM_ARRAY_TASK_ID}"
  --out-root "${OUT_ROOT}"
)
if [[ "${REBUILD}" == "1" ]]; then
  ARGS+=(--rebuild)
fi

python "${ARGS[@]}"
echo "Done stage 0 ${SPLIT} task ${SLURM_ARRAY_TASK_ID}"
