#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=build-pascal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_pascal_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_pascal_%j.err

# Pre-build PascalVOC-SP incidence-v2 caches for CPEN / CAPEN-Llama training.
# Defaults overwrite existing caches and bake WIRE coordinates (m=8) into the
# primary {split}.pt used by the Lightning datamodule.
#
# Download raw LRGB splits first (once):
#   sbatch scans/pascal/download_pascal_graphs.slurm
# or on a login node if compute nodes have no outbound network.
#
# Writes:
#   {data-root}/processed/pascalvoc-sp/incidence-v2/train.pt
#   {data-root}/processed/pascalvoc-sp/incidence-v2/val.pt
#   {data-root}/processed/pascalvoc-sp/incidence-v2/test.pt
#
# Submit from repo root:
#   sbatch scans/pascal/build_pascal_graphs.slurm
#
# Overrides:
#   sbatch --export=ALL,DATA_ROOT=/scratch/gpfs/BHANIN/jdezoort/datasets scans/pascal/build_pascal_graphs.slurm
#   sbatch --export=ALL,WIRE_COORDINATE_DIM=16 scans/pascal/build_pascal_graphs.slurm
#   sbatch --export=ALL,WIRE_COORDINATE_DIM=0 scans/pascal/build_pascal_graphs.slurm  # no WIRE
#   sbatch --export=ALL,SPLITS=train scans/pascal/build_pascal_graphs.slurm

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets}"
SPLITS="${SPLITS:-train,val,test}"
REBUILD="${REBUILD:-1}"
WIRE_COORDINATE_DIM="${WIRE_COORDINATE_DIM:-8}"
WIRE_COMBINATORIAL="${WIRE_COMBINATORIAL:-0}"

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
echo "splits: ${SPLITS}"
echo "rebuild: ${REBUILD}"
echo "wire-coordinate-dim: ${WIRE_COORDINATE_DIM}"
echo "wire-combinatorial: ${WIRE_COMBINATORIAL}"

ARGS=(
  scans/pascal/build_pascal_graphs.py
  --data-root "${DATA_ROOT}"
  --splits "${SPLITS}"
  --wire-coordinate-dim "${WIRE_COORDINATE_DIM}"
)

if [[ "${REBUILD}" == "1" ]]; then
  ARGS+=(--rebuild)
fi
if [[ "${WIRE_COMBINATORIAL}" == "1" ]]; then
  ARGS+=(--wire-combinatorial-laplacian)
fi

python "${ARGS[@]}"
echo "Done. Caches under ${DATA_ROOT}/processed/pascalvoc-sp/incidence-v2/"
if [[ "${WIRE_COORDINATE_DIM}" -gt 0 ]]; then
  echo "WIRE m=${WIRE_COORDINATE_DIM} is baked into the primary {split}.pt files."
fi
