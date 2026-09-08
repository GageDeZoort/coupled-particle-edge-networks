#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=download-pascal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=download_pascal_%j.out
#SBATCH --error=download_pascal_%j.err

# Download PascalVOC-SP from LRGB/PyG (Dropbox). Requires outbound network.
#
# Run this once before build_pascal_graphs.slurm. On many clusters compute
# nodes block external DNS; if sbatch fails with "Name or service not known",
# run the same python command interactively on a login node instead:
#
#   python scans/pascal/download_pascal_graphs.py \
#     --data-root /scratch/gpfs/BHANIN/jdezoort/datasets
#
# Submit from repo root:
#   sbatch scans/pascal/download_pascal_graphs.slurm
#
# Then build the CPEN cache:
#   sbatch scans/pascal/build_pascal_graphs.slurm

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets}"
SPLITS="${SPLITS:-train,val,test}"

cd "${REPO_ROOT}"
echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "data-root: ${DATA_ROOT}"
echo "splits: ${SPLITS}"

python scans/pascal/download_pascal_graphs.py \
  --data-root "${DATA_ROOT}" \
  --splits "${SPLITS}"

echo "Done. Raw PascalVOC-SP data under ${DATA_ROOT}/pascalvoc-sp/"
echo "Next: sbatch scans/pascal/build_pascal_graphs.slurm"
