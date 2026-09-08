#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=download-mnist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=download_mnist_%j.out
#SBATCH --error=download_mnist_%j.err

# Download PyG MNISTSuperpixels. Compute nodes often lack outbound DNS;
# if this fails, run the python command on a login node.
#
#   python scans/mnist/download_mnist_graphs.py \
#     --data-root /scratch/gpfs/BHANIN/jdezoort/datasets
#   sbatch scans/mnist/download_mnist_graphs.slurm
#   sbatch scans/mnist/build_mnist_graphs.slurm

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets}"

cd "${REPO_ROOT}"
echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unset}"
echo "data-root: ${DATA_ROOT}"

python scans/mnist/download_mnist_graphs.py --data-root "${DATA_ROOT}"
echo "Done. Raw data under ${DATA_ROOT}/MNISTSuperpixels/"
echo "Next: sbatch scans/mnist/build_mnist_graphs.slurm"
