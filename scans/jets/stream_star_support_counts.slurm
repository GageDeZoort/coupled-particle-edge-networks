#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --job-name=star-S-jetstd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/star_support_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/star_support_%j.err

# Jet-level |S| mean ± std over Plot-1 train shards 0/1/2 (384k each).
# Regenerates Nature panel-(b) |S|/jet inset after CSV write.
#
#   sbatch scans/jets/stream_star_support_counts.slurm

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/matplotlib-$$
mkdir -p "${MPLCONFIGDIR}"

echo "Host=$(hostname) Job=${SLURM_JOB_ID} mem=${SLURM_MEM_PER_NODE:-?} cpus=${SLURM_CPUS_PER_TASK}"
date -Iseconds

rm -f paper/data/graph_geometry_star_shards.csv

/home/jdezoort/.conda/envs/mamba-env/bin/python -u scans/jets/stream_star_support_counts.py \
  --shards 0 1 2 \
  --batch-size 512 \
  --num-workers 2 \
  --max-jets 384000

/home/jdezoort/.conda/envs/mamba-env/bin/python -u scans/jets/plot_graph_construction_nature.py --no-collect

echo ALL_DONE
date -Iseconds
