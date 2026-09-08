#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=build-qm9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_qm9_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_qm9_%j.err

# Incidence-v1 caches for QM9, standard benchmark split (110k train / 10k val /
# 10,831 test) from a seed-42 permutation.
#
#   {data-root}/processed/qm9/incidence-v1/{train,val,test}.pt
#
# Graphs are RDKit bond graphs with hydrogens; edge features are the bond-type
# one-hot plus interatomic distance. All 19 PyG target columns are stored, so
# --qm9-target switches property without a rebuild.
#
# Train must be built first (it writes the standardization statistics that
# val/test reuse); the build script enforces the order.
#
# PYG_ROOT reuses an already-downloaded torch_geometric QM9 root, e.g. the
# hp-transfer-gts copy at /scratch/gpfs/BHANIN/jgdezoort/datasets.
#
#   sbatch scans/qm9/build_qm9_graphs.slurm
#   PYG_ROOT=/scratch/gpfs/BHANIN/jgdezoort/datasets sbatch scans/qm9/build_qm9_graphs.slurm

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets}"
SPLITS="${SPLITS:-train,val,test}"
REBUILD="${REBUILD:-1}"
PYG_ROOT="${PYG_ROOT:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONNOUSERSITE=1
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs

cd "${REPO_ROOT}"
echo "PWD=$(pwd)  host=$(hostname)  job=${SLURM_JOB_ID:-unset}"
echo "data-root=${DATA_ROOT}  splits=${SPLITS}  rebuild=${REBUILD}  pyg-root=${PYG_ROOT:-<data-root>/QM9}"

ARGS=(scans/qm9/build_qm9_graphs.py --data-root "${DATA_ROOT}" --splits "${SPLITS}")
if [[ "${REBUILD}" == "1" ]]; then
  ARGS+=(--rebuild)
fi
if [[ -n "${PYG_ROOT}" ]]; then
  ARGS+=(--pyg-root "${PYG_ROOT}")
fi
python "${ARGS[@]}"
echo "Done. Caches under ${DATA_ROOT}/processed/qm9/incidence-v1/"
