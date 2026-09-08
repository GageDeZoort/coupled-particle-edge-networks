#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=build-star-R
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# n=200 full train stacks >400 GB before torch.save (OOM'd at 400G); the cpu
# partition has 3-6 TB nodes, so give generous headroom. Baking WIRE (m=8)
# adds ~8 GB to the train payload but needs another pass over incidence.
#SBATCH --mem=1500G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_star_graphs_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_star_graphs_%j.err

# Pre-build TopTagging star-R hypergraph caches (optionally with baked WIRE).
#
# Full train split (~1.2M jets, n=128) needs hundreds of GB RAM while tensors
# are stacked before torch.save. Do not run on a login node.
#
# Defaults overwrite existing caches and bake wire_coordinates (m=8) into the
# primary {split}.pt used by the Lightning datamodule.
#
# Submit from repo root:
#   sbatch scans/jets/build_star_graphs.slurm
#
# Modes:
#   MODE=full      rebuild star graphs + WIRE (default; expensive)
#   MODE=wire-only augment existing star {split}.pt in place (cheap; no HDF5 rebuild)
#
# Overrides:
#   sbatch --export=ALL,STAR_RADIUS=0.15,NUM_PARTICLES=128 scans/jets/build_star_graphs.slurm
#   sbatch --export=ALL,MODE=wire-only,WIRE_COORDINATE_DIM=8 scans/jets/build_star_graphs.slurm
#   sbatch --export=ALL,N_TRAIN=50000 scans/jets/build_star_graphs.slurm   # smoke test
#   sbatch --export=ALL,WIRE_COORDINATE_DIM=0 scans/jets/build_star_graphs.slurm  # no WIRE

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets/toptagging}"
STAR_RADIUS="${STAR_RADIUS:-0.15}"
NUM_PARTICLES="${NUM_PARTICLES:-128}"
DATA_SEED="${DATA_SEED:-42}"
REBUILD="${REBUILD:-1}"
MODE="${MODE:-full}"
WIRE_COORDINATE_DIM="${WIRE_COORDINATE_DIM:-8}"
WIRE_COMBINATORIAL="${WIRE_COMBINATORIAL:-0}"

# Optional subsample limits (leave empty for full splits). Only used in MODE=full.
N_TRAIN="${N_TRAIN:-}"
N_VAL="${N_VAL:-}"
N_TEST="${N_TEST:-}"

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
echo "star-R radius: ${STAR_RADIUS}"
echo "num-particles: ${NUM_PARTICLES}"
echo "mode: ${MODE}"
echo "rebuild: ${REBUILD}"
echo "wire-coordinate-dim: ${WIRE_COORDINATE_DIM}"
echo "wire-combinatorial: ${WIRE_COMBINATORIAL}"

if [[ "${MODE}" == "wire-only" ]]; then
  if [[ "${WIRE_COORDINATE_DIM}" -le 0 ]]; then
    echo "ERROR: MODE=wire-only requires WIRE_COORDINATE_DIM > 0"
    exit 1
  fi
  ARGS=(
    scans/jets/build_toptagging_wire_coordinates.py
    --data-root "${DATA_ROOT}"
    --radius "${STAR_RADIUS}"
    --num-particles "${NUM_PARTICLES}"
    --wire-coordinate-dim "${WIRE_COORDINATE_DIM}"
  )
  if [[ "${WIRE_COMBINATORIAL}" == "1" ]]; then
    ARGS+=(--wire-combinatorial-laplacian)
  fi
  if [[ "${REBUILD}" == "1" ]]; then
    ARGS+=(--rebuild)
  fi
  python "${ARGS[@]}"
else
  ARGS=(
    scans/jets/build_toptagging_star_graphs.py
    --data-root "${DATA_ROOT}"
    --radius "${STAR_RADIUS}"
    --num-particles "${NUM_PARTICLES}"
    --data-seed "${DATA_SEED}"
  )
  if [[ -n "${N_TRAIN}" ]]; then
    ARGS+=(--n-train "${N_TRAIN}")
  fi
  if [[ -n "${N_VAL}" ]]; then
    ARGS+=(--n-val "${N_VAL}")
  fi
  if [[ -n "${N_TEST}" ]]; then
    ARGS+=(--n-test "${N_TEST}")
  fi
  if [[ "${REBUILD}" == "1" ]]; then
    ARGS+=(--rebuild)
  fi
  if [[ "${WIRE_COORDINATE_DIM}" -gt 0 ]]; then
    ARGS+=(--wire-coordinate-dim "${WIRE_COORDINATE_DIM}")
    if [[ "${WIRE_COMBINATORIAL}" == "1" ]]; then
      ARGS+=(--wire-combinatorial-laplacian)
    fi
  fi
  python "${ARGS[@]}"
fi

echo "Done. Caches under ${DATA_ROOT}/processed/star-r*/n${NUM_PARTICLES}/"
if [[ "${WIRE_COORDINATE_DIM}" -gt 0 ]]; then
  echo "WIRE m=${WIRE_COORDINATE_DIM} is baked into the primary {split}.pt files."
fi
