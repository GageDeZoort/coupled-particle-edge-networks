#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=build-jetclass-lite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1500G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_jetclass_lite_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_jetclass_lite_%j.err

# Build JetClassLite particle archives + star-R hypergraph caches.
#
# Pipeline (one job):
#   1) Stratified sample from official ROOT splits (default 5M / 250k / 1M)
#      -> {CACHE_ROOT}/lite/n{NUM_PARTICLES}/{train,val,test}.pt
#   2) Materialize star-R graphs
#      -> {CACHE_ROOT}/processed/star-r{R}/n{NUM_PARTICLES}/{train,val,test}.pt
#
# Raw ROOT root (must contain train_100M/, val_5M/, test_20M/):
#   /tigress/jdezoort/jetclass
#
# Full 5M train stacks graphs in RAM before torch.save (same pattern as
# TopTagging). Prefer a high-mem CPU node; do not run on a login node.
#
# Submit from repo root:
#   sbatch scans/jets/build_jetclass_lite.slurm
#
# Smoke test (small subsample):
#   sbatch --export=ALL,N_TRAIN=50000,N_VAL=10000,N_TEST=10000 \
#     scans/jets/build_jetclass_lite.slurm
#
# Overrides:
#   sbatch --export=ALL,STAR_RADIUS=0.15,NUM_PARTICLES=128 \
#     scans/jets/build_jetclass_lite.slurm
#   sbatch --export=ALL,BUILD_LITE=0 scans/jets/build_jetclass_lite.slurm
#     # star graphs only (requires existing lite archives)

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
SOURCE_ROOT="${SOURCE_ROOT:-/tigress/jdezoort/jetclass}"
CACHE_ROOT="${CACHE_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets/jetclass}"
STAR_RADIUS="${STAR_RADIUS:-0.15}"
NUM_PARTICLES="${NUM_PARTICLES:-128}"
DATA_SEED="${DATA_SEED:-42}"
REBUILD="${REBUILD:-1}"
BUILD_LITE="${BUILD_LITE:-1}"

# Optional count overrides (empty => Lite defaults 5M / 250k / 1M when BUILD_LITE=1).
N_TRAIN="${N_TRAIN:-}"
N_VAL="${N_VAL:-}"
N_TEST="${N_TEST:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONNOUSERSITE=1

mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs "${CACHE_ROOT}"

cd "${REPO_ROOT}"
echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Mem: ${SLURM_MEM_PER_NODE:-?} MB"
echo "source-root: ${SOURCE_ROOT}"
echo "cache-root: ${CACHE_ROOT}"
echo "star-R radius: ${STAR_RADIUS}"
echo "num-particles: ${NUM_PARTICLES}"
echo "build-lite: ${BUILD_LITE}"
echo "rebuild: ${REBUILD}"
echo "n-train: ${N_TRAIN:-default}"
echo "n-val: ${N_VAL:-default}"
echo "n-test: ${N_TEST:-default}"

if [[ ! -d "${SOURCE_ROOT}/train_100M" ]]; then
  echo "ERROR: missing ${SOURCE_ROOT}/train_100M (JetClass ROOT splits)"
  exit 1
fi

ARGS=(
  scans/jets/build_jetclass_lite_star_graphs.py
  --source-root "${SOURCE_ROOT}"
  --cache-root "${CACHE_ROOT}"
  --radius "${STAR_RADIUS}"
  --num-particles "${NUM_PARTICLES}"
  --data-seed "${DATA_SEED}"
  --stage all
)

if [[ "${BUILD_LITE}" == "1" ]]; then
  ARGS+=(--build-lite)
fi
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

python "${ARGS[@]}"

echo "Done."
echo "  Lite archives: ${CACHE_ROOT}/lite/n${NUM_PARTICLES}/"
echo "  Star caches:   ${CACHE_ROOT}/processed/star-r*/n${NUM_PARTICLES}/"
echo "Train with: python scans/jets/test_cpen_jetclass.py --star-radius ${STAR_RADIUS} --data-root ${CACHE_ROOT} ..."
