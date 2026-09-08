#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --job-name=jc-cputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_jetclass_cputest_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/build_jetclass_cputest_%j.err

# One ~1h JetClassLite build piece for Della CPU QOS "test" (there is no
# "cputest" partition — use --qos=test, same idea as gputest).
#
# Prefer the helper:
#   bash scans/jets/submit_jetclass_lite_cputest.sh
#
# Or submit pieces manually, e.g.:
#   sbatch --export=ALL,STAGE=lite-shard,SPLITS=train,CLASS=0,REBUILD=1 \
#     scans/jets/build_jetclass_lite_cputest.slurm
#   sbatch --mem=1200G --export=ALL,STAGE=graphs,SPLITS=train,REBUILD=1 \
#     scans/jets/build_jetclass_lite_cputest.slurm
#
# Env knobs:
#   STAGE   = lite-shard | lite-merge | lite | graphs
#   SPLITS  = train | val | test | train,val | ...
#   CLASS   = 0..9          (required for lite-shard)
#   STAR_RADIUS (default 0.15)
#   REBUILD (default 1)

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
STAGE="${STAGE:?STAGE required (lite-shard|lite-merge|lite|graphs)}"
SPLITS="${SPLITS:?SPLITS required (e.g. train or val,test)}"
CLASS="${CLASS:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONNOUSERSITE=1

mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs "${CACHE_ROOT}"
cd "${REPO_ROOT}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "stage=${STAGE} splits=${SPLITS} class=${CLASS:-na}"
echo "star-R=${STAR_RADIUS} rebuild=${REBUILD}"
echo "mem=${SLURM_MEM_PER_NODE:-?} cpus=${SLURM_CPUS_PER_TASK}"

ARGS=(
  scans/jets/build_jetclass_lite_star_graphs.py
  --source-root "${SOURCE_ROOT}"
  --cache-root "${CACHE_ROOT}"
  --radius "${STAR_RADIUS}"
  --num-particles "${NUM_PARTICLES}"
  --data-seed "${DATA_SEED}"
  --stage "${STAGE}"
  --splits "${SPLITS}"
)
if [[ "${REBUILD}" == "1" ]]; then
  ARGS+=(--rebuild)
fi
if [[ "${STAGE}" == "lite-shard" ]]; then
  if [[ -z "${CLASS}" ]]; then
    echo "ERROR: CLASS=0..9 required for STAGE=lite-shard"
    exit 1
  fi
  ARGS+=(--lite-class-shard "${CLASS}")
fi

python "${ARGS[@]}"
echo "Done stage=${STAGE} splits=${SPLITS}."
