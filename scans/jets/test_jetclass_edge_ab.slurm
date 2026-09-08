#!/bin/bash
# Preprocessing A/B at fixed capacity and fixed D.
#
# Three arms x three seeds. Everything except the preprocessing is held fixed
# (CPEN, depth 4, width 256, 128 particles, R=0.15, D=1M, 30 epochs), so the
# only thing that moves is how particles and hyperedges are featurized:
#
#   cache          row-L2 node features + logdot-dp edges   (current baseline)
#   stream-logdot  affine-only node features + logdot-dp    (isolates row-L2)
#   stream-partint affine-only + ParT interaction edges      (isolates edges)
#
# Caveat: the cache arm draws its 1M jets from the stratified 5M lite archive
# while the streaming arms use the nested read plan. Both are class balanced and
# identically distributed, but they are not the same jets, so treat the cache
# arm as a distributional baseline rather than a paired comparison.
#
# Usage:  sbatch --array=0-8 scans/jets/test_jetclass_edge_ab.slurm
#SBATCH --account=bhanin
#SBATCH --job-name=jc-edge-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu80
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=2:30:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jc_edge_ab_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jc_edge_ab_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets

ARMS=(cache stream-logdot stream-partint)
SEEDS=(0 1 2)

TASK=${SLURM_ARRAY_TASK_ID:-0}
ARM=${ARMS[$((TASK / 3))]}
SEED=${SEEDS[$((TASK % 3))]}

RAW_ROOT=/projects/j/jdezoort/jetclass
CACHE_ROOT=/scratch/gpfs/BHANIN/jdezoort/datasets/jetclass
RUN_ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs

echo "Host: $(hostname)   arm=$ARM   seed=$SEED"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

COMMON=(
  --model cpen
  --operators incidence
  --operator-backend sparse
  --star-radius 0.15
  --mode sweep_lr
  --etas 0.25
  --optimizer adam
  --t-epoch 1
  --residual-structure transformer-like
  --layer-norm
  --epochs 30
  --depth 4 --width 256
  --num-particles 128
  --batch-size 256
  --n-gpus 1
  --num-workers 6
  --normalization uniform
  --n-train 1000000
  --n-val 250000
  --seed "$SEED"
  --root "$RUN_ROOT"
  --log-every-n-steps 200
  --check-val-every-n-epoch 1
  --no-torch-compile
)

case "$ARM" in
  cache)
    python test_cpen_jetclass.py "${COMMON[@]}" --data-root "$CACHE_ROOT"
    ;;
  stream-logdot)
    python test_cpen_jetclass.py "${COMMON[@]}" \
      --jetclass-stream \
      --jetclass-raw-root "$RAW_ROOT" \
      --data-root "$RAW_ROOT" \
      --jetclass-features full \
      --jetclass-edge-features logdot-dp
    ;;
  stream-partint)
    python test_cpen_jetclass.py "${COMMON[@]}" \
      --jetclass-stream \
      --jetclass-raw-root "$RAW_ROOT" \
      --data-root "$RAW_ROOT" \
      --jetclass-features full \
      --jetclass-edge-features part-interaction
    ;;
  *)
    echo "unknown arm: $ARM" >&2
    exit 1
    ;;
esac
