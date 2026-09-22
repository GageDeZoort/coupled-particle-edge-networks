#!/bin/bash
# Wing points for the η₀ transfer curve: η₀ ∈ {0.01, 1.0} at both sizes.
# Core grid η∈{0.05,0.1,0.25,0.5} already running as 14102731_[0-7].
#
# Array:
#   0  L=3 D=128  η₀=0.01
#   1  L=3 D=128  η₀=1.0
#   2  L=5 D=512  η₀=0.01
#   3  L=5 D=512  η₀=1.0
#
#   sbatch scans/jets/scaling_law_transfer_eta_knn6_5k_wings.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-xfer-eta-knn6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --qos=gpu-test
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:59:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_xfer_eta_knn6_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_xfer_eta_knn6_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
case "$TASK" in
  0) DEPTH=3; WIDTH=128; HEAD=4;  ETA0=0.01 ;;
  1) DEPTH=3; WIDTH=128; HEAD=4;  ETA0=1.00 ;;
  2) DEPTH=5; WIDTH=512; HEAD=16; ETA0=0.01 ;;
  3) DEPTH=5; WIDTH=512; HEAD=16; ETA0=1.00 ;;
  *)
    echo "array index $TASK out of range (need 0-3)" >&2
    exit 1
    ;;
esac

BATCH="${BATCH:-128}"
STAR_R=0.2
KNN_K=6
MAX_STEPS="${MAX_STEPS:-5000}"
N_TRAIN=$((MAX_STEPS * BATCH))
N_VAL="${N_VAL:-25000}"
SEED="${SEED:-0}"
D_HEAD=$((WIDTH / HEAD))
if (( WIDTH % HEAD != 0 )); then
  echo "width $WIDTH not divisible by heads $HEAD" >&2
  exit 1
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK  label=xfer+partint+knn${KNN_K}+star${STAR_R}+eta${ETA0}+L${DEPTH}+D${WIDTH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "study=transfer-eta-knn6  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "eta0=$ETA0  starR=$STAR_R  knn=$KNN_K  edge=part-interaction"
echo "n_train=$N_TRAIN  max_steps=$MAX_STEPS  B=$BATCH"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python test_cpen_jetclass.py \
  --model capen-llama-att \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --jetclass-edge-features part-interaction \
  --live-knn-k "$KNN_K" \
  --star-radius "$STAR_R" \
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs 1 \
  --max-steps "$MAX_STEPS" \
  --n-train "$N_TRAIN" \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law \
  --run-tag xfer-5k-b${BATCH} \
  --batch-size "$BATCH" \
  --n-gpus 1 \
  --num-workers 4 \
  --normalization uniform \
  --attention-normalization gamma \
  --num-particles 80 \
  --n-val "$N_VAL" \
  --seed "$SEED" \
  --log-every-n-steps 100 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 1000 \
  --checkpoint-every-n-steps 500 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile
