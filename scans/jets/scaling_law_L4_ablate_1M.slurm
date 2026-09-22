#!/bin/bash
# Cheap 1M unique-pass ablations at the scaling-family point:
#   L=4, D=256, H=8, η₀=0.25, B=512, 1×A100, gpu-short
#
# Array:
#   0  control     star-R=0.2  class-token pool  hyperedge M22 (current default)
#   1  star-R=0.1
#   2  star-R=0.3
#   3  unipool     mask-uniform mean of decoded X,E (not energy, not class tokens)
#   4  full M22    relation-22 over every live edge (not hyperedge-only)
#
#   sbatch scans/jets/scaling_law_L4_ablate_1M.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-ablate-1M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --qos=gpu-short
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:15:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_ablate_1M_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_ablate_1M_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
STAR_R=0.2
EXTRA=()
case "$TASK" in
  0) LABEL=control ;;
  1) LABEL=star-R0.1; STAR_R=0.1 ;;
  2) LABEL=star-R0.3; STAR_R=0.3 ;;
  3) LABEL=unipool; EXTRA+=(--uniform-output-pool) ;;
  4) LABEL=fullm22; EXTRA+=(--all-edge-m22) ;;
  *)
    echo "array index $TASK out of range (need 0-4)" >&2
    exit 1
    ;;
esac

ETA0=0.25
WIDTH=256
HEAD=8
DEPTH=4
N_TRAIN=1000000
N_VAL=25000
SEED=0
BATCH=512
MAX_STEPS=$(( (N_TRAIN + BATCH - 1) / BATCH ))

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK  label=$LABEL"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "L=$DEPTH D=$WIDTH H=$HEAD eta0=$ETA0 starR=$STAR_R extra=${EXTRA[*]:-none}"
echo "n_train=$N_TRAIN max_steps=$MAX_STEPS B=$BATCH"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python test_cpen_jetclass.py \
  --model capen-llama-att \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --jetclass-edge-features part-interaction \
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
  --run-tag ablate \
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
  --checkpoint-every-n-steps 200 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile \
  "${EXTRA[@]+"${EXTRA[@]}"}"
