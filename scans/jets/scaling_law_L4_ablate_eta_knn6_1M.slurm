#!/bin/bash
# 1M unique-pass η₀ re-check on the winning edge stack.
#   Evidence the frozen η₀=0.25 (from pre-knn 1M) is too hot for knn6+star:
#   val CE spikes of +0.03–0.13 and D=512 lagging D=256 at matched S.
#
#   Stack: star-R=0.2 + kNN k=6 + part-interaction
#   L=4, d_head=32, p=80, 1×A100 gpu-test
#
# Array:
#   0  D=256  η₀=0.05   B=256
#   1  D=256  η₀=0.10   B=256
#   2  D=256  η₀=0.25   B=256   (control; resumes finished knnk run)
#   3  D=512  η₀=0.05   B=128
#   4  D=512  η₀=0.10   B=128
#   5  D=512  η₀=0.25   B=128
#
#   sbatch scans/jets/scaling_law_L4_ablate_eta_knn6_1M.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-ablate-eta-knn6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --qos=gpu-test
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:59:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_ablate_eta_knn6_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_ablate_eta_knn6_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
case "$TASK" in
  0) WIDTH=256; HEAD=8;  ETA0=0.05; BATCH=256 ;;
  1) WIDTH=256; HEAD=8;  ETA0=0.10; BATCH=256 ;;
  2) WIDTH=256; HEAD=8;  ETA0=0.25; BATCH=256 ;;
  3) WIDTH=512; HEAD=16; ETA0=0.05; BATCH=128 ;;
  4) WIDTH=512; HEAD=16; ETA0=0.10; BATCH=128 ;;
  5) WIDTH=512; HEAD=16; ETA0=0.25; BATCH=128 ;;
  *)
    echo "array index $TASK out of range (need 0-5)" >&2
    exit 1
    ;;
esac

DEPTH=4
STAR_R=0.2
KNN_K=6
N_TRAIN=1000000
N_VAL=25000
SEED=0
MAX_STEPS=$(( (N_TRAIN + BATCH - 1) / BATCH ))

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK  label=partint+knn${KNN_K}+star${STAR_R}+eta${ETA0}+D${WIDTH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "L=$DEPTH D=$WIDTH H=$HEAD eta0=$ETA0 starR=$STAR_R knn=$KNN_K"
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
  --run-tag ablate-eta-b${BATCH} \
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
  --no-torch-compile
