#!/bin/bash
# L=4 unique-pass width family with the winning edge stack:
#   star-R=0.2 hyperedges + directed ΔR-kNN k=6 + part-interaction
#   Compare to archived logdot-dp 40k width runs (no knn).
#
#   L=4, d_head=32 → (D,H)=(64,2),(128,4),(256,8),(512,16)
#   B=256 for D≤256; B=128 for D=512 (B=512 OOM'd with knn on 40GB A100;
#   D=512×B=256 is unproven and likely tight vs D=256×B=256 ~half-card).
#   max_steps=40000 → n = 40k × B unique jets (10.24M / 5.12M).
#   Fresh start (do not seed old logdot-dp ckpts).
#
#   sbatch scans/jets/scaling_law_L4_width_40k_knn6.slurm
#   sbatch --array=3 scans/jets/scaling_law_L4_width_40k_knn6.slurm  # D=512 only
#   BATCH=64 sbatch --array=3 ...   # emergency OOM fallback
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-scale-L4W40k-knn6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --qos=gpu-medium
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_scale_L4W40k_knn6_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_scale_L4W40k_knn6_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

WIDTHS=(64 128 256 512)
HEADS=(2 4 8 16)
TASK=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK < 0 || TASK >= ${#WIDTHS[@]} )); then
  echo "array index $TASK out of range (need 0-$((${#WIDTHS[@]} - 1)))" >&2
  exit 1
fi
WIDTH=${WIDTHS[$TASK]}
HEAD=${HEADS[$TASK]}
ETA0="${ETA0:-0.25}"
SEED="${SEED:-0}"
# Default batch by width; allow BATCH= override for OOM fallbacks.
if [[ -n "${BATCH:-}" ]]; then
  :
elif (( WIDTH >= 512 )); then
  BATCH=128
else
  BATCH=256
fi
N_GPUS=1
GLOBAL_BATCH=$((BATCH * N_GPUS))
MAX_STEPS="${MAX_STEPS:-40000}"
N_TRAIN=$((MAX_STEPS * GLOBAL_BATCH))
N_VAL="${N_VAL:-25000}"
DEPTH=4
STAR_R=0.2
KNN_K=6
D_HEAD=$((WIDTH / HEAD))
if (( WIDTH % HEAD != 0 )); then
  echo "width $WIDTH not divisible by heads $HEAD" >&2
  exit 1
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "study=scaling-law-40k-knn6  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "eta0=$ETA0  starR=$STAR_R  knn=$KNN_K  edge=part-interaction"
echo "n_train=$N_TRAIN  max_steps=$MAX_STEPS  B_gpu=$BATCH  n_gpus=$N_GPUS  B=$GLOBAL_BATCH"
echo "val_check_interval=1000  root=jetclass_scaling_law  run_tag=scale-knn6"

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
  --run-tag scale-knn6 \
  --batch-size "$BATCH" \
  --n-gpus "$N_GPUS" \
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
