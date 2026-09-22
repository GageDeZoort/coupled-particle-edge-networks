#!/bin/bash
# BaselineTransformer JetClass edge ablation: identical CompleteP / Llama / class-token
# pool to capen-llama-att, but M11 only (no 12/21/22, no live star graph).
#
# 59 min → gpu-test. last.ckpt every 200 steps. Resubmitting the same array
# task resumes weights and skips already-seen jets.
#
# Grid (14 tasks, arch slow, η0 fast):
#   (L,D,h) = (3,256,8), (6,512,16)
#   η0      = 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5
#
#   sbatch scans/jets/test_baseline_transformer.slurm
#   sbatch --job-name=jc-bt-10M --export=N_TRAIN=10000000,EPOCHS=3 \
#     --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_bt_10M_%A_%a.out \
#     --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_bt_10M_%A_%a.err \
#     scans/jets/test_baseline_transformer.slurm
#   sbatch --array=0-6 --job-name=jc-bt-5M-w1024 \
#     --export=N_TRAIN=5000000,EPOCHS=3,BATCH_SIZE=128,DEPTH=4,WIDTH=1024,HEADS=32 \
#     --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_bt_5M_w1024_%A_%a.out \
#     --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_bt_5M_w1024_%A_%a.err \
#     scans/jets/test_baseline_transformer.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-bt-1M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0-13
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_bt_1M_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_bt_1M_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets

ARCHS=(
  "3 256 8"
  "6 512 16"
)
ETAS=(0.01 0.05 0.1 0.25 0.5 1.0 2.5)

TASK=${SLURM_ARRAY_TASK_ID:-0}
N_ARCH=${#ARCHS[@]}
N_ETA=${#ETAS[@]}
ENV_DEPTH="${DEPTH:-}"
ENV_WIDTH="${WIDTH:-}"
ENV_HEADS="${HEADS:-}"

if [[ -n "$ENV_DEPTH" && -n "$ENV_WIDTH" && -n "$ENV_HEADS" ]]; then
  if (( TASK < 0 || TASK >= N_ETA )); then
    echo "array index $TASK out of range (need 0-$((N_ETA - 1)) for single-arch)" >&2
    exit 1
  fi
  DEPTH="$ENV_DEPTH"
  WIDTH="$ENV_WIDTH"
  HEADS="$ENV_HEADS"
  ETA0=${ETAS[$TASK]}
else
  ARCH_IDX=$((TASK / N_ETA))
  ETA_IDX=$((TASK % N_ETA))
  if (( ARCH_IDX < 0 || ARCH_IDX >= N_ARCH || ETA_IDX < 0 || ETA_IDX >= N_ETA )); then
    echo "array index $TASK out of range (need 0-$((N_ARCH * N_ETA - 1)))" >&2
    exit 1
  fi
  read -r DEPTH WIDTH HEADS <<< "${ARCHS[$ARCH_IDX]}"
  ETA0=${ETAS[$ETA_IDX]}
fi

N_TRAIN="${N_TRAIN:-1000000}"
N_VAL="${N_VAL:-25000}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SEED="${SEED:-0}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "eta0=$ETA0  arch=L${DEPTH}_D${WIDTH}_h${HEADS}  n_train=$N_TRAIN  epochs=$EPOCHS  batch=$BATCH_SIZE"
echo "model=baseline-transformer  graph=none (M11 only)  features=kin7"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python test_cpen_jetclass.py \
  --model baseline-transformer \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs "$EPOCHS" \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEADS" \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs \
  --batch-size "$BATCH_SIZE" \
  --n-gpus 4 \
  --num-workers 4 \
  --normalization uniform \
  --attention-normalization gamma \
  --num-particles 80 \
  --n-train "$N_TRAIN" \
  --n-val "$N_VAL" \
  --seed "$SEED" \
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --checkpoint-every-n-steps 200 \
  --checkpoint-monitor val_roc_auc \
  --checkpoint-mode max \
  --no-torch-compile
