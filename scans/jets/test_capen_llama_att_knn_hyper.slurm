#!/bin/bash
# CAPEN-Llama-att JetClass transfer check: 1M jets, 1 epoch, 59 min → gpu-test.
#
#   M11      all-to-all particle attention (capen-llama-att)
#   M12/M21  live star-R incidence (JetClass has no hier/kNN+VN cache)
#   M22      hyperedges only (deg > 2)
#   readout  writeup class-token pool: independent X/E, 1/sqrt(2) fuse, per-class U
#
# Grid (14 tasks, arch slow, η0 fast):
#   (L,D,h) = (3,256,8), (6,512,16)
#   η0      = 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5
#
#   sbatch scans/jets/test_capen_llama_att_knn_hyper.slurm
#   sbatch --job-name=jc-att-rope --export=ALL,USE_ROPE=1,EPOCHS=1 \
#     --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_att_1M_rope_%A_%a.out \
#     --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_att_1M_rope_%A_%a.err \
#     scans/jets/test_capen_llama_att_knn_hyper.slurm
#   sbatch --job-name=jc-att-10M --export=ALL,N_TRAIN=10000000,EPOCHS=3,USE_ROPE=0 \
#     --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_att_10M_%A_%a.out \
#     --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_att_10M_%A_%a.err \
#     scans/jets/test_capen_llama_att_knn_hyper.slurm
#   sbatch --array=0-6 --job-name=jc-att-5M-w1024 \
#     --export=ALL,N_TRAIN=5000000,EPOCHS=3,USE_ROPE=0,BATCH_SIZE=128,DEPTH=4,WIDTH=1024,HEADS=32 \
#     --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_att_5M_w1024_%A_%a.out \
#     --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_att_5M_w1024_%A_%a.err \
#     scans/jets/test_capen_llama_att_knn_hyper.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-att-1M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0-13
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_att_1M_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_att_1M_%A_%a.err

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
USE_ROPE="${USE_ROPE:-0}"

ROPE_ARGS=()
if [[ "${USE_ROPE}" == "1" ]]; then
  ROPE_ARGS+=(--use-rope)
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "eta0=$ETA0  arch=L${DEPTH}_D${WIDTH}_h${HEADS}  n_train=$N_TRAIN  epochs=$EPOCHS  batch=$BATCH_SIZE"
echo "model=capen-llama-att  graph=star-R=0.2  features=kin7  use_rope=${USE_ROPE}"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python test_cpen_jetclass.py \
  --model capen-llama-att \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --jetclass-edge-features part-interaction \
  --star-radius 0.2 \
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
  --no-torch-compile \
  "${ROPE_ARGS[@]}"
