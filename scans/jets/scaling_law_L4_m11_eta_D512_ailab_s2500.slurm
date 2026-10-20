#!/bin/bash
# Panel (c) M11-only η₀×D — D=512 on Della **ailab**.
#
# Twin of gputest D=256/384 from scaling_law_L4_m11_eta_gputest_s2500.slurm.
# Same tag so collectors / canvas see one family.
#
#   B_gpu=96 × 4 → B_global=384, Adam, 2500 steps, n_train=1e6, epochs=1
#   η₀ ∈ {0.01, 0.05, 0.1, 0.25, 0.5, 1.0}  (array 0–5)
#
#   sbatch scans/jets/scaling_law_L4_m11_eta_D512_ailab_s2500.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-m11-a512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=ailab
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=1:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_m11_s2500_ailab_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_m11_s2500_ailab_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
WIDTH=512
ETAS=(0.01 0.05 0.1 0.25 0.5 1.0)
if (( TASK < 0 || TASK >= ${#ETAS[@]} )); then
  echo "array index $TASK out of range (need 0-5)" >&2
  exit 1
fi

ETA0=${ETAS[$TASK]}
D_HEAD=32
HEAD=$((WIDTH / D_HEAD))
DEPTH=4
MODEL=baseline-transformer
N_TRAIN=1000000
N_VAL=25000
SEED="${SEED:-0}"
BATCH=96
N_GPUS=4
MAX_STEPS="${MAX_STEPS:-2500}"
EPOCHS="${EPOCHS:-1}"
ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="m11-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK  partition=ailab"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=m11-eta-panel-c-s2500-ailab-D512  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "model=$MODEL  (M11 all-to-all, no edges)  eta0=$ETA0  optimizer=adam  seed=$SEED"
echo "B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))  max_steps=$MAX_STEPS  epochs=$EPOCHS"
echo "run_tag=$RUN_TAG"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
print(f"n_cuda={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i} {p.name} {p.total_memory/1024**3:.1f} GiB")
if torch.cuda.device_count() < 4:
    sys.exit(f"Need 4 GPUs; visible count={torch.cuda.device_count()}")
PY

python test_cpen_jetclass.py \
  --model "$MODEL" \
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
  --max-steps "$MAX_STEPS" \
  --n-train "$N_TRAIN" \
  --n-test 0 \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law \
  --run-tag "$RUN_TAG" \
  --batch-size "$BATCH" \
  --n-gpus "$N_GPUS" \
  --num-workers 4 \
  --normalization uniform \
  --attention-normalization gamma \
  --num-particles 80 \
  --n-val "$N_VAL" \
  --seed "$SEED" \
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 500 \
  --checkpoint-every-n-steps 500 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile
