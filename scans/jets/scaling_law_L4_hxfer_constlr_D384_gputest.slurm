#!/bin/bash
# Hparam-xfer Arm 2b: const-LR AdamW t-sweep at D=384 on Della gpu-test.
#
# Matches Lambda Arm 2 geometry / WD recipe, different host packing:
#   CAPEN k★=12 R★=0.1, L=4 D=384 H=12 (d_h=32), η₀=0.25
#   B_gpu=128 × 4 → B_eff=512 (same as Lambda 64×8)
#   --scheduler none, max_steps=10000, AdamW t ∈ {1,5,10}
#   prescribed n_train=100M for t_epoch WD
#
# Array: task 0→t=1, 1→t=5, 2→t=10
#
# Prefer the pool watchdog (59m walltime; resume until 10k)::
#   bash scans/jets/start_hxfer_constlr_D384_gputest_watchdogs.sh
#
# One-shot (may need requeue)::
#   sbatch scans/jets/scaling_law_L4_hxfer_constlr_D384_gputest.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-hxfer-c384
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
# Della routes time<=59m onto gputest; do NOT set partition=gputest.
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_hxfer_c384_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_hxfer_c384_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
T_EPOCHS=(1 5 10)
if (( TASK < 0 || TASK >= ${#T_EPOCHS[@]} )); then
  echo "array index $TASK out of range (need 0-2)" >&2
  exit 1
fi
T_EPOCH=${T_EPOCHS[$TASK]}

DEPTH=4
WIDTH=384
HEAD=12
D_HEAD=32
ETA0="${ETA0:-0.25}"
BATCH="${BATCH:-128}"
N_GPUS=4
MAX_STEPS="${MAX_STEPS:-10000}"
N_TRAIN="${N_TRAIN:-100000000}"
N_VAL="${N_VAL:-25000}"
SEED="${SEED:-0}"
STAR_R=0.1
KNN_K=12
ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
T_TAG=$(python3 -c "e=float('${T_EPOCH}'); print(f'{e:g}'.replace('.','p'))")
# Same tag family as Lambda Arm 2, but b128-g4 packing.
RUN_TAG="hxfer-constlr-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-k${KNN_K}-R${R_TAG}-adamw-t${T_TAG}"

if (( WIDTH % HEAD != 0 || WIDTH / HEAD != D_HEAD )); then
  echo "D=$WIDTH H=$HEAD does not give d_h=$D_HEAD" >&2
  exit 1
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=hxfer-constlr-D384-gputest  L=$DEPTH  D=$WIDTH  H=$HEAD  d_h=$D_HEAD"
echo "optimizer=adamw  t_epoch=$T_EPOCH  eta0=$ETA0  scheduler=none"
echo "k=$KNN_K  R=$STAR_R  B_gpu=$BATCH  n_gpus=$N_GPUS  B_eff=$((BATCH*N_GPUS))"
echo "max_steps=$MAX_STEPS  n_train=$N_TRAIN (prescribed for WD)  run_tag=$RUN_TAG"

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
  --model capen-llama-att \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --jetclass-edge-features part-interaction \
  --live-knn-k "$KNN_K" \
  --star-radius "$STAR_R" \
  --mode sweep_lr \
  --optimizer adamw \
  --scheduler none \
  --t-epoch "$T_EPOCH" \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs 1 \
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
  --log-every-n-steps 100 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 1000 \
  --checkpoint-every-n-steps 500 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile
