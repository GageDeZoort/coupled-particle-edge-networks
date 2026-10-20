#!/bin/bash
# AdamW + constant LR · t_epoch sweep — L=4 D=256, η₀=0.1, B=256×1, 40k steps.
#
#   Compare vs panel-(c) Adam η₀=0.1 @ B_global=256 (edge-b64-g4-s40000).
#   Graph OP: all-seed star kin7 + ΔR-kNN noloop, (k★,R★)=(7,0.1)
#   t_epoch ∈ {0.1, 0.5, 1, 5, 10, 25, 50}
#   n_train = 100M  (prescribed D for AdamW λ = √width·B/(t_epoch·η₀·n_train);
#                    stream pool = full JetClass train; 40k×256=10.24M presentations
#                    < 100M ⇒ unique, no wrap)
#   tag     = aw-tepoch-t{T}-n100M-b256-g1-s40000-dh32-k7-R0p1
#
# Prefer the watchdog (2h a100 slots, resume last.ckpt):
#   bash scans/jets/start_adamw_tepoch_D256_1a100_watchdogs.sh
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-aw-tepoch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --array=0-6
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_aw_tepoch_s40k_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_aw_tepoch_s40k_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
T_EPOCHS=(0.1 0.5 1 5 10 25 50)
if (( TASK < 0 || TASK >= ${#T_EPOCHS[@]} )); then
  echo "array index $TASK out of range (need 0-$(( ${#T_EPOCHS[@]} - 1 )))" >&2
  exit 1
fi

T_EPOCH=${T_EPOCHS[$TASK]}
ETA0=0.1
DEPTH=4
WIDTH=256
D_HEAD=32
HEAD=$((WIDTH / D_HEAD))
STAR_R=0.1
KNN_K=7
BATCH=256
N_GPUS=1
MAX_STEPS="${MAX_STEPS:-40000}"
SEED="${SEED:-0}"
N_VAL=25000
# Prescribed D = full JetClass train (drives AdamW t_epoch λ). Stream is unique
# for this budget because presentations = max_steps×B ≪ 100M.
N_TRAIN=100000000
EPOCHS=1
PRESENTATIONS=$((MAX_STEPS * BATCH * N_GPUS))
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
T_TAG=$(python3 -c "e=float('${T_EPOCH}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="aw-tepoch-t${T_TAG}-n100M-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}"

if (( PRESENTATIONS >= N_TRAIN )); then
  echo "ERROR: presentations=$PRESENTATIONS would wrap n_train=$N_TRAIN" >&2
  exit 1
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}  TASK=$TASK  partition=a100  wall=2h"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=adamw-tepoch-L4D256-s40k-1a100  L=$DEPTH  D=$WIDTH  H=$HEAD"
echo "model=capen-llama-att  eta0=$ETA0  optimizer=adamw  scheduler=none  t_epoch=$T_EPOCH"
echo "k=$KNN_K  R=$STAR_R  seed=$SEED  (noloop; no --live-knn-include-self)"
echo "B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))  max_steps=$MAX_STEPS"
echo "n_train=$N_TRAIN  (prescribed D=100M for t_epoch λ)  presentations=$PRESENTATIONS  (unique, no wrap)"
echo "run_tag=$RUN_TAG  t_tag=$T_TAG"

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
PY

# Guard: n_train=100M must stream (not RAM-subset).
python - <<'PY'
from cpen.apps.jets.jetclass_stream_datamodule import RAM_SUBSET_LIMIT
n = 100_000_000
if n <= RAM_SUBSET_LIMIT:
    raise SystemExit(f"n_train={n} would ram-subset (limit={RAM_SUBSET_LIMIT})")
print(f"[guard] n_train={n} > RAM_SUBSET_LIMIT={RAM_SUBSET_LIMIT} → STREAM")
PY

# Canonical OP: all-seed star kin7 + ΔR-kNN noloop.
python test_cpen_jetclass.py \
  --model capen-llama-att \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --jetclass-edge-features part-interaction \
  --live-knn-k "$KNN_K" \
  --star-radius "$STAR_R" \
  --live-star-kin7 \
  --mode sweep_lr \
  --optimizer adamw \
  --scheduler none \
  --t-epoch "$T_EPOCH" \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --n-train "$N_TRAIN" \
  --n-test 0 \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
  --dropout 0.0 \
  --root "$ROOT" \
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

echo "done t_epoch=$T_EPOCH task=$TASK max_steps=$MAX_STEPS n_train=$N_TRAIN"
