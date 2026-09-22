#!/bin/bash
# Embedding-LR correction scan (off gputest):
#   L=4 D=256 H=8, η₀=0.5 fixed, B=512, 1×A100-80GB, qos=gpu-short (1.5h)
#   corr C ∈ {5, 10, 20}: encoder_{x,e} LR = C·η₀/√D; trunk/readout = η₀/√D
#   Same edge stack as the D384 transfer arms (star-R=0.2, knn=6, part-interaction).
#
#   sbatch scans/jets/scaling_law_L4_corr_embed_D256.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-corr-D256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-short
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_corr_D256_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_corr_D256_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
CORRS=(5 10 20)
if (( TASK < 0 || TASK >= ${#CORRS[@]} )); then
  echo "array index $TASK out of range (need 0-$((${#CORRS[@]} - 1)))" >&2
  exit 1
fi
CORR=${CORRS[$TASK]}

ETA0=0.5
WIDTH=256
HEAD=8
DEPTH=4
BATCH=512
N_GPUS=1
# ~1.5h at ~0.7–1.0 s/step on 80GB; walltime is the real stop.
MAX_STEPS="${MAX_STEPS:-10000}"
N_VAL="${N_VAL:-25000}"
SEED="${SEED:-0}"
STAR_R=0.2
KNN_K=6
RUN_TAG="corr-embed-b${BATCH}-g${N_GPUS}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "study=L4-D256-corr-embed  L=$DEPTH D=$WIDTH H=$HEAD d_head=$((WIDTH / HEAD))"
echo "eta0=$ETA0  corr=$CORR  starR=$STAR_R  knn=$KNN_K  edge=part-interaction"
echo "max_steps=$MAX_STEPS  B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python - <<'PY'
import os, sys, torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
props = torch.cuda.get_device_properties(0)
gib = props.total_memory / 1024**3
print(
    f"GPU0 {props.name} {gib:.1f} GiB  "
    f"count={torch.cuda.device_count()}  "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
)
if gib < 70:
    sys.exit(
        f"Need A100 80GB (#SBATCH --constraint=gpu80); this node has {gib:.1f} GiB."
    )
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
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --corr "$CORR" \
  --epochs 1 \
  --max-steps "$MAX_STEPS" \
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
