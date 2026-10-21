#!/bin/bash
# Typed-edge η₀ — L4/D256 on **ailab + qos=gpu-test** (H200 test slice).
#
# 1×GPU, B=256 (= gputest B_eff 64×4). time≤59m. Separate run_tag from the
# Della gputest b64-g4 dirs so checkpoints do not collide.
#
#   sbatch scans/jets/scaling_law_L4_typed_edge_eta_D256_k7_ailab_gputest.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-typed-ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ailab
#SBATCH --qos=gpu-test
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:59:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_typed_k7_ailab_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_typed_k7_ailab_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
ETAS=(0.01 0.05 0.1 0.25 0.5)
if (( TASK < 0 || TASK >= ${#ETAS[@]} )); then
  echo "array index $TASK out of range" >&2
  exit 1
fi

ETA0=${ETAS[$TASK]}
DEPTH=4
WIDTH=256
D_HEAD=32
HEAD=$((WIDTH / D_HEAD))
STAR_R=0.1
KNN_K=7
BATCH="${BATCH:-256}"
N_GPUS=1
MAX_STEPS="${MAX_STEPS:-40000}"
SEED="${SEED:-0}"
N_VAL=25000
N_TRAIN=$((MAX_STEPS * BATCH * N_GPUS))
EPOCHS=2
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="typed-edge-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}-s1-ailab"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}  TASK=$TASK  partition=ailab qos=gpu-test"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "typed-edge ailab-gputest  L=$DEPTH D=$WIDTH H=$HEAD eta0=$ETA0"
echo "B_gpu=$BATCH n_gpus=$N_GPUS B_global=$((BATCH*N_GPUS)) max_steps=$MAX_STEPS"
echo "run_tag=$RUN_TAG"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
p = torch.cuda.get_device_properties(0)
print(f"GPU0 {p.name} {p.total_memory/1024**3:.1f} GiB")
PY

set +e
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
  --typed-edge-streams \
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
rc=$?
set -e
echo "done typed-ailab eta0=$ETA0 rc=$rc"
exit 0
