#!/bin/bash
# Probe max per-GPU batch for L=4 D=256 live CAPEN on **1×A100** (non-ailab).
# Worst-case geometry for panel (a): k=10, R★=0.1 (most edges).
# Array: BATCHES=(96 128 160 192 224 256); each task tries a few train steps.
#
#   sbatch scans/jets/probe_L4D256_batch_1a100.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-bprobe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:25:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_bprobe_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_bprobe_%A_%a.err

set -euo pipefail
module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

BATCHES=(96 128 160 192 224 256)
TASK=${SLURM_ARRAY_TASK_ID:-0}
BATCH=${BATCHES[$TASK]}
KNN_K=10
STAR_R=0.1
MAX_STEPS=20

echo "probe L4/D256 1×A100 B_gpu=$BATCH k=$KNN_K R=$STAR_R steps=$MAX_STEPS"
nvidia-smi -L || true
python - <<PY
import torch
assert torch.cuda.is_available()
p = torch.cuda.get_device_properties(0)
print(f"GPU0 {p.name} {p.total_memory/1024**3:.1f} GiB")
PY

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas 0.25 \
  --epochs 1 \
  --max-steps "$MAX_STEPS" \
  --n-train 100000 \
  --n-test 0 \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth 4 --width 256 --heads 8 \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law \
  --run-tag "bprobe-1a100-k${KNN_K}-b${BATCH}" \
  --batch-size "$BATCH" \
  --n-gpus 1 \
  --num-workers 2 \
  --normalization uniform \
  --attention-normalization gamma \
  --num-particles 80 \
  --n-val 2000 \
  --seed 0 \
  --log-every-n-steps 5 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 100 \
  --checkpoint-every-n-steps 100 \
  --no-torch-compile
rc=$?
set -e
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits || true
if [[ $rc -eq 0 ]]; then
  echo "RESULT B=$BATCH OK"
else
  echo "RESULT B=$BATCH FAIL rc=$rc"
fi
exit $rc
