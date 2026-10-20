#!/bin/bash
# Panel (b′) R★-sweep — CompleteP v2, ΔR-kNN **noloop**, k★=7.
#
#   L=4 D=256 H=8 · η₀=0.1 · Adam · B=128 · 2500 steps
#   R★ ∈ {0.06, 0.08, 0.1, 0.12, 0.14} × 3 shards × 2 seeds = 30 cells
#   Tag: v2cpNL
#
#   sbatch scans/jets/scaling_law_L4_graph_R_noloop_k7_1a100_v2cp.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-gR-noloop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:05:00
#SBATCH --array=0-29%8
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gR_noloop_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gR_noloop_%A_%a.err

set -euo pipefail
module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/graph_construction_completep_v2/R_k7_noloop

TASK=${SLURM_ARRAY_TASK_ID:-0}
RADII=(0.06 0.08 0.1 0.12 0.14)
N_SHARD=3
N_INIT=2
N_PER_R=$((N_SHARD * N_INIT))

if (( TASK < 0 || TASK >= ${#RADII[@]} * N_PER_R )); then
  echo "array index $TASK out of range" >&2
  exit 1
fi

R_IDX=$((TASK / N_PER_R))
REM=$((TASK % N_PER_R))
SHARD=$((REM / N_INIT))
INIT=$((REM % N_INIT))
STAR_R=${RADII[$R_IDX]}

KNN_K=7
ETA0=0.1
WIDTH=256
HEAD=8
DEPTH=4
SEED=$INIT
BATCH=128
N_GPUS=1
MAX_STEPS=2500
N_TRAIN=$((MAX_STEPS * BATCH * N_GPUS))
TRAIN_SKIP=$((SHARD * N_TRAIN))
RUN_TAG="graph-R-noloop-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-k${KNN_K}-shard${SHARD}-seed${SEED}-v2cpNL"

echo "study=graph-R-noloop-completep-v2  L=$DEPTH D=$WIDTH k=$KNN_K R=$STAR_R (no self)"
echo "eta0=$ETA0 shard=$SHARD seed=$SEED n_train=$N_TRAIN tag=$RUN_TAG"
nvidia-smi -L || true

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
  --train-skip-jets "$TRAIN_SKIP" \
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
  --n-val 25000 \
  --seed "$SEED" \
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 500 \
  --checkpoint-every-n-steps 500 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile

echo "done R=$STAR_R k=$KNN_K noloop shard=$SHARD seed=$SEED"
