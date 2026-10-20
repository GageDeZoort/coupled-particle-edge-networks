#!/bin/bash
# Panel (a′) depth/width check — k-sweep **noloop** at L=8 D=1024.
#
#   Fair vs L4/D256 noloop: same jets (320k) and same steps (2500).
#   B_gpu=32 × 4 GPUs → B_global=128 · η₀=0.1 · R★=0.1
#   k ∈ {1,2,3,4,5,6,7,8,10} × 3 shards × 2 seeds = 54 cells
#   d_h=32 → H=32. Tag: v2cpNL8
#
#   sbatch scans/jets/scaling_law_L8_graph_k_noloop_4a100_v2cp.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-gk-nl-L8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
# L8/D1024 + denser kNN → request 2h (L4 noloop cells were ~10–28m on 1×A100)
#SBATCH --time=2:00:00
# Cap concurrency (4-GPU jobs); avoid QOSMaxJobsPerUserLimit stalls
#SBATCH --array=0-53%3
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gk_noloop_L8_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gk_noloop_L8_%A_%a.err

set -euo pipefail
module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/graph_construction_completep_v2/k_R0p1_noloop_L8D1024

TASK=${SLURM_ARRAY_TASK_ID:-0}
KS=(1 2 3 4 5 6 7 8 10)
N_SHARD=3
N_INIT=2
N_PER_K=$((N_SHARD * N_INIT))

if (( TASK < 0 || TASK >= ${#KS[@]} * N_PER_K )); then
  echo "array index $TASK out of range" >&2
  exit 1
fi

K_IDX=$((TASK / N_PER_K))
REM=$((TASK % N_PER_K))
SHARD=$((REM / N_INIT))
INIT=$((REM % N_INIT))
KNN_K=${KS[$K_IDX]}

STAR_R=0.1
ETA0=0.1
WIDTH=1024
D_HEAD=32
HEAD=$((WIDTH / D_HEAD))
DEPTH=8
SEED=$INIT
BATCH=32
N_GPUS=4
N_TRAIN=320000
MAX_STEPS=2500
if (( MAX_STEPS * BATCH * N_GPUS != N_TRAIN )); then
  echo "N_TRAIN=$N_TRAIN != MAX_STEPS*B_global=$((MAX_STEPS * BATCH * N_GPUS))" >&2
  exit 1
fi
TRAIN_SKIP=$((SHARD * N_TRAIN))
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="graph-k-noloop-L${DEPTH}-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-R${R_TAG}-shard${SHARD}-seed${SEED}-v2cpNL8"

echo "study=graph-k-noloop-L8D1024-completep-v2  L=$DEPTH D=$WIDTH H=$HEAD d_h=$D_HEAD"
echo "k=$KNN_K R=$STAR_R (no self) eta0=$ETA0 shard=$SHARD seed=$SEED"
echo "n_train=$N_TRAIN steps=$MAX_STEPS B_global=$((BATCH*N_GPUS)) tag=$RUN_TAG"
nvidia-smi -L || true

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import sys, torch
if torch.cuda.device_count() < 4:
    sys.exit(f"Need 4 GPUs; got {torch.cuda.device_count()}")
print(f"n_cuda={torch.cuda.device_count()}")
PY

# Do NOT pass --live-knn-include-self → true neighbors only.
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

echo "done L=$DEPTH D=$WIDTH k=$KNN_K noloop shard=$SHARD seed=$SEED"
