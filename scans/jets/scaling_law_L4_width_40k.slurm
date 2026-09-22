#!/bin/bash
# Extend the L=4 η₀=0.25 unique-pass width family from ~19.4k to 40k steps.
#
#   n = 40k × 512 = 20.48M unique jets (still one pass of JetClass 100M)
#   Seeds last.ckpt from the finished D10M-s19p532k runs so we do not replay.
#
#   sbatch scans/jets/scaling_law_L4_width_40k.slurm
#   sbatch --array=3 scans/jets/scaling_law_L4_width_40k.slurm  # D=512 only
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-scale-L4W40k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --qos=gpu-short
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_scale_L4W40k_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_scale_L4W40k_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets

mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

WIDTHS=(64 128 256 512)
HEADS=(2 4 8 16)
TASK=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK < 0 || TASK >= ${#WIDTHS[@]} )); then
  echo "array index $TASK out of range (need 0-$((${#WIDTHS[@]} - 1)))" >&2
  exit 1
fi
WIDTH=${WIDTHS[$TASK]}
HEAD=${HEADS[$TASK]}
ETA0="${ETA0:-0.25}"
N_TRAIN="${N_TRAIN:-20480000}"
N_VAL="${N_VAL:-25000}"
SEED="${SEED:-0}"
BATCH=512
N_GPUS=1
GLOBAL_BATCH=$((BATCH * N_GPUS))
MAX_STEPS=$(( (N_TRAIN + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))
DEPTH=4
D_HEAD=$((WIDTH / HEAD))
if (( WIDTH % HEAD != 0 )); then
  echo "width $WIDTH not divisible by heads $HEAD" >&2
  exit 1
fi

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
SWEEP="$ROOT/jetclass_output/adam/sweep_lr"

# Must match scans/common/sweep_common.py::_format_count so the seed lands
# in the same run dir Lightning will open (e.g. 20480000 -> ntr20480k).
NTR_TAG=$(python - <<PY
n = int(${N_TRAIN})
for scale, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
    if n >= scale:
        value = n / scale
        text = f"{value:g}".replace(".", "p")
        print(f"{text}{suffix}")
        break
else:
    print(n)
PY
)
echo "n_train tag for run dir: ntr${NTR_TAG}"

seed_last_ckpt() {
  local width="$1"
  local src dst
  src=$(find "$SWEEP" -maxdepth 1 -type d -name "capen-llama-att_4_${width}_0p25_*D10M-s19p532k-seed0-scale" | head -n 1 || true)
  # Prefer the trainer's exact ntr tag so resume hits the same directory.
  dst=$(find "$SWEEP" -maxdepth 1 -type d -name "capen-llama-att_4_${width}_0p25_*ntr${NTR_TAG}_*D20p48M-s40k-seed0-scale" | head -n 1 || true)
  if [[ -z "${src:-}" || ! -f "$src/last.ckpt" ]]; then
    echo "no D10M last.ckpt to seed for D=$width" >&2
    return 0
  fi
  if [[ -z "${dst:-}" ]]; then
    dst="${src/ntr10M/ntr${NTR_TAG}}"
    dst="${dst/D10M-s19p532k/D20p48M-s40k}"
    mkdir -p "$dst"
  fi
  # Always (re)seed from the finished 10M unique-pass ckpt when dst is missing
  # or clearly not a continuation (>= ~19k). Same specs => same run dir => resume.
  local need_seed=1
  if [[ -f "$dst/last.ckpt" ]]; then
    local step
    step=$(python - <<PY
import torch
ck = torch.load(r"$dst/last.ckpt", map_location="cpu", weights_only=False)
print(int(ck.get("global_step", -1)))
PY
)
    if (( step >= 19000 )); then
      need_seed=0
      echo "dest last.ckpt already present at step=$step: $dst/last.ckpt"
    else
      echo "dest last.ckpt at step=$step is too early; reseeding from D10M"
      rm -f "$dst/last.ckpt" "$dst/last-v1.ckpt" "$dst/best.ckpt"
    fi
  fi
  if (( need_seed )); then
    echo "seeding $dst/last.ckpt from $src/last.ckpt"
    cp -a "$src/last.ckpt" "$dst/last.ckpt"
  fi
}

seed_last_ckpt "$WIDTH"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "study=scaling-law-40k  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "eta0=$ETA0  n_train=$N_TRAIN  max_steps=$MAX_STEPS  B_gpu=$BATCH  n_gpus=$N_GPUS  B=$GLOBAL_BATCH"
echo "val_check_interval=1000  root=jetclass_scaling_law  run_tag=scale"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python test_cpen_jetclass.py \
  --model capen-llama-att \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --jetclass-edge-features logdot-dp \
  --star-radius 0.2 \
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs 1 \
  --max-steps "$MAX_STEPS" \
  --n-train "$N_TRAIN" \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law \
  --run-tag scale \
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
  --checkpoint-every-n-steps 200 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile
