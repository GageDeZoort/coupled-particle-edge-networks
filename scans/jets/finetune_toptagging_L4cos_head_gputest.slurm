#!/bin/bash
# TopTagging head-only FT from frozen L4/D256 JetClass cosine ckpt.
#
# Attentional readout (not uniform pool). Backbone frozen via --freeze-backbone.
# INIT_FROM must be the immutable copy under toptagging_finetune/init_ckpts/.
# Default: B=256×1 on a single A100 (same global batch as former B=64×4).
#
#   ETA0=0.1 sbatch --export=ALL,ETA0 scans/jets/finetune_toptagging_L4cos_head_gputest.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=tt-ft-L4cos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/logs/tt_ft_L4cos_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/logs/tt_ft_L4cos_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"

FT_ROOT="${FT_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune}"
mkdir -p "$FT_ROOT/logs"

INIT_FROM="${INIT_FROM:-$FT_ROOT/init_ckpts/L4D256-cosine-s781250-frozen.ckpt}"
ETA0="${ETA0:?Set ETA0 (e.g. 0.1)}"
DEPTH="${DEPTH:-4}"
WIDTH="${WIDTH:-256}"
D_HEAD="${D_HEAD:-32}"
HEAD=$((WIDTH / D_HEAD))
BATCH="${BATCH:-256}"
N_GPUS="${N_GPUS:-1}"
NUM_PARTICLES="${NUM_PARTICLES:-80}"
STAR_R="${STAR_R:-0.1}"
KNN_K="${KNN_K:-7}"
EPOCHS="${EPOCHS:-10}"
N_VAL="${N_VAL:-20000}"
SEED="${SEED:-0}"
ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="${RUN_TAG:-ft-L${DEPTH}D${WIDTH}-head-apool-cos781k-eta${ETA_TAG}-b${BATCH}-g${N_GPUS}-ep${EPOCHS}-k${KNN_K}-R${R_TAG}}"

case "$FT_ROOT" in
  *jetclass_pretrain*|*jetclass_output*|*jetclass_scaling_law*)
    echo "ERROR: FT_ROOT=$FT_ROOT points at JetClass pretrain tree" >&2
    exit 1
    ;;
esac
case "$INIT_FROM" in
  *jetclass_pretrain/jetclass_output*)
    echo "ERROR: refuse INIT_FROM under live pretrain sweep tree: $INIT_FROM" >&2
    echo "Use the immutable copy under $FT_ROOT/init_ckpts/" >&2
    exit 1
    ;;
esac
test -f "$INIT_FROM" || { echo "missing INIT_FROM=$INIT_FROM"; exit 1; }

echo "PWD=$(pwd)"
echo "Host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "init_from=$INIT_FROM (read-only seed)"
echo "L=$DEPTH D=$WIDTH H=$HEAD η₀=$ETA0 B=${BATCH}×${N_GPUS} epochs=$EPOCHS"
echo "graph R=$STAR_R k=$KNN_K attentional-pool freeze-backbone run_tag=$RUN_TAG"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<PY
import sys, torch
if not torch.cuda.is_available():
    sys.exit("No CUDA")
need = int("${N_GPUS}")
if torch.cuda.device_count() < need:
    sys.exit(f"Need {need} GPU(s); visible={torch.cuda.device_count()}")
p = torch.cuda.get_device_properties(0)
print(f"n_cuda={torch.cuda.device_count()} GPU0={p.name} {p.total_memory/1024**3:.1f} GiB")
PY

# Never write back into the init ckpt (open it read-only as a sanity check).
python - <<PY
from pathlib import Path
p = Path("${INIT_FROM}")
assert p.is_file(), p
# Refuse if the path is still the live pretrain last.ckpt
assert "init_ckpts" in str(p), f"INIT_FROM must live under init_ckpts/: {p}"
print(f"[ft] init seed ok mode={oct(p.stat().st_mode & 0o777)} size={p.stat().st_size}")
PY

python test_cpen_toptagging_ft.py \
  --model capen-llama-att \
  --operators incidence \
  --init-from "$INIT_FROM" \
  --freeze-backbone \
  --toptagging-live \
  --star-radius "$STAR_R" \
  --live-knn-k "$KNN_K" \
  --live-star-kin7 \
  --jetclass-edge-features part-interaction \
  --num-particles "$NUM_PARTICLES" \
  --mode sweep_lr \
  --optimizer adam \
  --scheduler none \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs "$EPOCHS" \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
  --dropout 0.0 \
  --root "$FT_ROOT" \
  --run-tag "$RUN_TAG" \
  --batch-size "$BATCH" \
  --n-gpus "$N_GPUS" \
  --num-workers 4 \
  --normalization uniform \
  --attention-normalization gamma \
  --n-val "$N_VAL" \
  --seed "$SEED" \
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --checkpoint-every-n-steps 500 \
  --checkpoint-monitor val_roc_auc \
  --checkpoint-mode max \
  --no-torch-compile

# Mark complete if Lightning reached the final epoch (0-indexed → epochs-1).
python - <<PY
import csv, os, re
from pathlib import Path
ft_root = Path("${FT_ROOT}")
run_tag = "${RUN_TAG}"
epochs = int("${EPOCHS}")
sweep = ft_root / "toptagging_output" / "adam" / "sweep_lr"
best = -1
for d in sweep.glob(f"*{run_tag}*") if sweep.is_dir() else []:
    if not d.is_dir():
        continue
    for met in d.rglob("metrics.csv"):
        try:
            with met.open() as f:
                for row in csv.DictReader(f):
                    raw = row.get("epoch") or row.get("Epoch")
                    if raw not in (None, ""):
                        best = max(best, int(float(raw)))
        except Exception:
            pass
    if best >= epochs - 1:
        (d / "FT_EPOCHS_DONE").write_text(f"epoch={best} target={epochs}\n")
        print(f"[ft] marked FT_EPOCHS_DONE in {d} (epoch={best})")
print(f"[ft] post-check best_epoch={best} target_last={epochs-1}")
PY

echo "done FT L4cos head η₀=$ETA0 epochs=$EPOCHS"
