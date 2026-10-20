#!/bin/bash
# TopTagging FULL FT probe: AdamW t_epoch=2 + cosine→0.01×, η₀=0.05.
# gputest slice (time≤59m): B=256×4. Lightning resumes last.ckpt across slices.
#
#   sbatch scans/jets/finetune_toptagging_L4cos_adamw_t2_gputest.slurm
#   # or via watchdog:
#   bash scans/jets/start_ft_L4cos_adamw_t2_gputest_watchdogs.sh
#
#SBATCH --account=bhanin
#SBATCH --job-name=tt-ft-aw-t2-e0p05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=ailab
#SBATCH --qos=gpu-test
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/logs/tt_ft_aw_t2_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/logs/tt_ft_aw_t2_%j.err

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
ETA0="${ETA0:-0.05}"
DEPTH="${DEPTH:-4}"
WIDTH="${WIDTH:-256}"
D_HEAD="${D_HEAD:-32}"
HEAD=$((WIDTH / D_HEAD))
BATCH="${BATCH:-256}"
N_GPUS="${N_GPUS:-4}"
NUM_PARTICLES="${NUM_PARTICLES:-80}"
STAR_R="${STAR_R:-0.1}"
KNN_K="${KNN_K:-7}"
EPOCHS="${EPOCHS:-10}"
T_EPOCH="${T_EPOCH:-2}"
LR_DECAY_MIN_FRAC="${LR_DECAY_MIN_FRAC:-0.01}"
N_VAL="${N_VAL:-20000}"
SEED="${SEED:-0}"
ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
MIN_TAG=$(python3 -c "e=float('${LR_DECAY_MIN_FRAC}'); print(f'{e:g}'.replace('.','p'))")
T_TAG=$(python3 -c "e=float('${T_EPOCH}'); print(f'{e:g}'.replace('.','p'))")
# Short tag — NAME_MAX; include g4 so it does not collide with 1-GPU arms.
RUN_TAG="${RUN_TAG:-aw-t${T_TAG}-c${MIN_TAG}-e${ETA_TAG}-b${BATCH}x${N_GPUS}-ep${EPOCHS}}"

case "$FT_ROOT" in
  *jetclass_pretrain*|*jetclass_output*|*jetclass_scaling_law*)
    echo "ERROR: FT_ROOT=$FT_ROOT points at JetClass pretrain tree" >&2
    exit 1
    ;;
esac
case "$INIT_FROM" in
  *jetclass_pretrain/jetclass_output*)
    echo "ERROR: refuse INIT_FROM under live pretrain sweep tree: $INIT_FROM" >&2
    exit 1
    ;;
esac
test -f "$INIT_FROM" || { echo "missing INIT_FROM=$INIT_FROM"; exit 1; }

echo "PWD=$(pwd)"
echo "Host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "init_from=$INIT_FROM (read-only seed)"
echo "FULL FT AdamW t=${T_EPOCH} +cosine L=$DEPTH D=$WIDTH H=$HEAD η₀=$ETA0 B=${BATCH}×${N_GPUS} epochs=$EPOCHS"
echo "lr_decay_min_frac=$LR_DECAY_MIN_FRAC graph R=$STAR_R k=$KNN_K run_tag=$RUN_TAG"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export N_GPUS

python - <<PY
import os, sys, torch
if not torch.cuda.is_available():
    sys.exit("No CUDA")
need = int(os.environ.get("N_GPUS", "4"))
n = torch.cuda.device_count()
if n < need:
    sys.exit(f"Need {need} GPU(s); visible={n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    gib = p.total_memory / 1024**3
    print(f"GPU{i}={p.name} {gib:.1f} GiB")
# B=256/GPU needs ~80GB historically; warn loudly on MIG/40GB.
gib0 = torch.cuda.get_device_properties(0).total_memory / 1024**3
if gib0 < 70.0:
    print(
        f"[warn] GPU0 has {gib0:.1f} GiB — B=256/GPU may OOM on MIG; "
        "watchdog can retry with BATCH=128 if this slice fails.",
        flush=True,
    )
PY

python - <<PY
from pathlib import Path
p = Path("${INIT_FROM}")
assert p.is_file(), p
assert "init_ckpts" in str(p), f"INIT_FROM must live under init_ckpts/: {p}"
print(f"[ft-aw-t2] init seed ok mode={oct(p.stat().st_mode & 0o777)}")
PY

# Rough WD sanity (uses probed n_train at runtime; this is illustrative).
python - <<PY
import math
B, g, t, eta, n, D = ${BATCH}, ${N_GPUS}, float("${T_EPOCH}"), float("${ETA0}"), 1_211_000, ${WIDTH}
lam = math.sqrt(D) * (B * g) / (t * eta * n)
print(f"[ft-aw-t2] expected λ≈{lam:.4g}  (B_eff={B*g}, t_epoch={t}, η₀={eta})")
PY

set +e
python test_cpen_toptagging_ft.py \
  --model capen-llama-att \
  --operators incidence \
  --init-from "$INIT_FROM" \
  --toptagging-live \
  --star-radius "$STAR_R" \
  --live-knn-k "$KNN_K" \
  --live-star-kin7 \
  --jetclass-edge-features part-interaction \
  --num-particles "$NUM_PARTICLES" \
  --mode sweep_lr \
  --optimizer adamw \
  --scheduler cosine \
  --lr-decay-min-frac "$LR_DECAY_MIN_FRAC" \
  --t-epoch "$T_EPOCH" \
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
  --checkpoint-every-n-steps 250 \
  --checkpoint-monitor val_roc_auc \
  --checkpoint-mode max \
  --no-torch-compile
rc=$?
set -e

python - <<PY
import csv
from pathlib import Path
ft_root = Path("${FT_ROOT}")
run_tag = "${RUN_TAG}"
epochs = int("${EPOCHS}")
sweep = ft_root / "toptagging_output" / "adamw" / "sweep_lr"
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
        print(f"[ft-aw-t2] marked FT_EPOCHS_DONE in {d} (epoch={best})")
print(f"[ft-aw-t2] post-check best_epoch={best} target_last={epochs-1} train_rc={${rc}}")
PY

if [[ "$rc" -ne 0 ]]; then
  echo "train exited rc=$rc (gputest slice may have been preempted/OOM — watchdog will retry)" >&2
  exit "$rc"
fi
echo "done FT AdamW t=$T_EPOCH η₀=$ETA0 B=${BATCH}×${N_GPUS} epochs=$EPOCHS"
