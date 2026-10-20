#!/bin/bash
# JetClass pretrain continue — L8/D1024 Adam η₀=0.1 on Della gputest.
#
# Resumes the paused Lambda giant run under jetclass_pretrain.
# Per-GPU batch matches Lambda (B=32); 4×A100 → B_eff=128 (Lambda was 32×8=256).
# One unique pass over n_train=10.24M @ B_eff=128 is max_steps=80000 (was 40k
# matching Lambda's B_eff=256 data budget — half a Della epoch).
#
#   bash scans/jets/start_pretrain_L8_continue_gputest_watchdogs.sh
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-pretrain-L8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/logs/jc_pretrain_L8_gt_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/logs/jc_pretrain_L8_gt_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain
SEED_CKPT="$ROOT/lambda_L8D1024/checkpoints/last.ckpt"
mkdir -p "$ROOT/logs"

ETA0=0.1
DEPTH=8
WIDTH=1024
D_HEAD=32
HEAD=$((WIDTH / D_HEAD))
STAR_R=0.1
KNN_K=7
BATCH=32
N_GPUS=4
# Epoch-2 at const η₀: 2 unique passes over 10.24M @ B_eff=128 → 160000 steps.
N_TRAIN=10240000
MAX_STEPS=160000
SEED=0
N_VAL=0
# Lightning epoch length is the stream period; max_steps is the stop.
EPOCHS=2
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="pretrain-L8D1024-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}-eta${ETA_TAG}-s160k"
# Prior Della continue (finished unique epoch @ 80k) — seed from its last.ckpt.
PREV_TAG="pretrain-L8D1024-b${BATCH}-g${N_GPUS}-s80000-dh${D_HEAD}-k${KNN_K}-R${R_TAG}-eta${ETA_TAG}-s80k"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "study=jetclass-pretrain-L8-continue  L=$DEPTH  D=$WIDTH  H=$HEAD  d_h=$D_HEAD"
echo "eta0=$ETA0  optimizer=adam  scheduler=none  k=$KNN_K  R=$STAR_R"
echo "B_gpu=$BATCH  n_gpus=$N_GPUS  B_eff=$((BATCH*N_GPUS))  max_steps=$MAX_STEPS"
echo "seed_ckpt=$SEED_CKPT  run_tag=$RUN_TAG  root=$ROOT"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
# Avoid expandable_segments on Della — hit CUDACachingAllocator INTERNAL ASSERT
# (!block->expandable_segment_) during L8 attention. Keep fragmentation knob only.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("No CUDA")
if torch.cuda.device_count() < 4:
    sys.exit(f"Need 4 GPUs; visible={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i} {p.name} {p.total_memory/1024**3:.1f} GiB")
PY

# Seed Della run dir from mirrored Lambda last.ckpt if needed.
python - <<PY
import json, shutil
from datetime import datetime
from pathlib import Path
import torch

root = Path("${ROOT}")
sweep = root / "jetclass_output/adam/sweep_lr"
seed = Path("${SEED_CKPT}")
run_tag = "${RUN_TAG}"
prev_tag = "${PREV_TAG}"
width = int("${WIDTH}")
eta_tag = "${ETA_TAG}"
assert seed.is_file(), f"missing Lambda mirror ckpt {seed} (rsync still running?)"

# Prefer the farthest-along Della continue ckpt (40k half-epoch), else Lambda seed.
best_src, best_step = seed, 0
for tag in (run_tag, prev_tag):
    for p in sweep.glob(f"*_{width}_{eta_tag}_*{tag}*"):
        if not p.is_dir() or (p / "SUPERSEDED.txt").exists():
            continue
        for ckpt in p.glob("last*.ckpt"):
            try:
                payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                step = int(payload.get("global_step", 0) or 0)
                if step >= best_step:
                    best_src, best_step = ckpt, step
            except Exception:
                pass
payload = torch.load(best_src, map_location="cpu", weights_only=False)
step = int(payload.get("global_step", 0) or 0)
print(f"[seed] source {best_src} global_step={step}")

cands = [
    p for p in sweep.glob(f"*_{width}_{eta_tag}_*{run_tag}*")
    if p.is_dir() and not (p / "SUPERSEDED.txt").exists()
]
if cands:
    dest = max(cands, key=lambda p: p.stat().st_mtime)
else:
    # Predicted name (stream tags + run_tag); training will use the same tagging.
    dest = sweep / (
        f"capen-llama-att_8_1024_0p1_ln_op-incidence_onorm-degree_res-transformer-like"
        f"_h32_ntr10240k_nval0_ntst0_nf7_gstar0p1_stream-kin7-partint-p80-knn7-noloop-sk7"
        f"-D10p24M-s80k-seed0-{run_tag}"
    )
    dest.mkdir(parents=True, exist_ok=True)

dst = dest / "last.ckpt"
if not dst.is_file() or best_step > 0:
    # Refresh if source is ahead of dest (40k → 80k epoch continue).
    dest_step = 0
    if dst.is_file():
        try:
            dest_step = int(
                torch.load(dst, map_location="cpu", weights_only=False).get(
                    "global_step", 0
                )
                or 0
            )
        except Exception:
            dest_step = 0
    if (not dst.is_file()) or (best_step > dest_step):
        shutil.copy2(best_src, dst)
        print(f"[seed] copied step={best_step} → {dst}")
    else:
        print(f"[seed] existing {dst} step={dest_step}")
else:
    cur = torch.load(dst, map_location="cpu", weights_only=False)
    print(f"[seed] existing {dst} step={int(cur.get('global_step', 0) or 0)}")

manifest = {
    "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
    "source_ckpt": str(best_src),
    "source_global_step": step,
    "dest_dir": str(dest),
    "run_tag": run_tag,
    "prev_tag": prev_tag,
    "batch_gpu": int("${BATCH}"),
    "n_gpus": int("${N_GPUS}"),
    "B_eff": int("${BATCH}") * int("${N_GPUS}"),
    "max_steps": int("${MAX_STEPS}"),
    "n_train": int("${N_TRAIN}"),
    "eta0": float("${ETA0}"),
    "scheduler": "none",
    "note": (
        "L8/D1024 Della continue → full unique epoch (80k steps @ B_eff=128). "
        "Seeded from 40k half-epoch ckpt when available."
    ),
}
(root / "manifests").mkdir(exist_ok=True)
(root / "manifests" / "L8D1024_della_continue_seed.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print(json.dumps(manifest, indent=2))
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
  --live-star-kin7 \
  --mode sweep_lr \
  --optimizer adam \
  --scheduler none \
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
  --num-workers 1 \
  --normalization uniform \
  --attention-normalization gamma \
  --num-particles 80 \
  --n-val "$N_VAL" \
  --limit-val-batches 0 \
  --seed "$SEED" \
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --checkpoint-every-n-steps 1000 \
  --no-torch-compile

echo "done L8 continue max_steps=$MAX_STEPS"
