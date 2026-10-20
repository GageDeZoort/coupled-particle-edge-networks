#!/bin/bash
# Keep exactly ONE ailab slot busy: L=8 D=512 η₀=0.1 AdamW t=10 n_train=100M
# attn pool, const LR, B=128×4, to 40k steps. Resubmit until done.
set -uo pipefail

REPO=/home/jdezoort/coupled-particle-edge-networks
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
LOGS="$ROOT/logs"
SLURM="$REPO/scans/jets/scaling_law_hxfer_L8_d512_adamw_t10_n100M_ailab.slurm"
TARGET="${TARGET:-40000}"
ETA0="${ETA0:-0.1}"
SLEEP_SEC="${SLEEP_SEC:-120}"
MAX_ITERS="${MAX_ITERS:-3000}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
TAG="hxfer-L8-d512-ap-aw-t10-n100M-s40000-k6-R0p1"
DEPTH=8
WIDTH=512
JOB_PREFIX="jc-awL8d512ap"
SWEEP="$ROOT/jetclass_output/adamw/sweep_lr"
mkdir -p "$LOGS" "$SWEEP"

eta_tag() {
  python3 -c "e=float('$1'); print(f'{e:g}'.replace('.','p'))"
}

JOB_NAME="${JOB_PREFIX}-e$(eta_tag "$ETA0")"

cell_step() {
  local et
  et=$(eta_tag "$ETA0")
  "$PYTHON" - "$SWEEP" "$LOGS" "$DEPTH" "$WIDTH" "$et" "$TAG" "$JOB_PREFIX" <<'PY'
import glob, re, sys
from pathlib import Path
root, logs, depth, width, et, tag, prefix = sys.argv[1:8]
best = 0
for c in glob.glob(f"{root}/*_{depth}_{width}_{et}_*{tag}*"):
    p = Path(c)
    pq = p if p.suffix == ".parquet" else Path(str(p) + ".parquet")
    if pq.is_file():
        try:
            import pyarrow.parquet as pqmod
            df = pqmod.read_table(pq, columns=["global_step"]).to_pandas()
            if len(df):
                best = max(best, int(df["global_step"].max()))
        except Exception:
            pass
    # Lightning ckpt fallback (parquet may lag)
    for ck in ("last.ckpt", "best.ckpt"):
        ckpt = p / ck if p.is_dir() else None
        if ckpt is None or not ckpt.is_file():
            continue
        try:
            import torch
            obj = torch.load(ckpt, map_location="cpu", weights_only=False)
            gs = obj.get("global_step")
            if gs is not None:
                best = max(best, int(gs))
        except Exception:
            pass
# Log fallback only when a live run dir/parquet exists — archived logs must
# not mark the cell done after a WD=0 (or other) re-run archive.
live = [
    c
    for c in glob.glob(f"{root}/*_{depth}_{width}_{et}_*{tag}*")
    if Path(c).is_dir() or Path(c).suffix == ".parquet"
]
if live:
    for f in sorted(Path(logs).glob(f"{prefix}_e{et}_*.out")):
        try:
            text = f.read_text(errors="ignore")[-200_000:]
        except Exception:
            continue
        for m in re.finditer(r"global_step=(\d+)", text):
            best = max(best, int(m.group(1)))
        for m in re.finditer(r"step=(\d+)", text):
            best = max(best, int(m.group(1)))
print(best)
PY
}

job_queued() {
  timeout 30 squeue -u jdezoort -n "$JOB_NAME" -h 2>/dev/null | grep -q .
}

# Count only THIS job's arrays on ailab (never >1).
our_ailab_count() {
  timeout 30 squeue -u jdezoort -p ailab -h -o "%j" 2>/dev/null \
    | grep -c "^${JOB_PREFIX}" || true
}

submit() {
  local et
  et=$(eta_tag "$ETA0")
  sbatch --parsable \
    --job-name="$JOB_NAME" \
    --output="$LOGS/${JOB_PREFIX}_e${et}_%j.out" \
    --error="$LOGS/${JOB_PREFIX}_e${et}_%j.err" \
    --export=ALL,ETA0="$ETA0",SEED=0,MAX_STEPS="$TARGET" \
    "$SLURM" 2>&1 || true
}

log() { echo "$(date -Is) $*"; }

log "START single-slot ailab L8/D512 η₀=$ETA0 attn target=$TARGET B=128×4"

for iter in $(seq 1 "$MAX_ITERS"); do
  step=$(cell_step)
  if (( step >= TARGET )); then
    log "DONE step=$step >= $TARGET"
    exit 0
  fi
  if job_queued; then
    log "iter=$iter hold Q/R step=$step name=$JOB_NAME"
    sleep "$SLEEP_SEC"
    continue
  fi
  n=$(our_ailab_count)
  n=${n:-0}
  if (( n >= 1 )); then
    log "iter=$iter wait (ailab already has $n ${JOB_PREFIX}*) step=$step"
    sleep "$SLEEP_SEC"
    continue
  fi
  jid=$(submit)
  if [[ "$jid" =~ ^[0-9]+$ ]]; then
    log "SUBMIT step=$step → $jid ($JOB_NAME)"
  else
    log "FAIL submit: $jid"
  fi
  sleep "$SLEEP_SEC"
done
log "FAIL MAX_ITERS"
exit 1
