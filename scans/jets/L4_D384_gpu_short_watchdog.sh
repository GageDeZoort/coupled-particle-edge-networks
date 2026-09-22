#!/bin/bash
# Resubmit L4 D=384 B=128×4 a100/gpu-short (1.5h) chain until TARGET steps.
#
# Prefer:
#   bash scans/jets/start_L4D384_gpu_short_watchdogs.sh
# Or one arm:
#   ETA0=0.01 TARGET=100000 SLEEP_SEC=900 nohup bash scans/jets/L4_D384_gpu_short_watchdog.sh &
#
set -uo pipefail

ETA0="${ETA0:-0.1}"
ETA_TAG=$(python3 -c "e=float('${ETA0}'); s=f'{e:g}'.replace('.','p'); print(s)")
if [[ "$ETA_TAG" == "0p1" ]]; then
  JOB_NAME="${JOB_NAME:-jc-L4D384-a100}"
  RUN_TAG="${RUN_TAG:-L4D384-a100-b128-g4}"
else
  JOB_NAME="${JOB_NAME:-jc-L4D384-a100-e${ETA_TAG}}"
  RUN_TAG="${RUN_TAG:-L4D384-a100-b128-g4-e${ETA_TAG}}"
fi

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
SLURM=/home/jdezoort/coupled-particle-edge-networks/scans/jets/scaling_law_L4_D384_b128_s100k_gpu_short.slurm
LOG_CANDIDATES=(
  "$ROOT/logs/watchdog_L4D384_gpu_short_e${ETA_TAG}.log"
  "${HOME}/.cache/jc_L4D384_watchdogs/watchdog_L4D384_gpu_short_e${ETA_TAG}.log"
)
TARGET="${TARGET:-100000}"
MAX_RESUBMITS="${MAX_RESUBMITS:-200}"
# 1.5h wall → poll every 15m
SLEEP_SEC="${SLEEP_SEC:-900}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
LOG=""
for cand in "${LOG_CANDIDATES[@]}"; do
  mkdir -p "$(dirname "$cand")" 2>/dev/null || true
  if ( : >>"$cand" ) 2>/dev/null; then
    LOG="$cand"
    break
  fi
done
if [[ -z "$LOG" ]]; then
  echo "ERROR: cannot write watchdog log" >&2
  exit 1
fi

RUN_GLOB="$ROOT/jetclass_output/adam/sweep_lr/capen-llama-att_4_384_${ETA_TAG}_*${RUN_TAG}*"

current_step() {
  "$PYTHON" - "$RUN_GLOB" <<'PY'
import glob, sys
from pathlib import Path
pats = sys.argv[1]
cands = sorted(glob.glob(pats))
best = 0

def step_from_ckpt(ckpt: Path) -> int:
    try:
        import torch
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        return int(payload.get("global_step", 0) or 0)
    except Exception:
        return 0

for c in cands:
    p = Path(c)
    if p.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(p, columns=["global_step"])
            col = t.column("global_step")
            if len(col):
                best = max(best, int(col[-1].as_py()))
        except Exception:
            pass
        continue
    if not p.is_dir():
        continue
    for ckpt in p.glob("last*.ckpt"):
        best = max(best, step_from_ckpt(ckpt))
    pq = p.parent / f"{p.name}.parquet"
    if pq.exists():
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(pq, columns=["global_step"])
            col = t.column("global_step")
            if len(col):
                best = max(best, int(col[-1].as_py()))
        except Exception:
            pass
print(best)
PY
}

latest_val() {
  "$PYTHON" - "$RUN_GLOB" <<'PY'
import glob, sys
from pathlib import Path
pats = sys.argv[1]
rows = []
for c in sorted(glob.glob(pats)):
    p = Path(c)
    pq = p if p.suffix == ".parquet" else p.parent / f"{p.name}.parquet"
    if not pq.exists():
        continue
    try:
        import pyarrow.parquet as pq
        df = pq.read_table(pq).to_pandas()
        sub = df.dropna(subset=["val_loss"]) if "val_loss" in df.columns else df.iloc[0:0]
        if len(sub):
            r = sub.iloc[-1]
            rows.append(
                (
                    int(r["global_step"]),
                    float(r["val_loss"]),
                    float(r.get("val_roc_auc", float("nan"))),
                    float(r.get("val_acc", float("nan"))),
                )
            )
    except Exception:
        pass
if not rows:
    print("none")
else:
    s, loss, roc, acc = max(rows, key=lambda x: x[0])
    print(f"step={s} val_loss={loss:.4f} val_roc={roc:.4f} val_acc={acc:.4f}")
PY
}

job_running() {
  squeue -u jdezoort -n "$JOB_NAME" -h -o '%i' 2>/dev/null | head -1
}

n_subs=0
echo "$(date -Iseconds) watchdog start ETA0=$ETA0 ETA_TAG=$ETA_TAG JOB_NAME=$JOB_NAME RUN_TAG=$RUN_TAG TARGET=$TARGET MAX_RESUBMITS=$MAX_RESUBMITS RUN_GLOB=$RUN_GLOB" | tee -a "$LOG"

while true; do
  step=$(current_step)
  val=$(latest_val)
  jid=$(job_running || true)
  echo "$(date -Iseconds) step=$step $val job=${jid:-none} n_subs=$n_subs" | tee -a "$LOG"

  if (( step >= TARGET )); then
    echo "$(date -Iseconds) TARGET $TARGET reached (step=$step); exiting" | tee -a "$LOG"
    exit 0
  fi

  if [[ -n "${jid:-}" ]]; then
    sleep "$SLEEP_SEC"
    continue
  fi

  if (( n_subs >= MAX_RESUBMITS )); then
    echo "$(date -Iseconds) hit MAX_RESUBMITS=$MAX_RESUBMITS at step=$step; exiting" | tee -a "$LOG"
    exit 0
  fi

  out=$(sbatch \
    --job-name="$JOB_NAME" \
    --exclude=della-l02g6,della-l03g11 \
    --export=ALL,ETA0="$ETA0",RUN_TAG="$RUN_TAG",MAX_STEPS=100000,BATCH=128 \
    "$SLURM" 2>&1) || {
    echo "$(date -Iseconds) sbatch failed: $out" | tee -a "$LOG"
    sleep "$SLEEP_SEC"
    continue
  }
  n_subs=$((n_subs + 1))
  echo "$(date -Iseconds) submitted ($n_subs): $out" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
