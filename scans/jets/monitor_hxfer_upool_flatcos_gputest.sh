#!/bin/bash
# Light status logger for hxfc gputest campaign (companion to watchdog).
set -uo pipefail
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
LOG="$ROOT/logs/jc_hxfc_monitor.log"
SWEEP="$ROOT/jetclass_output/adam/sweep_lr"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
mkdir -p "$ROOT/logs"

while true; do
  {
    echo "=== $(date -Is) ==="
    squeue -u jdezoort -o '%.18i %.9P %.28j %.8T %.10M' 2>/dev/null | grep -E 'JOBID|hxfc' || echo '(no hxfc in squeue)'
    "$PYTHON" - <<'PY'
import glob
from pathlib import Path
import pyarrow.parquet as pq
root = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/jetclass_output/adam/sweep_lr")
rows = []
for w, tgt in ((128, 40000), (256, 20000)):
    tag = f"hxfer-d{w}-fc-k6-R0p1"
    for et in ["0p01", "0p05", "0p1", "0p25", "0p5", "1"]:
        step = 0
        best = None
        for p in sorted(root.glob(f"*_{w}_{et}_*{tag}*")):
            pqpath = p if p.suffix == ".parquet" else Path(str(p) + ".parquet")
            if not pqpath.is_file():
                continue
            try:
                df = pq.read_table(pqpath, columns=["global_step", "val_roc_auc"]).to_pandas()
                if len(df):
                    step = max(step, int(df["global_step"].max()))
                    sub = df.dropna(subset=["val_roc_auc"])
                    if len(sub):
                        best = float(sub["val_roc_auc"].max())
            except Exception:
                pass
        if step or w == 128 or et in {"0p01", "0p05", "0p1"}:
            frac = step / tgt
            flag = "DONE" if step >= tgt else ("run" if step else "pend")
            br = f"{best:.5f}" if best is not None else "-"
            print(f"  D={w} eta={et:4s} step={step:5d}/{tgt} ({frac:5.1%}) best={br} {flag}")
PY
    # recent job failures
    for f in $(ls -t "$ROOT"/logs/jc_hxfc_D128_e*.out 2>/dev/null | head -3); do
      if grep -qE 'FAIL|Error|Traceback|OOM|CANCELLED' "$f" 2>/dev/null; then
        echo "  WARN $(basename "$f"): $(grep -E 'FAIL|Error|Traceback|OOM|CANCELLED' "$f" | tail -1 | cut -c1-120)"
      fi
      step=$(grep -oE 'global_step=[0-9]+' "$f" 2>/dev/null | tail -1 || true)
      echo "  log $(basename "$f"): ${step:-no_step}"
    done
    echo
  } >>"$LOG" 2>&1
  sleep 300
done
