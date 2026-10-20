#!/bin/bash
# Launch D=128 remainder + D=256 full uniform-pool flat→cosine hxfer on gputest.
#
# Keeps Lambda D=128 η₀∈{0.01,0.05,0.1} @20k. Wipes incomplete D=128 η₀=0.5
# (killed mid-flat on Lambda) so that cell restarts cleanly. Watchdog then
# drives η₀∈{0.25,0.5,1} @ D=128 and full η₀×6 @ D=256 to 20k steps.
#
#   bash scans/jets/submit_hxfer_upool_flatcos_gputest_wave.sh
#
set -euo pipefail

REPO=/home/jdezoort/coupled-particle-edge-networks
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
SWEEP="$ROOT/jetclass_output/adam/sweep_lr"
LOGS="$ROOT/logs"
WATCH="$REPO/scans/jets/hxfer_upool_flatcos_gputest_watchdog.sh"
WATCH_LOG="$LOGS/jc_hxfc_watchdog.log"
mkdir -p "$LOGS" "$SWEEP"

echo "=== hxfer upool flatcos gputest wave $(date -Is) ==="

# Wipe incomplete D=128 η₀=0.5 (keep 0.01/0.05/0.1).
TAG128=hxfer-d128-fc-k6-R0p1
echo "--- inventory D=128 before wipe ---"
python3 - <<PY
import glob
from pathlib import Path
import pyarrow.parquet as pq
root = Path("$SWEEP")
tag = "$TAG128"
for et in ["0p01", "0p05", "0p1", "0p25", "0p5", "1"]:
    dirs = sorted(root.glob(f"*_{128}_{et}_*{tag}*"))
    dirs = [p for p in dirs if p.is_dir() or p.suffix == ".parquet"]
    step = 0
    for p in dirs:
        pqpath = p if p.suffix == ".parquet" else Path(str(p) + ".parquet")
        if pqpath.is_file():
            try:
                df = pq.read_table(pqpath, columns=["global_step"]).to_pandas()
                if len(df):
                    step = max(step, int(df["global_step"].max()))
            except Exception:
                pass
    print(f"  eta={et}: step={step} paths={len(dirs)}")
PY

echo "--- wipe incomplete D=128 eta=0.5 ---"
shopt -s nullglob
WIPED=0
for p in "$SWEEP"/*_128_0p5_*${TAG128}*; do
  echo "  rm $p"
  rm -rf "$p"
  WIPED=$((WIPED + 1))
done
echo "wiped $WIPED path(s)"

# Stop any prior hxfc watchdog.
if [[ -f "$LOGS/jc_hxfc_watchdog.pid" ]]; then
  old=$(cat "$LOGS/jc_hxfc_watchdog.pid" || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "stopping old watchdog pid=$old"
    kill "$old" 2>/dev/null || true
    sleep 1
  fi
fi

# Also stop any other hxfc watchdog by pattern (best-effort).
pkill -f "hxfer_upool_flatcos_gputest_watchdog.sh" 2>/dev/null || true
sleep 1

chmod +x "$WATCH"
nohup bash "$WATCH" >>"$WATCH_LOG" 2>&1 &
echo $! >"$LOGS/jc_hxfc_watchdog.pid"
echo "watchdog_pid=$(cat "$LOGS/jc_hxfc_watchdog.pid") log=$WATCH_LOG"

sleep 3
echo "--- watchdog head ---"
tail -n 20 "$WATCH_LOG" || true
echo "--- squeue ---"
squeue -u jdezoort -o '%.18i %.9P %.28j %.8T %.10M' 2>/dev/null | head -40 || true
