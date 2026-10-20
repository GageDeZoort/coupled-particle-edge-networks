#!/bin/bash
# Pool watchdog: L6/D512 Adam pretrain continue 40k → 390625 (flat η₀=0.1).
#
#   bash scans/jets/start_pretrain_L6_epoch_gputest_watchdogs.sh
#   touch ~/.cache/jc_pretrain_L6_watchdogs/STOP
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_L6_pretrain_epoch_gputest.slurm"
JOB_NAME="${JOB_NAME:-jc-pretrain-L6}"
TARGET="${TARGET:-390625}"
MAX_GPTEST="${MAX_GPTEST:-3}"
SLEEP_SEC="${SLEEP_SEC:-90}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/jc_pretrain_L6_watchdogs"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog_pretrain_L6.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_pretrain_L6")
LOCK="$CACHE/watchdog_pretrain_L6.lock"
RUN_TAG="pretrain-L6D512-b64-g4-s390625-dh32-k7-R0p1"
SWEEP="$ROOT/jetclass_output/adam/sweep_lr"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another pretrain-L6 watchdog holds $LOCK; exiting"
  exit 0
fi

current_step() {
  "$PYTHON" - "$SWEEP" "$RUN_TAG" "$LOG_DIR" <<'PY'
import re, sys
from pathlib import Path
sweep, run_tag, log_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
best = 0
cands = [p for p in sweep.glob(f"*6_512_0p1*{run_tag}*")]
live = [p for p in cands if "s390p625k" in p.name]
if live:
    cands = live
for p in cands:
    if p.is_dir() and (p / "SUPERSEDED.txt").exists():
        continue
    pq = p if p.suffix == ".parquet" else p.parent / f"{p.name}.parquet"
    if pq.is_file():
        try:
            import pyarrow.parquet as pqmod
            t = pqmod.read_table(pq, columns=["global_step"])
            col = t.column("global_step")
            if len(col):
                best = max(best, max(int(x.as_py()) for x in col if x.as_py() is not None))
        except Exception:
            pass
    run_dir = p if p.is_dir() else (sweep / p.name.replace(".parquet", ""))
    if run_dir.is_dir() and not (run_dir / "SUPERSEDED.txt").exists():
        try:
            import torch
            for ckpt in run_dir.glob("last*.ckpt"):
                try:
                    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                    best = max(best, int(payload.get("global_step", 0) or 0))
                except Exception:
                    pass
        except Exception:
            pass
logs = sorted(log_dir.glob("jc_pretrain_L6_gt_*.out"), key=lambda q: q.stat().st_mtime)
if logs:
    text = logs[-1].read_text(errors="ignore")[-200000:]
    steps = [int(x) for x in re.findall(r"global_step=(\d+)", text)]
    steps += [int(x) for x in re.findall(r"\[train\] step=(\d+)", text)]
    if steps:
        best = max(best, max(steps))
print(best)
PY
}

n_gputest() {
  timeout 30 squeue -u jdezoort -h -o '%P' 2>/dev/null | grep -c '^gputest$' || true
}

n_in_queue() {
  timeout 30 squeue -u jdezoort -n "$JOB_NAME" -h 2>/dev/null | wc -l
}

echo "$(date -Iseconds) pretrain-L6 watchdog start TARGET=$TARGET MAX_GPTEST=$MAX_GPTEST RUN_TAG=$RUN_TAG" | tee -a "$LOG"

while true; do
  for stopf in "${STOP_FILES[@]}"; do
    if [[ -f "$stopf" ]]; then
      echo "$(date -Iseconds) STOP $stopf; exiting" | tee -a "$LOG"
      exit 0
    fi
  done

  step=$(current_step)
  step=${step:-0}
  gt=$(n_gputest)
  q=$(n_in_queue)
  echo "$(date -Iseconds) step=$step/$TARGET gputest=$gt pretrain_q=$q" | tee -a "$LOG"

  if (( step >= TARGET )); then
    echo "$(date -Iseconds) DONE step=$step — exiting watchdog" | tee -a "$LOG"
    exit 0
  fi

  if (( q < 1 && gt < MAX_GPTEST )); then
    jid=$(sbatch --parsable "$SLURM" 2>/dev/null || true)
    echo "$(date -Iseconds) submitted job=$jid step=$step" | tee -a "$LOG"
  else
    echo "$(date -Iseconds) hold (q=$q gt=$gt max=$MAX_GPTEST)" | tee -a "$LOG"
  fi

  sleep "$SLEEP_SEC"
done
