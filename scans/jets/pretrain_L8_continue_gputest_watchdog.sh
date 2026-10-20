#!/bin/bash
# Pool watchdog: L8/D1024 Adam pretrain continue on Della gputest → 160k
# (second unique epoch @ B_eff=128, const η₀=0.1). Shares gputest with FT.
#
#   bash scans/jets/start_pretrain_L8_continue_gputest_watchdogs.sh
#   touch ~/.cache/jc_pretrain_L8_watchdogs/STOP
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_L8_pretrain_continue_gputest.slurm"
JOB_NAME="${JOB_NAME:-jc-pretrain-L8}"
TARGET="${TARGET:-160000}"
MAX_GPTEST="${MAX_GPTEST:-2}"
SLEEP_SEC="${SLEEP_SEC:-90}"
# Share gputest with FT — don't thrash Priority with short cancels.
MAX_PENDING_MIN="${MAX_PENDING_MIN:-45}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/jc_pretrain_L8_watchdogs"
LOG_DIR="$ROOT/logs"
SEED_CKPT="$ROOT/lambda_L8D1024/checkpoints/last.ckpt"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog_pretrain_L8.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_pretrain_L8")
LOCK="$CACHE/watchdog_pretrain_L8.lock"
RUN_TAG="pretrain-L8D1024-b32-g4-s160000-dh32-k7-R0p1-eta0p1-s160k"
PREV_TAG="pretrain-L8D1024-b32-g4-s80000-dh32-k7-R0p1-eta0p1-s80k"
SWEEP="$ROOT/jetclass_output/adam/sweep_lr"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another pretrain-L8 watchdog holds $LOCK; exiting"
  exit 0
fi

# Della: partition=a100 + time≤59m routes onto gputest (do not set -p gputest).
submit_gputest() {
  sbatch --parsable -p a100 -t 0:59:00 "$SLURM" 2>/dev/null || true
}

reap_bad_queue() {
  local line jid part state submit age_min
  while read -r line; do
    [[ -z "$line" ]] && continue
    jid=$(cut -d'|' -f1 <<<"$line")
    part=$(cut -d'|' -f2 <<<"$line")
    state=$(cut -d'|' -f3 <<<"$line")
    submit=$(cut -d'|' -f4 <<<"$line")
    [[ -z "$jid" ]] && continue
    if [[ "$part" != "gputest" ]]; then
      echo "$(date -Iseconds) scancel wrong partition jid=$jid part=$part (want gputest)" | tee -a "$LOG"
      timeout 30 scancel "$jid" 2>/dev/null || true
      continue
    fi
    if [[ "$state" == "PENDING" ]]; then
      age_min=$("$PYTHON" -c "
from datetime import datetime
s='''$submit'''.strip()
try:
  t=datetime.fromisoformat(s)
  if t.tzinfo is None:
    t=t.astimezone()
  print(max(0, int((datetime.now().astimezone()-t).total_seconds()//60)))
except Exception:
  print(-1)
" 2>/dev/null || echo -1)
      if (( age_min >= MAX_PENDING_MIN )); then
        echo "$(date -Iseconds) scancel stale PENDING jid=$jid age=${age_min}m → requeue gputest" | tee -a "$LOG"
        timeout 30 scancel "$jid" 2>/dev/null || true
      fi
    fi
  done < <(timeout 30 squeue -u jdezoort -n "$JOB_NAME" -h -o '%i|%P|%T|%V' 2>/dev/null)
}

current_step() {
  "$PYTHON" - "$SWEEP" "$RUN_TAG" "$PREV_TAG" "$LOG_DIR" "$SEED_CKPT" <<'PY'
import re, sys
from pathlib import Path
sweep, tag, prev, log_dir, seed = (
    Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
)
best = 0
try:
    import torch
    if seed.is_file():
        payload = torch.load(seed, map_location="cpu", weights_only=False)
        best = max(best, int(payload.get("global_step", 0) or 0))
    for t in (tag, prev):
        for p in sweep.glob(f"*8_1024_0p1*{t}*"):
            if not p.is_dir() or (p / "SUPERSEDED.txt").exists():
                continue
            for ckpt in p.glob("last*.ckpt"):
                try:
                    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                    best = max(best, int(payload.get("global_step", 0) or 0))
                except Exception:
                    pass
except Exception:
    pass
logs = sorted(log_dir.glob("jc_pretrain_L8_gt_*.out"), key=lambda q: q.stat().st_mtime)
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

echo "$(date -Iseconds) pretrain-L8 watchdog start TARGET=$TARGET RUN_TAG=$RUN_TAG" | tee -a "$LOG"

while true; do
  for stopf in "${STOP_FILES[@]}"; do
    if [[ -f "$stopf" ]]; then
      echo "$(date -Iseconds) STOP $stopf; exiting" | tee -a "$LOG"
      exit 0
    fi
  done

  reap_bad_queue

  if [[ ! -f "$SEED_CKPT" ]]; then
    echo "$(date -Iseconds) waiting for mirrored Lambda ckpt at $SEED_CKPT" | tee -a "$LOG"
    sleep "$SLEEP_SEC"
    continue
  fi

  step=$(current_step)
  step=${step:-0}
  gt=$(n_gputest)
  q=$(n_in_queue)
  echo "$(date -Iseconds) step=$step/$TARGET gputest=$gt q=$q" | tee -a "$LOG"

  if (( step >= TARGET )); then
    echo "$(date -Iseconds) DONE step=$step — exiting" | tee -a "$LOG"
    exit 0
  fi

  if (( q < 1 && gt < MAX_GPTEST )); then
    jid=$(submit_gputest)
    echo "$(date -Iseconds) submitted job=$jid step=$step (a100/59m→gputest)" | tee -a "$LOG"
  fi

  sleep "$SLEEP_SEC"
done
