#!/bin/bash
# Pool watchdog: L4/D256 Adam pretrain epoch-3 constant min LR (781250 → 1171875).
#
#   bash scans/jets/start_pretrain_L4_minlr_epoch_gputest_watchdogs.sh
#   touch ~/.cache/jc_pretrain_L4min_watchdogs/STOP
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_L4_pretrain_minlr_epoch_gputest.slurm"
JOB_NAME="${JOB_NAME:-jc-pretrain-L4min}"
TARGET_COS="${TARGET_COS:-781250}"
TARGET="${TARGET:-1171875}"
MAX_GPTEST="${MAX_GPTEST:-3}"
SLEEP_SEC="${SLEEP_SEC:-90}"
MAX_PENDING_MIN="${MAX_PENDING_MIN:-45}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/jc_pretrain_L4min_watchdogs"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog_pretrain_L4min.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_pretrain_L4min")
LOCK="$CACHE/watchdog_pretrain_L4min.lock"
COS_TAG="pretrain-cos-b64-g4-s781250"
MIN_TAG="pretrain-minlr-b64-g4-s1171875"
SWEEP="$ROOT/jetclass_output/adam/sweep_lr"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another pretrain-L4min watchdog holds $LOCK; exiting"
  exit 0
fi

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

step_for_tag() {
  local tag="$1"
  "$PYTHON" - "$SWEEP" "$tag" "$LOG_DIR" <<'PY'
import re, sys
from pathlib import Path
sweep, tag, log_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
best = 0
try:
    import torch
    for p in sweep.glob(f"*4_256*{tag}*"):
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
pat = "jc_pretrain_L4min_gt_*.out" if "minlr" in tag else "jc_pretrain_L4cos_gt_*.out"
logs = sorted(log_dir.glob(pat), key=lambda q: q.stat().st_mtime)
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

echo "$(date -Iseconds) pretrain-L4min watchdog start TARGET=$TARGET cos=$TARGET_COS" | tee -a "$LOG"

while true; do
  for stopf in "${STOP_FILES[@]}"; do
    if [[ -f "$stopf" ]]; then
      echo "$(date -Iseconds) STOP $stopf; exiting" | tee -a "$LOG"
      exit 0
    fi
  done

  reap_bad_queue

  cos=$(step_for_tag "$COS_TAG")
  cos=${cos:-0}
  step=$(step_for_tag "$MIN_TAG")
  step=${step:-0}
  if (( step < cos && cos >= TARGET_COS )); then
    step=$cos
  fi
  gt=$(n_gputest)
  q=$(n_in_queue)
  echo "$(date -Iseconds) cos=$cos/$TARGET_COS minlr=$step/$TARGET gputest=$gt q=$q" | tee -a "$LOG"

  if (( step >= TARGET )); then
    echo "$(date -Iseconds) DONE minlr step=$step — exiting" | tee -a "$LOG"
    exit 0
  fi

  if (( cos < TARGET_COS - 1000 )); then
    echo "$(date -Iseconds) waiting for cosine epoch (cos=$cos)" | tee -a "$LOG"
    sleep "$SLEEP_SEC"
    continue
  fi

  if (( q < 1 && gt < MAX_GPTEST )); then
    jid=$(submit_gputest)
    echo "$(date -Iseconds) submitted job=$jid minlr_step=$step (a100/59m→gputest)" | tee -a "$LOG"
  fi

  sleep "$SLEEP_SEC"
done
