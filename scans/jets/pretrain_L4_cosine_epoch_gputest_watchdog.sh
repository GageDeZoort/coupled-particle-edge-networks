#!/bin/bash
# Pool watchdog: L4/D256 Adam pretrain epoch-2 cosine (390625 → 781250).
#
# Waits until const-LR epoch (TARGET_E1) is done, then keeps a gputest slot
# filled for the cosine continue.
#
#   bash scans/jets/start_pretrain_L4_cosine_epoch_gputest_watchdogs.sh
#   touch ~/.cache/jc_pretrain_L4cos_watchdogs/STOP
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_L4_pretrain_cosine_epoch_gputest.slurm"
JOB_NAME="${JOB_NAME:-jc-pretrain-L4cos}"
TARGET_E1="${TARGET_E1:-390625}"
TARGET="${TARGET:-781250}"
MAX_GPTEST="${MAX_GPTEST:-3}"
SLEEP_SEC="${SLEEP_SEC:-90}"
# Requeue if stuck PENDING this many minutes (Priority / wrong partition).
# Keep high while sharing gputest with L8 — short cancels just burn Priority.
MAX_PENDING_MIN="${MAX_PENDING_MIN:-45}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/jc_pretrain_L4cos_watchdogs"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog_pretrain_L4cos.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_pretrain_L4cos")
LOCK="$CACHE/watchdog_pretrain_L4cos.lock"
CONST_TAG="pretrain-b64-g4-s390625-dh32-k7-R0p1"
COS_TAG="pretrain-cos-b64-g4-s781250-dh32-k7-R0p1-min0p01"
SWEEP="$ROOT/jetclass_output/adam/sweep_lr"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another pretrain-L4cos watchdog holds $LOCK; exiting"
  exit 0
fi

# Della: partition=a100 + time≤59m routes onto gputest (do not set -p gputest).
submit_gputest() {
  sbatch --parsable -p a100 -t 0:59:00 "$SLURM" 2>/dev/null || true
}

# Cancel jobs on the wrong partition (normal a100/gpu long queue) or stale PENDING.
reap_bad_queue() {
  local line jid part state submit age_min
  while read -r line; do
    [[ -z "$line" ]] && continue
    # pipe-separated so Reason never shifts fields
    jid=$(cut -d'|' -f1 <<<"$line")
    part=$(cut -d'|' -f2 <<<"$line")
    state=$(cut -d'|' -f3 <<<"$line")
    submit=$(cut -d'|' -f4 <<<"$line")
    [[ -z "$jid" ]] && continue
    # Wrong partition: landed on normal a100/gpu instead of gputest.
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
  local prefer="$2"
  "$PYTHON" - "$SWEEP" "$tag" "$prefer" "$LOG_DIR" <<'PY'
import re, sys
from pathlib import Path
sweep, tag, prefer, log_dir = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
best = 0
cands = [p for p in sweep.glob(f"*4_256_0p1*{tag}*")]
if prefer:
    live = [p for p in cands if prefer in p.name]
    if live:
        cands = live
for p in cands:
    if p.is_dir() and (p / "SUPERSEDED.txt").exists():
        continue
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
# Freshest matching slurm log
pat = "jc_pretrain_L4cos_gt_*.out" if "cos" in tag else "jc_pretrain_L4_gt_*.out"
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

echo "$(date -Iseconds) pretrain-L4cos watchdog start TARGET=$TARGET E1=$TARGET_E1" | tee -a "$LOG"

while true; do
  for stopf in "${STOP_FILES[@]}"; do
    if [[ -f "$stopf" ]]; then
      echo "$(date -Iseconds) STOP $stopf; exiting" | tee -a "$LOG"
      exit 0
    fi
  done

  reap_bad_queue

  e1=$(step_for_tag "$CONST_TAG" "s390p625k")
  e1=${e1:-0}
  step=$(step_for_tag "$COS_TAG" "s781p25k")
  # If cosine not seeded yet, step may be 0 while e1 is high — that's fine.
  step=${step:-0}
  # Also count const step if cosine dir still absent but we're past E1
  if (( step < e1 && e1 >= TARGET_E1 )); then
    step=$e1
  fi
  gt=$(n_gputest)
  q=$(n_in_queue)
  echo "$(date -Iseconds) e1=$e1/$TARGET_E1 cos=$step/$TARGET gputest=$gt q=$q" | tee -a "$LOG"

  if (( step >= TARGET )); then
    echo "$(date -Iseconds) DONE cosine step=$step — exiting" | tee -a "$LOG"
    exit 0
  fi

  # Allow cosine to start once const is within one ckpt interval of target.
  if (( e1 < TARGET_E1 - 1000 )); then
    echo "$(date -Iseconds) waiting for const epoch (e1=$e1)" | tee -a "$LOG"
    sleep "$SLEEP_SEC"
    continue
  fi

  if (( q < 1 && gt < MAX_GPTEST )); then
    jid=$(submit_gputest)
    echo "$(date -Iseconds) submitted job=$jid cos_step=$step (a100/59m→gputest)" | tee -a "$LOG"
  fi

  sleep "$SLEEP_SEC"
done
