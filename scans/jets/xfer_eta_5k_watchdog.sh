#!/bin/bash
# Resubmit transfer-η cells until each has a val @ step=5000.
# Covers L=3/D=128, L=5/D=512, L=4/{256,512,1024} × η∈{0.01…1}.
#
# Done = log (any jc_xfer_eta_*) contains [val] step=5000 for that (L,D,η),
#     or last.ckpt global_step >= 4900 in the matching xfer-5k run dir.
# On OOM for D=1024, resubmit with BATCH=32.
#
#   nohup bash scans/jets/xfer_eta_5k_watchdog.sh >/dev/null 2>&1 &
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
SLURM=/home/jdezoort/coupled-particle-edge-networks/scans/jets/scaling_law_transfer_eta_knn6_5k_full.slurm
LOG="$ROOT/logs/watchdog_xfer_eta_5k.log"
STATE=/tmp/watchdog_xfer_eta_5k_state.txt
JOB_NAME=jc-xfer-eta-full
TARGET=5000
SLEEP_SEC=180
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
mkdir -p "$ROOT/logs"

# task -> L D eta batch_default
SPEC=(
  "0:3:128:0.01:128"
  "1:3:128:0.05:128"
  "2:3:128:0.10:128"
  "3:3:128:0.25:128"
  "4:3:128:0.50:128"
  "5:3:128:1.00:128"
  "6:5:512:0.01:128"
  "7:5:512:0.05:128"
  "8:5:512:0.10:128"
  "9:5:512:0.25:128"
  "10:5:512:0.50:128"
  "11:5:512:1.00:128"
  "12:4:256:0.01:128"
  "13:4:256:0.05:128"
  "14:4:256:0.10:128"
  "15:4:256:0.25:128"
  "16:4:256:0.50:128"
  "17:4:256:1.00:128"
  "18:4:512:0.01:128"
  "19:4:512:0.05:128"
  "20:4:512:0.10:128"
  "21:4:512:0.25:128"
  "22:4:512:0.50:128"
  "23:4:512:1.00:128"
  "24:4:1024:0.01:64"
  "25:4:1024:0.05:64"
  "26:4:1024:0.10:64"
  "27:4:1024:0.25:64"
  "28:4:1024:0.50:64"
  "29:4:1024:1.00:64"
)

eta_tag() {
  # 0.01 -> 0p01 ; 0.10 -> 0p1 ; 1.00 -> 1
  "$PYTHON" - "$1" <<'PY'
import sys
e=float(sys.argv[1])
if abs(e-int(e))<1e-12:
    print(str(int(e)))
else:
    s=f"{e:.10g}".replace(".","p")
    print(s)
PY
}

cell_done() {
  local L="$1" D="$2" eta="$3" batch="$4"
  local tag
  tag=$(eta_tag "$eta")
  if "$PYTHON" - "$L" "$D" "$eta" "$ROOT/logs" <<'PY'
import re, sys
from pathlib import Path
L, D, eta = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
logdir = Path(sys.argv[4])
# Exact eta0= tokens; (?!\d) so eta0=0.1 does not match eta0=0.10's prefix
# and eta0=0 never matches eta0=0.50.
cands = {f"{eta:.2f}", f"{eta:.1f}", f"{eta:g}"}
if abs(eta - int(eta)) < 1e-12:
    cands.add(str(int(eta)))
pat_eta = re.compile(
    r"eta0=("
    + "|".join(re.escape(c) for c in sorted(cands, key=len, reverse=True))
    + r")(?!\d)"
)
val5 = re.compile(r"\[val\].*?step=5000\s+val_loss=")
pat_L = re.compile(rf"\bL={L}\b")
pat_D = re.compile(rf"\bD={D}\b")
for f in sorted(logdir.glob("jc_xfer_eta_*.out"), key=lambda p: p.stat().st_mtime, reverse=True):
    try:
        text = f.read_text(errors="replace")
    except Exception:
        continue
    if not val5.search(text):
        continue
    if not (pat_L.search(text) and pat_D.search(text) and pat_eta.search(text)):
        continue
    if not re.search(r"transfer-eta|xfer-5k|label=xfer", text):
        continue
    print(f.name)
    raise SystemExit(0)
raise SystemExit(1)
PY
  then
    return 0
  fi

  local hit
  hit=$(find "$ROOT/jetclass_output/adam/sweep_lr" -maxdepth 1 -type d \
    -name "capen-llama-att_${L}_${D}_${tag}_*xfer-5k-b${batch}" 2>/dev/null | head -n 1 || true)
  if [[ -z "${hit:-}" ]]; then
    return 1
  fi
  local step
  step=$("$PYTHON" - "$hit/last.ckpt" <<'PY'
import sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    print(-1); raise SystemExit
try:
    import torch
    ck=torch.load(p, map_location="cpu", weights_only=False)
    print(int(ck.get("global_step", -1)))
except Exception:
    print(-1)
PY
)
  (( step >= TARGET - 100 ))
}

task_in_queue() {
  local task="$1"
  squeue -u "$USER" -n "$JOB_NAME" -r -h -o '%i' 2>/dev/null \
    | awk -F_ -v t="$task" '$NF==t {found=1} END{exit !found}'
}

# Also treat legacy job names still filling L3/L5 as "in queue" for those cells
legacy_running_for() {
  local L="$1" D="$2" eta="$3"
  # Parse running jc-xfer-eta-knn6 logs is hard; use squeue + recent out headers
  squeue -u "$USER" -n jc-xfer-eta-knn6 -r -h -o '%i' 2>/dev/null | while read -r jid; do
    [[ -z "$jid" ]] && continue
    local f
    f=$(ls -t "$ROOT/logs"/jc_xfer_eta_knn6_"${jid}".out 2>/dev/null | head -n 1 || true)
    [[ -z "${f:-}" ]] && f="$ROOT/logs/jc_xfer_eta_knn6_${jid}.out"
    [[ -f "$f" ]] || continue
    if grep -q "L=$L" "$f" && grep -q "D=$D" "$f" && grep -qE "eta0=${eta}([^0-9]|$)" "$f"; then
      echo "$jid"
      return 0
    fi
  done
  return 1
}

log_has_oom() {
  local task="$1"
  local f
  f=$(ls -t "$ROOT/logs"/jc_xfer_eta_full_*_"${task}.err" 2>/dev/null | head -n 1 || true)
  [[ -n "${f:-}" ]] || return 1
  grep -qE 'CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError' "$f" 2>/dev/null
}

submit_task() {
  local task="$1" batch="$2"
  local out
  out=$(sbatch --export=ALL,BATCH="$batch" --array="$task" "$SLURM" 2>&1) || true
  echo "$(date -Is) submit task=$task B=$batch out=$out"
}

{
  echo "$(date -Is) watchdog start pid=$$ target=$TARGET"
  while true; do
    done_n=0
    need_n=0
    active_n=0
    for spec in "${SPEC[@]}"; do
      IFS=: read -r task L D eta batch <<<"$spec"
      if cell_done "$L" "$D" "$eta" "$batch"; then
        done_n=$((done_n + 1))
        echo "$(date -Is) DONE L=$L D=$D eta=$eta"
        continue
      fi
      if task_in_queue "$task"; then
        active_n=$((active_n + 1))
        if log_has_oom "$task" && (( D >= 1024 )); then
          echo "$(date -Is) OOM L=$L D=$D eta=$eta — cancel + BATCH=32"
          mapfile -t hits < <(squeue -u "$USER" -n "$JOB_NAME" -r -h -o '%i' | awk -F_ -v t="$task" '$NF==t{print}')
          for j in "${hits[@]:-}"; do scancel "$j" 2>/dev/null || true; done
          sleep 2
          submit_task "$task" 32
        else
          echo "$(date -Is) running/queued task=$task L=$L D=$D eta=$eta"
        fi
        continue
      fi
      # legacy L3/L5 still finishing under old job name
      if legacy_running_for "$L" "$D" "$eta" >/dev/null; then
        active_n=$((active_n + 1))
        echo "$(date -Is) legacy running L=$L D=$D eta=$eta"
        continue
      fi
      need_n=$((need_n + 1))
      b="$batch"
      if log_has_oom "$task" && (( D >= 1024 )); then
        b=32
      fi
      submit_task "$task" "$b"
      active_n=$((active_n + 1))
    done
    echo "done=$done_n/30 need_submit_pass=$need_n active=$active_n ts=$(date -Is)" | tee "$STATE"
    echo "AGENT_LOOP_TICK_xfer_watchdog done=$done_n/30"
    if (( done_n == 30 )); then
      echo "$(date -Is) ALL 30 cells at ${TARGET}; exit"
      echo "ALL_DONE"
      break
    fi
    sleep "$SLEEP_SEC"
  done
} >>"$LOG" 2>&1
