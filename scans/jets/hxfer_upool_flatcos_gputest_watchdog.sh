#!/bin/bash
# Keep D=128 remaining + D=256 full flat→cosine hxfer cells moving on gputest.
#
# MaxJobsPU≈3: at most 3 concurrent gputest array/job IDs. Resubmits until
# each cell reaches TARGET steps (default 20000). Lightning resumes last.ckpt.
#
#   nohup bash scans/jets/hxfer_upool_flatcos_gputest_watchdog.sh \
#     > /scratch/.../logs/jc_hxfc_watchdog.log 2>&1 &
#
set -uo pipefail

REPO=/home/jdezoort/coupled-particle-edge-networks
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
LOGS="$ROOT/logs"
SLURM="$REPO/scans/jets/scaling_law_hxfer_upool_flatcos_gputest.slurm"
WATCH_LOG="${WATCH_LOG:-$LOGS/jc_hxfc_watchdog.log}"
# D=128: 40k. D=256: 20k. Capacity probe L=6 D=512 η₀=0.1: 40k.
TARGET_D128="${TARGET_D128:-40000}"
TARGET_D256="${TARGET_D256:-20000}"
TARGET_L6D512="${TARGET_L6D512:-40000}"
MAX_GPTEST="${MAX_GPTEST:-3}"
SLEEP_SEC="${SLEEP_SEC:-120}"
MAX_ITERS="${MAX_ITERS:-2000}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
mkdir -p "$LOGS"

# Cells are L:D:eta. Probe first so a freed gputest slot goes to L=6 D=512.
CELLS=(
  "6:512:0.1"
  "4:128:0.05"
  "4:128:0.1"
  "4:256:0.01"
  "4:256:0.05"
  "4:256:0.1"
  "4:256:0.25"
  "4:256:0.5"
  "4:256:1.0"
  "4:128:0.25"
  "4:128:0.5"
  "4:128:1.0"
  "4:128:0.01"
)

target_for() {
  local L="$1" w="$2"
  if [[ "$L" == "6" && "$w" == "512" ]]; then
    echo "$TARGET_L6D512"
  elif [[ "$w" == "128" ]]; then
    echo "$TARGET_D128"
  else
    echo "$TARGET_D256"
  fi
}

eta_tag() {
  python3 -c "e=float('$1'); print(f'{e:g}'.replace('.','p'))"
}

run_tag_for() {
  local L="$1" w="$2"
  if [[ "$L" == "4" ]]; then
    echo "hxfer-d${w}-fc-k6-R0p1"
  else
    echo "hxfer-L${L}-d${w}-fc-k6-R0p1"
  fi
}

job_name_for() {
  local L="$1" w="$2" et
  et=$(eta_tag "$3")
  if [[ "$L" == "4" ]]; then
    echo "jc-hxfc-D${w}-e${et}"
  else
    echo "jc-hxfc-L${L}D${w}-e${et}"
  fi
}

cell_step() {
  local L="$1" w="$2" eta="$3"
  local et tag
  et=$(eta_tag "$eta")
  tag=$(run_tag_for "$L" "$w")
  "$PYTHON" - "$ROOT" "$L" "$w" "$et" "$tag" <<'PY'
import glob, sys
from pathlib import Path
root, L, w, et, tag = sys.argv[1:6]
pat = f"{root}/jetclass_output/adam/sweep_lr/*_{L}_{w}_{et}_*{tag}*"
best = 0
for c in sorted(glob.glob(pat)):
    p = Path(c)
    pqpath = p if p.suffix == ".parquet" else p.parent / f"{p.name}.parquet"
    if not pqpath.exists() or pqpath.suffix != ".parquet":
        continue
    try:
        import pyarrow.parquet as pq
        df = pq.read_table(pqpath, columns=["global_step"]).to_pandas()
        if len(df):
            best = max(best, int(df["global_step"].max()))
    except Exception:
        pass
print(best)
PY
}

gputest_arrays() {
  # Unique parent job IDs on gputest (MaxJobsPU counts these).
  squeue -u jdezoort -p gputest -h -o "%i" 2>/dev/null \
    | sed 's/_.*//' | sort -u | wc -l
}

cell_queued() {
  local name="$1"
  squeue -u jdezoort -n "$name" -h 2>/dev/null | grep -q .
}

submit_cell() {
  local L="$1" w="$2" eta="$3" tgt="$4"
  local name et out
  name=$(job_name_for "$L" "$w" "$eta")
  et=$(eta_tag "$eta")
  out=$(sbatch --parsable \
    --job-name="$name" \
    --output="$LOGS/jc_hxfc_L${L}D${w}_e${et}_%j.out" \
    --error="$LOGS/jc_hxfc_L${L}D${w}_e${et}_%j.err" \
    --export=ALL,DEPTH="$L",WIDTH="$w",ETA0="$eta",SEED=0,MAX_STEPS="$tgt" \
    "$SLURM" 2>&1) || true
  echo "$out"
}

log() {
  # Caller should redirect stdout to WATCH_LOG (no tee — avoids doubles).
  echo "$(date -Is) $*"
}

log "START hxfer upool flatcos gputest watchdog D128→${TARGET_D128} D256→${TARGET_D256} L6D512→${TARGET_L6D512} max_gputest=$MAX_GPTEST"
log "cells=${CELLS[*]}"

for iter in $(seq 1 "$MAX_ITERS"); do
  n=$(gputest_arrays)
  pending=0
  done_n=0
  status_line=""
  for cell in "${CELLS[@]}"; do
    L=${cell%%:*}
    rest=${cell#*:}
    w=${rest%%:*}
    eta=${rest##*:}
    name=$(job_name_for "$L" "$w" "$eta")
    tgt=$(target_for "$L" "$w")
    step=$(cell_step "$L" "$w" "$eta")
    if (( step >= tgt )); then
      done_n=$((done_n + 1))
      status_line+=" L${L}D${w}@${eta}=done(${step}/${tgt})"
      continue
    fi
    pending=$((pending + 1))
    if cell_queued "$name"; then
      status_line+=" L${L}D${w}@${eta}=Q/R(${step}/${tgt})"
      continue
    fi
    if (( n < MAX_GPTEST )); then
      jid=$(submit_cell "$L" "$w" "$eta" "$tgt")
      if [[ "$jid" =~ ^[0-9]+$ ]]; then
        log "SUBMIT L=$L D=$w eta=$eta step=$step tgt=$tgt → job=$jid ($name)"
        n=$((n + 1))
        status_line+=" L${L}D${w}@${eta}=sub(${step}→$jid)"
      else
        log "FAIL submit L=$L D=$w eta=$eta: $jid"
        status_line+=" L${L}D${w}@${eta}=fail(${step})"
      fi
    else
      status_line+=" L${L}D${w}@${eta}=wait(${step}/${tgt})"
    fi
  done

  log "iter=$iter gputest_arrays=$n done=$done_n/$(( ${#CELLS[@]} )) pending=$pending$status_line"

  if (( pending == 0 )); then
    log "ALL cells reached per-width targets — exiting"
    exit 0
  fi
  sleep "$SLEEP_SEC"
done

log "FAIL watchdog hit MAX_ITERS=$MAX_ITERS"
exit 1
