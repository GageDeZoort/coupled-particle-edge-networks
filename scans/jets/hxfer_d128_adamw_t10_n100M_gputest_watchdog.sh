#!/bin/bash
# Drive D=128 AdamW t=10 n_train=100M η₀∈{0.05,0.1,0.25} to 40k on gputest.
set -uo pipefail

REPO=/home/jdezoort/coupled-particle-edge-networks
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
LOGS="$ROOT/logs"
SLURM="$REPO/scans/jets/scaling_law_hxfer_d128_adamw_t10_n100M_gputest.slurm"
WATCH_LOG="${WATCH_LOG:-$LOGS/jc_aw100M_watchdog.log}"
TARGET="${TARGET:-40000}"
MAX_GPTEST="${MAX_GPTEST:-3}"
SLEEP_SEC="${SLEEP_SEC:-120}"
MAX_ITERS="${MAX_ITERS:-2000}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
TAG="hxfer-d128-aw-t10-n100M-s40000-k6-R0p1"
SWEEP="$ROOT/jetclass_output/adamw/sweep_lr"
mkdir -p "$LOGS" "$SWEEP"

CELLS=(0.05 0.1 0.25)

eta_tag() {
  python3 -c "e=float('$1'); print(f'{e:g}'.replace('.','p'))"
}

job_name_for() {
  echo "jc-aw100M-D128-e$(eta_tag "$1")"
}

cell_step() {
  local et
  et=$(eta_tag "$1")
  "$PYTHON" - "$SWEEP" "$et" "$TAG" <<'PY'
import glob, sys
from pathlib import Path
root, et, tag = sys.argv[1:4]
best = 0
for c in glob.glob(f"{root}/*_4_128_{et}_*{tag}*"):
    p = Path(c)
    pq = p if p.suffix == ".parquet" else Path(str(p) + ".parquet")
    if not pq.is_file():
        continue
    try:
        import pyarrow.parquet as pqmod
        df = pqmod.read_table(pq, columns=["global_step"]).to_pandas()
        if len(df):
            best = max(best, int(df["global_step"].max()))
    except Exception:
        pass
print(best)
PY
}

gputest_arrays() {
  timeout 30 squeue -u jdezoort -p gputest -h -o "%i" 2>/dev/null \
    | sed 's/_.*//' | sort -u | wc -l
}

cell_queued() {
  timeout 30 squeue -u jdezoort -n "$1" -h 2>/dev/null | grep -q .
}

submit_cell() {
  local eta="$1" et name
  et=$(eta_tag "$eta")
  name=$(job_name_for "$eta")
  sbatch --parsable \
    --job-name="$name" \
    --output="$LOGS/jc_aw100M_D128_e${et}_%j.out" \
    --error="$LOGS/jc_aw100M_D128_e${et}_%j.err" \
    --export=ALL,ETA0="$eta",SEED=0,MAX_STEPS="$TARGET" \
    "$SLURM" 2>&1 || true
}

log() { echo "$(date -Is) $*"; }

log "START D=128 AdamW t=10 n_train=100M target=$TARGET etas=${CELLS[*]}"

for iter in $(seq 1 "$MAX_ITERS"); do
  n=$(gputest_arrays)
  pending=0
  done_n=0
  status_line=""
  for eta in "${CELLS[@]}"; do
    name=$(job_name_for "$eta")
    step=$(cell_step "$eta")
    if (( step >= TARGET )); then
      done_n=$((done_n + 1))
      status_line+=" e${eta}=done(${step})"
      continue
    fi
    pending=$((pending + 1))
    if cell_queued "$name"; then
      status_line+=" e${eta}=Q/R(${step})"
      continue
    fi
    if (( n < MAX_GPTEST )); then
      jid=$(submit_cell "$eta")
      if [[ "$jid" =~ ^[0-9]+$ ]]; then
        log "SUBMIT eta=$eta step=$step → $jid ($name)"
        n=$((n + 1))
        status_line+=" e${eta}=sub(${step}→$jid)"
      else
        log "FAIL submit eta=$eta: $jid"
        status_line+=" e${eta}=fail"
      fi
    else
      status_line+=" e${eta}=wait(${step})"
    fi
  done
  log "iter=$iter gputest=$n done=$done_n/3 pending=$pending$status_line"
  if (( pending == 0 )); then
    log "ALL three cells reached $TARGET"
    exit 0
  fi
  sleep "$SLEEP_SEC"
done
log "FAIL MAX_ITERS"
exit 1
