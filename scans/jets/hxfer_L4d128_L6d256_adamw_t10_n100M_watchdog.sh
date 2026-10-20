#!/bin/bash
# Drive L=4 D=128 (gputest) + L=6 D=256 (ailab) AdamW t=10 n_train=100M
# attentional pool, const LR 40k, η₀∈{0.05,0.1,0.25}.
set -uo pipefail

REPO=/home/jdezoort/coupled-particle-edge-networks
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
LOGS="$ROOT/logs"
TARGET="${TARGET:-40000}"
MAX_GPTEST="${MAX_GPTEST:-3}"
MAX_AILAB="${MAX_AILAB:-3}"
SLEEP_SEC="${SLEEP_SEC:-120}"
MAX_ITERS="${MAX_ITERS:-2000}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
SWEEP="$ROOT/jetclass_output/adamw/sweep_lr"
mkdir -p "$LOGS" "$SWEEP"

# cell: depth:width:eta:partition:slurm:job_prefix:tag
CELLS=(
  "4:128:0.05:gputest:scaling_law_hxfer_L4_d128_adamw_t10_n100M_gputest.slurm:jc-awL4d128ap:hxfer-L4-d128-ap-aw-t10-n100M-s40000-k6-R0p1"
  "4:128:0.1:gputest:scaling_law_hxfer_L4_d128_adamw_t10_n100M_gputest.slurm:jc-awL4d128ap:hxfer-L4-d128-ap-aw-t10-n100M-s40000-k6-R0p1"
  "4:128:0.25:gputest:scaling_law_hxfer_L4_d128_adamw_t10_n100M_gputest.slurm:jc-awL4d128ap:hxfer-L4-d128-ap-aw-t10-n100M-s40000-k6-R0p1"
  "6:256:0.05:ailab:scaling_law_hxfer_L6_d256_adamw_t10_n100M_ailab.slurm:jc-awL6d256ap:hxfer-L6-d256-ap-aw-t10-n100M-s40000-k6-R0p1"
  "6:256:0.1:ailab:scaling_law_hxfer_L6_d256_adamw_t10_n100M_ailab.slurm:jc-awL6d256ap:hxfer-L6-d256-ap-aw-t10-n100M-s40000-k6-R0p1"
  "6:256:0.25:ailab:scaling_law_hxfer_L6_d256_adamw_t10_n100M_ailab.slurm:jc-awL6d256ap:hxfer-L6-d256-ap-aw-t10-n100M-s40000-k6-R0p1"
)

eta_tag() {
  python3 -c "e=float('$1'); print(f'{e:g}'.replace('.','p'))"
}

job_name_for() {
  local prefix="$1" eta="$2"
  echo "${prefix}-e$(eta_tag "$eta")"
}

cell_step() {
  local depth="$1" width="$2" eta="$3" tag="$4"
  local et
  et=$(eta_tag "$eta")
  "$PYTHON" - "$SWEEP" "$depth" "$width" "$et" "$tag" <<'PY'
import glob, sys
from pathlib import Path
root, depth, width, et, tag = sys.argv[1:6]
best = 0
for c in glob.glob(f"{root}/*_{depth}_{width}_{et}_*{tag}*"):
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

partition_count() {
  local part="$1" prefix="$2"
  timeout 30 squeue -u jdezoort -p "$part" -h -o "%i %j" 2>/dev/null \
    | awk -v p="$prefix" '$2 ~ p {print $1}' | sed 's/_.*//' | sort -u | wc -l
}

cell_queued() {
  timeout 30 squeue -u jdezoort -n "$1" -h 2>/dev/null | grep -q .
}

submit_cell() {
  local eta="$1" slurm="$2" prefix="$3" part="$4"
  local et name
  et=$(eta_tag "$eta")
  name=$(job_name_for "$prefix" "$eta")
  sbatch --parsable \
    --job-name="$name" \
    --output="$LOGS/${prefix}_e${et}_%j.out" \
    --error="$LOGS/${prefix}_e${et}_%j.err" \
    --export=ALL,ETA0="$eta",SEED=0,MAX_STEPS="$TARGET" \
    "$REPO/scans/jets/$slurm" 2>&1 || true
}

log() { echo "$(date -Is) $*"; }

log "START L4D128+gputest L6D256+ailab AdamW t=10 n100M ATTN-pool const target=$TARGET"

for iter in $(seq 1 "$MAX_ITERS"); do
  n_gt=$(partition_count gputest jc-awL4d128ap)
  n_ai=$(partition_count ailab jc-awL6d256ap)
  pending=0
  done_n=0
  status_line=""
  for cell in "${CELLS[@]}"; do
    IFS=: read -r depth width eta part slurm prefix tag <<<"$cell"
    name=$(job_name_for "$prefix" "$eta")
    step=$(cell_step "$depth" "$width" "$eta" "$tag")
    label="L${depth}D${width}e${eta}"
    if (( step >= TARGET )); then
      done_n=$((done_n + 1))
      status_line+=" ${label}=done(${step})"
      continue
    fi
    pending=$((pending + 1))
    if cell_queued "$name"; then
      status_line+=" ${label}=Q/R(${step})"
      continue
    fi
    if [[ "$part" == gputest ]]; then
      if (( n_gt < MAX_GPTEST )); then
        jid=$(submit_cell "$eta" "$slurm" "$prefix" "$part")
        if [[ "$jid" =~ ^[0-9]+$ ]]; then
          log "SUBMIT $label step=$step → $jid"
          n_gt=$((n_gt + 1))
          status_line+=" ${label}=sub(${step}→$jid)"
        else
          log "FAIL submit $label: $jid"
          status_line+=" ${label}=fail"
        fi
      else
        status_line+=" ${label}=wait(${step})"
      fi
    else
      if (( n_ai < MAX_AILAB )); then
        jid=$(submit_cell "$eta" "$slurm" "$prefix" "$part")
        if [[ "$jid" =~ ^[0-9]+$ ]]; then
          log "SUBMIT $label step=$step → $jid"
          n_ai=$((n_ai + 1))
          status_line+=" ${label}=sub(${step}→$jid)"
        else
          log "FAIL submit $label: $jid"
          status_line+=" ${label}=fail"
        fi
      else
        status_line+=" ${label}=wait(${step})"
      fi
    fi
  done
  log "iter=$iter gputest=$n_gt ailab=$n_ai done=$done_n/6 pending=$pending$status_line"
  if (( pending == 0 )); then
    log "ALL six cells reached $TARGET"
    exit 0
  fi
  sleep "$SLEEP_SEC"
done
log "FAIL MAX_ITERS"
exit 1
