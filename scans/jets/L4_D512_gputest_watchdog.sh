#!/bin/bash
# Resubmit L4 D=512 H=16 baseline (corr=1) B=128×4 gpu-test chain until TARGET.
#
# Prefer:
#   ETAS="0.1 0.25 0.5" TARGET=100000 bash scans/jets/start_L4_D512_gputest_watchdogs.sh
# Or one arm:
#   ETA0=0.25 TARGET=100000 SLEEP_SEC=900 nohup bash scans/jets/L4_D512_gputest_watchdog.sh &
#
# Raising TARGET past 45k changes the auto run-dir tag (s45k→s100k). Before each
# submit we rename an existing s45k dir to s100k (when idle) so Lightning resumes.
#
set -uo pipefail

ETA0="${ETA0:-0.25}"
ETA_TAG=$(python3 -c "e=float('${ETA0}'); s=f'{e:g}'.replace('.','p'); print(s)")

JOB_NAME="${JOB_NAME:-jc-L4D512-gt-e${ETA_TAG}}"
RUN_TAG="${RUN_TAG:-L4D512-gputest-b128-g4-e${ETA_TAG}}"

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
SLURM=/home/jdezoort/coupled-particle-edge-networks/scans/jets/scaling_law_L4_D512_b128_s45k_gputest.slurm
LOG_CANDIDATES=(
  "$ROOT/logs/watchdog_L4D512_gputest_e${ETA_TAG}.log"
  "${HOME}/.cache/jc_L4D512_watchdogs/watchdog_e${ETA_TAG}.log"
)
TARGET="${TARGET:-100000}"
MAX_STEPS="${MAX_STEPS:-$TARGET}"
MAX_RESUBMITS="${MAX_RESUBMITS:-200}"
SLEEP_SEC="${SLEEP_SEC:-900}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
LOG=""
for cand in "${LOG_CANDIDATES[@]}"; do
  mkdir -p "$(dirname "$cand")" 2>/dev/null || true
  if ( : >>"$cand" ) 2>/dev/null; then
    LOG="$cand"
    break
  fi
done
if [[ -z "$LOG" ]]; then
  echo "ERROR: cannot write watchdog log" >&2
  exit 1
fi

# Baseline runs have no c* tag; isolate via run_tag (includes eta).
RUN_GLOB="$ROOT/jetclass_output/adam/sweep_lr/capen-llama-att_4_512_${ETA_TAG}_*${RUN_TAG}*"

migrate_run_dir_for_target() {
  # Only migrate when the *submit* budget (MAX_STEPS) exceeds 45k.
  local sweep="$ROOT/jetclass_output/adam/sweep_lr"
  local old new stag
  # shellcheck disable=SC2086
  old=$(ls -d ${sweep}/capen-llama-att_4_512_${ETA_TAG}_*Dfull-s45k*${RUN_TAG}* 2>/dev/null | head -1 || true)
  stag=$("$PYTHON" -c "n=int('${MAX_STEPS}'); print(f's{n//1000}k' if n%1000==0 and n>=1000 else f's{n}')")
  if [[ -z "$old" || ! -d "$old" ]]; then
    return 0
  fi
  if [[ "$old" == *"-${stag}-"* ]] || [[ "$old" == *"_${stag}_"* ]] || [[ "$old" == *"-${stag}_"* ]]; then
    return 0
  fi
  new="${old/Dfull-s45k/Dfull-${stag}}"
  if [[ "$new" == "$old" || -e "$new" ]]; then
    [[ -e "$new" ]] && echo "$(date -Iseconds) migrate skip: dest exists $new" | tee -a "$LOG"
    return 0
  fi
  echo "$(date -Iseconds) migrate run dir: $old -> $new" | tee -a "$LOG"
  mv "$old" "$new"
  if [[ -f "${old}.parquet" && ! -e "${new}.parquet" ]]; then
    mv "${old}.parquet" "${new}.parquet"
  fi
}

current_step() {
  "$PYTHON" - "$RUN_GLOB" <<'PY'
import glob, sys
from pathlib import Path
pats = sys.argv[1]
cands = sorted(glob.glob(pats))
best = 0

def step_from_ckpt(ckpt: Path) -> int:
    try:
        import torch
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        return int(payload.get("global_step", 0) or 0)
    except Exception:
        return 0

for c in cands:
    p = Path(c)
    if p.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(p, columns=["global_step"])
            col = t.column("global_step")
            if len(col):
                best = max(best, int(col[-1].as_py()))
        except Exception:
            pass
        continue
    if not p.is_dir():
        continue
    for ckpt in p.glob("last*.ckpt"):
        best = max(best, step_from_ckpt(ckpt))
    pq = p.parent / f"{p.name}.parquet"
    if pq.exists():
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(pq, columns=["global_step"])
            col = t.column("global_step")
            if len(col):
                best = max(best, int(col[-1].as_py()))
        except Exception:
            pass
print(best)
PY
}

latest_val() {
  "$PYTHON" - "$RUN_GLOB" <<'PY'
import glob, sys
from pathlib import Path
pats = sys.argv[1]
rows = []
for c in sorted(glob.glob(pats)):
    p = Path(c)
    pq = p if p.suffix == ".parquet" else p.parent / f"{p.name}.parquet"
    if not pq.exists():
        continue
    try:
        import pyarrow.parquet as pq
        df = pq.read_table(pq).to_pandas()
        sub = df.dropna(subset=["val_loss"]) if "val_loss" in df.columns else df.iloc[0:0]
        if len(sub):
            r = sub.iloc[-1]
            rows.append(
                (
                    int(r["global_step"]),
                    float(r["val_loss"]),
                    float(r.get("val_roc_auc", float("nan"))),
                    float(r.get("val_acc", float("nan"))),
                )
            )
    except Exception:
        pass
if not rows:
    print("none")
else:
    s, loss, roc, acc = max(rows, key=lambda x: x[0])
    print(f"step={s} val_loss={loss:.4f} val_roc={roc:.4f} val_acc={acc:.4f}")
PY
}

job_running() {
  timeout 30 squeue -u jdezoort -n "$JOB_NAME" -h -o '%i' 2>/dev/null | head -1 || true
}

n_subs=0
echo "$(date -Iseconds) watchdog start ETA0=$ETA0 ETA_TAG=$ETA_TAG JOB_NAME=$JOB_NAME RUN_TAG=$RUN_TAG TARGET=$TARGET MAX_STEPS=$MAX_STEPS MAX_RESUBMITS=$MAX_RESUBMITS RUN_GLOB=$RUN_GLOB" | tee -a "$LOG"

while true; do
  step=$(current_step)
  val=$(latest_val)
  jid=$(job_running || true)
  echo "$(date -Iseconds) step=$step $val job=${jid:-none} n_subs=$n_subs" | tee -a "$LOG"

  if (( step >= TARGET )); then
    echo "$(date -Iseconds) TARGET $TARGET reached (step=$step); exiting" | tee -a "$LOG"
    exit 0
  fi

  if [[ -n "${jid:-}" ]]; then
    sleep "$SLEEP_SEC"
    continue
  fi

  if (( n_subs >= MAX_RESUBMITS )); then
    echo "$(date -Iseconds) hit MAX_RESUBMITS=$MAX_RESUBMITS at step=$step; exiting" | tee -a "$LOG"
    exit 0
  fi

  migrate_run_dir_for_target || true

  out=$(timeout 60 sbatch \
    --job-name="$JOB_NAME" \
    --exclude=della-l02g6,della-l03g11 \
    --export=ALL,ETA0="$ETA0",RUN_TAG="$RUN_TAG",MAX_STEPS="$MAX_STEPS",BATCH=128 \
    "$SLURM" 2>&1) || {
    echo "$(date -Iseconds) sbatch failed: $out" | tee -a "$LOG"
    sleep "$SLEEP_SEC"
    continue
  }
  n_subs=$((n_subs + 1))
  echo "$(date -Iseconds) submitted ($n_subs): $out" | tee -a "$LOG"
  sleep "$SLEEP_SEC"
done
