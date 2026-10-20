#!/bin/bash
set -euo pipefail
HOST="${LAMBDA_HOST:-ubuntu@168.110.21.94}"
JET_LOCAL="${JET_LOCAL:-/projects/j/jdezoort/jetclass}"
REMOTE_DATA="${REMOTE_DATA:-/home/ubuntu/data/jetclass}"
FILES_PER_CLASS="${FILES_PER_CLASS:-21}"
PARALLEL="${PARALLEL:-6}"
LOG="${LOG:-/tmp/lambda_jetclass_stage.log}"
CLASSES=(HToBB HToCC HToGG HToWW2Q1L HToWW4Q TTBar TTBarLep WToQQ ZJetsToNuNu ZToQQ)
# Independent SSH per rsync — ControlMaster mux deadlocks under parallel bulk xfer
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ControlMaster=no"

echo "$(date -Iseconds) parallel-stage start files/class=$FILES_PER_CLASS P=$PARALLEL (no-mux)" >>"$LOG"
ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no "$HOST" \
  "mkdir -p $REMOTE_DATA/train_100M $REMOTE_DATA/manifests" >/dev/null

LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT
for ((i=0; i<FILES_PER_CLASS; i++)); do
  idx=$(printf '%03d' "$i")
  for c in "${CLASSES[@]}"; do
    echo "${c}_${idx}.root" >> "$LIST"
  done
done
echo "$(date -Iseconds) queue=$(wc -l <"$LIST")" >>"$LOG"

xfer_one() {
  local f="$1"
  local src="$JET_LOCAL/train_100M/$f"
  local dst="$REMOTE_DATA/train_100M/$f"
  local loc_sz rem_sz tries=0
  loc_sz=$(stat -c%s "$src")
  rem_sz=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no \
    "$HOST" "stat -c%s '$dst' 2>/dev/null || echo 0")
  if [[ "$rem_sz" == "$loc_sz" ]]; then
    echo "$(date -Iseconds) SKIP train/$f" >>"$LOG"
    return 0
  fi
  echo "$(date -Iseconds) XFER train/$f ($loc_sz bytes, have=$rem_sz)" >>"$LOG"
  while (( tries < 6 )); do
    if rsync -a --partial --timeout=300 -e "$RSYNC_RSH" "$src" "$HOST:$dst"; then
      echo "$(date -Iseconds) OK train/$f" >>"$LOG"
      return 0
    fi
    tries=$((tries+1))
    echo "$(date -Iseconds) RETRY train/$f try=$tries" >>"$LOG"
    sleep $((8*tries))
  done
  echo "$(date -Iseconds) FAIL train/$f" >>"$LOG"
  return 1
}
export -f xfer_one
export HOST JET_LOCAL REMOTE_DATA RSYNC_RSH LOG

set +e
cat "$LIST" | xargs -P "$PARALLEL" -n 1 -I{} bash -c 'xfer_one "$@"' _ {}
rc=$?
set -e
echo "$(date -Iseconds) STAGE_DONE rc=$rc" >>"$LOG"
exit $rc
