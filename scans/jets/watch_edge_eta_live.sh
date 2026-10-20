#!/bin/bash
# Poll Lambda edge η×D + hparam-xfer arms every 30s → live canvas + CSVs.
#
#   nohup bash scans/jets/watch_edge_eta_live.sh \
#     >~/.cache/edge_eta_monitor/watch.nohup 2>&1 &
#
set -uo pipefail

REPO="${REPO:-/home/jdezoort/coupled-particle-edge-networks}"
PY="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/edge_eta_monitor"
LOG="$CACHE/watch.log"
LOCK="$CACHE/watch.lock"
STOP="$CACHE/STOP"
SLEEP_SEC="${SLEEP_SEC:-30}"
UPDATE="$REPO/scans/jets/update_edge_eta_live_canvas.py"

mkdir -p "$CACHE"

# Prefer passphrase-unlocked agent.
if [[ -z "${SSH_AUTH_SOCK:-}" ]] || ! ssh-add -l >/dev/null 2>&1; then
  for s in /tmp/ssh-*/agent.* /tmp/ssh-agent-*.sock; do
    [[ -S "$s" ]] || continue
    if SSH_AUTH_SOCK="$s" ssh-add -l >/dev/null 2>&1; then
      export SSH_AUTH_SOCK="$s"
      break
    fi
  done
fi

if [[ -f "$LOCK" ]]; then
  old=$(cat "$LOCK" 2>/dev/null || true)
  if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
    echo "$(date -Iseconds) another edge-eta monitor pid=$old; exiting"
    exit 0
  fi
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "$(date -Iseconds) edge-eta+hxfer+constlr live monitor start SLEEP=$SLEEP_SEC sock=${SSH_AUTH_SOCK:-none}" | tee -a "$LOG"

while true; do
  if [[ -f "$STOP" ]]; then
    echo "$(date -Iseconds) STOP; exit" | tee -a "$LOG"
    exit 0
  fi
  if "$PY" "$UPDATE" >>"$LOG" 2>&1; then
    :
  else
    echo "$(date -Iseconds) update failed (see log)" | tee -a "$LOG"
  fi
  # Stop when edge 15/15 AND const-LR arms + Adam η₀×D complete.
  if [[ -f "$CACHE/status.json" ]]; then
    done_pair=$("$PY" -c "
import json
d=json.load(open('$CACHE/status.json'))
edge=sum(1 for c in d.get('cells',[]) if c['status']=='done')
cl=d.get('constlr_cells',[])
cl_done=sum(1 for c in cl if c['status']=='done')
cl384=d.get('constlr_d384_cells',[])
cl384_done=sum(1 for c in cl384 if c['status']=='done')
ae=d.get('adam_eta_cells',[])
ae_done=sum(1 for c in ae if c['status']=='done')
print(f'{edge},{cl_done},{len(cl)},{cl384_done},{len(cl384)},{ae_done},{len(ae)}')
" 2>/dev/null || echo "0,0,0,0,0,0,0")
    IFS=',' read -r edge_n cl_n cl_tot cl384_n cl384_tot ae_n ae_tot <<<"$done_pair"
    if [[ "$edge_n" == "15" && "$cl_tot" -ge 4 && "$cl_n" == "$cl_tot" \
          && "$cl384_tot" -ge 4 && "$cl384_n" == "$cl384_tot" \
          && "$ae_tot" -ge 8 && "$ae_n" == "$ae_tot" ]]; then
      echo "$(date -Iseconds) edge 15/15 + constlr ${cl_n}/${cl_tot} + d384 ${cl384_n}/${cl384_tot} + adam_eta ${ae_n}/${ae_tot} done; exit" | tee -a "$LOG"
      echo "EDGE_ETA_CONSTLR_COMPLETE $(date -Iseconds)" >"$CACHE/COMPLETE"
      exit 0
    fi
  fi
  sleep "$SLEEP_SEC"
done
