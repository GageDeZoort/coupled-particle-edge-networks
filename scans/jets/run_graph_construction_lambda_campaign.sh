#!/bin/bash
# Orchestrate refined graph-construction Lambda campaigns:
#   1) odd-k fill at R★=0.1  (k∈{1,3,5,7})
#   2) R★ sweep at k★=6
#
#   nohup bash scans/jets/run_graph_construction_lambda_campaign.sh \
#     >~/logs/graph_campaign_master.log 2>&1 &
#
set -uo pipefail
REPO="${HOME}/cpen"
mkdir -p ~/logs/graph_k ~/logs/graph_R
cd "$REPO"

echo "$(date -Iseconds) CAMPAIGN: odd-k fill k∈{1,3,5,7} @ R★=0.1"
KS_CSV=1,3,5,7 bash scans/jets/run_graph_k_lambda8.sh \
  >>~/logs/graph_k/master.log 2>&1
rc=$?
if (( rc != 0 )); then
  echo "$(date -Iseconds) CAMPAIGN FAIL at odd-k (rc=$rc)"
  exit $rc
fi

echo "$(date -Iseconds) CAMPAIGN: R★ sweep at k★=6"
bash scans/jets/run_graph_R_lambda8.sh \
  >>~/logs/graph_R/master.log 2>&1
rc=$?
echo "$(date -Iseconds) CAMPAIGN DONE (rc=$rc)"
exit $rc
