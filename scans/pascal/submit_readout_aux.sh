#!/bin/bash
# Submit the PascalVOC-SP readout-asymmetry grid (does not wait).
#
#   D ∈ {64,128,256}  ×  η0 ∈ {0.05,0.1,0.25,0.5,1.0}
#     ×  {node, node+edge}  ×  seed ∈ {0,1,2}
#   = 90 jobs. Della gputest QOS MaxJobsPU≈3; the rest queue.
#   Smoke: SEEDS=0 bash scans/pascal/submit_readout_aux.sh   # 30 jobs
#
# From repo root:
#   bash scans/pascal/submit_readout_aux.sh
#   DRY_RUN=1 bash scans/pascal/submit_readout_aux.sh
#   WIDTHS=128 ETAS=0.25 READOUTS=node+edge SEEDS=0,1,2 bash scans/pascal/submit_readout_aux.sh
#
# Positionals on the slurm script (readout, width, eta0, seed) are the source of
# truth; --export is extra (Della sometimes drops --export=ALL).
# Run dirs are tagged seed{N} so repeats do not resume each other.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/test_cpen_readout_aux.slurm"

WIDTHS="${WIDTHS:-64,128,256}"
ETAS="${ETAS:-0.05,0.1,0.25,0.5,1.0}"
READOUTS="${READOUTS:-node,node+edge}"
SEEDS="${SEEDS:-0,1,2}"
EPOCHS="${EPOCHS:-5}"
DEPTH="${DEPTH:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EDGE_LOSS_WEIGHT="${EDGE_LOSS_WEIGHT:-1.0}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a WIDTH_ARR <<< "${WIDTHS}"
IFS=',' read -r -a ETA_ARR <<< "${ETAS}"
IFS=',' read -r -a READOUT_ARR <<< "${READOUTS}"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

n_jobs=$(( ${#WIDTH_ARR[@]} * ${#ETA_ARR[@]} * ${#READOUT_ARR[@]} * ${#SEED_ARR[@]} ))

echo "Pascal readout-aux grid"
echo "  widths=${WIDTHS}"
echo "  etas=${ETAS}"
echo "  readouts=${READOUTS}"
echo "  seeds=${SEEDS}"
echo "  epochs=${EPOCHS} depth=${DEPTH} bs=${BATCH_SIZE} lambda=${EDGE_LOSS_WEIGHT}"
echo "  n_jobs=${n_jobs}"
echo

node_ids=()
edge_ids=()

for mode in "${READOUT_ARR[@]}"; do
  mode="${mode// /}"
  for width in "${WIDTH_ARR[@]}"; do
    width="${width// /}"
    for eta in "${ETA_ARR[@]}"; do
      eta="${eta// /}"
      eta_tag="${eta//./p}"
      for seed in "${SEED_ARR[@]}"; do
        seed="${seed// /}"
        if [[ "${mode}" == "node+edge" ]]; then
          jname="voc-eb-d${width}-e${eta_tag}-s${seed}"
        else
          jname="voc-node-d${width}-e${eta_tag}-s${seed}"
        fi
        export_vars="ALL,READOUT_MODE=${mode},WIDTH=${width},ETAS=${eta},SEED=${seed},EPOCHS=${EPOCHS},DEPTH=${DEPTH},BATCH_SIZE=${BATCH_SIZE},EDGE_LOSS_WEIGHT=${EDGE_LOSS_WEIGHT}"
        if [[ "${DRY_RUN}" == "1" ]]; then
          echo "DRY  ${jname}  ${mode}  D=${width}  eta0=${eta}  seed=${seed}"
          jid="dry"
        else
          jid=$(sbatch --parsable \
            --job-name="${jname}" \
            --export="${export_vars}" \
            "${SLURM_SCRIPT}" "${mode}" "${width}" "${eta}" "${seed}")
          echo "  ${jname} -> ${jid}"
        fi
        if [[ "${mode}" == "node+edge" ]]; then
          edge_ids+=("${jid}")
        else
          node_ids+=("${jid}")
        fi
      done
    done
  done
done

echo
echo "NODE_ONLY = ${node_ids[*]}"
echo "NODE_PLUS_BOUNDARY = ${edge_ids[*]}"
echo "Notebook collates by globbing run dirs (seed{N} in the name); job ids are optional."
echo "Track with: squeue -u \$USER"
