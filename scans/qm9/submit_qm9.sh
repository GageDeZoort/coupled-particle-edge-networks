#!/bin/bash
# Submit the QM9 dipole-moment transfer grid (does not wait).
#
#   D ∈ {128,256,512}  ×  η0 ∈ {0.05,0.25,1.0}  ×  target ∈ {mu}  ×  seed ∈ {0}
#
#   bash scans/qm9/submit_qm9.sh
#   DRY_RUN=1 bash scans/qm9/submit_qm9.sh
#   TARGETS=mu,gap WIDTHS=256 ETAS=0.25 SEEDS=0 bash scans/qm9/submit_qm9.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/test_cpen_qm9.slurm"

WIDTHS="${WIDTHS:-128,256,512}"
ETAS="${ETAS:-0.05,0.25,1.0}"
TARGETS="${TARGETS:-mu}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-300}"
DEPTH="${DEPTH:-6}"
BATCH_SIZE="${BATCH_SIZE:-128}"
QM9_LOSS="${QM9_LOSS:-l1}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a WIDTH_ARR <<< "${WIDTHS}"
IFS=',' read -r -a ETA_ARR <<< "${ETAS}"
IFS=',' read -r -a TARGET_ARR <<< "${TARGETS}"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

n_jobs=$(( ${#WIDTH_ARR[@]} * ${#ETA_ARR[@]} * ${#TARGET_ARR[@]} * ${#SEED_ARR[@]} ))

echo "QM9 regression grid"
echo "  widths=${WIDTHS}"
echo "  etas=${ETAS}"
echo "  targets=${TARGETS}"
echo "  seeds=${SEEDS}"
echo "  epochs=${EPOCHS} depth=${DEPTH} bs=${BATCH_SIZE} loss=${QM9_LOSS}"
echo "  n_jobs=${n_jobs}"
echo

for target in "${TARGET_ARR[@]}"; do
  target="${target// /}"
  for width in "${WIDTH_ARR[@]}"; do
    width="${width// /}"
    for eta in "${ETA_ARR[@]}"; do
      eta="${eta// /}"
      eta_tag="${eta//./p}"
      for seed in "${SEED_ARR[@]}"; do
        seed="${seed// /}"
        jname="qm9-${target}-d${width}-e${eta_tag}-s${seed}"
        export_vars="ALL,TARGET=${target},WIDTH=${width},ETAS=${eta},SEED=${seed},EPOCHS=${EPOCHS},DEPTH=${DEPTH},BATCH_SIZE=${BATCH_SIZE},QM9_LOSS=${QM9_LOSS}"
        if [[ "${DRY_RUN}" == "1" ]]; then
          echo "DRY  ${jname}  ${target}  D=${width}  eta0=${eta}  seed=${seed}"
        else
          jid=$(sbatch --parsable \
            --job-name="${jname}" \
            --export="${export_vars}" \
            "${SLURM_SCRIPT}" "${target}" "${width}" "${eta}" "${seed}")
          echo "  ${jname} -> ${jid}"
        fi
      done
    done
  done
done
echo "Track with: squeue -u \$USER"
