#!/bin/bash
# Submit stages 0→4 for one split, each waiting on the previous array.
#
#   bash scans/streams/submit_pipeline.sh              # train, array 0-18
#   bash scans/streams/submit_pipeline.sh val 0-9
#   bash scans/streams/submit_pipeline.sh test 0-9
#   REBUILD=1 bash scans/streams/submit_pipeline.sh train 0-18
set -euo pipefail

SPLIT="${1:-${SPLIT:-train}}"
ARRAY="${2:-${ARRAY:-0-18}}"
REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
EXPORT="ALL,SPLIT=${SPLIT}"
if [[ "${REBUILD:-0}" == "1" ]]; then
  EXPORT="${EXPORT},REBUILD=1"
fi

cd "${REPO_ROOT}"
echo "Submitting ${SPLIT} array=${ARRAY}"

j0=$(sbatch --parsable --export="${EXPORT}" --array="${ARRAY}" scans/streams/0_preprocess_stream_df.slurm)
# aftercorr: array task i of stage n waits only on task i of stage n-1.
# afterok on the whole array never starts if any single galaxy fails.
j1=$(sbatch --parsable --export="${EXPORT}" --array="${ARRAY}" --dependency=aftercorr:"${j0}" scans/streams/1_physical_cuts.slurm)
j2=$(sbatch --parsable --export="${EXPORT}" --array="${ARRAY}" --dependency=aftercorr:"${j1}" scans/streams/2_fit_gmm.slurm)
j3=$(sbatch --parsable --export="${EXPORT}" --array="${ARRAY}" --dependency=aftercorr:"${j2}" scans/streams/3_split_blobs.slurm)
j4=$(sbatch --parsable --export="${EXPORT}" --array="${ARRAY}" --dependency=aftercorr:"${j3}" scans/streams/4_build_graphs.slurm)

echo "stage0 cells  ${j0}"
echo "stage1 phys   ${j1}  aftercorr:${j0}"
echo "stage2 gmm    ${j2}  aftercorr:${j1}"
echo "stage3 blobs  ${j3}  aftercorr:${j2}"
echo "stage4 graphs ${j4}  aftercorr:${j3}"
