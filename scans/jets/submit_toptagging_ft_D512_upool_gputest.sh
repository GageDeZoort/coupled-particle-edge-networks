#!/bin/bash
# JetClass → TopTagging FT on Della gpu-test (≤59m).
# Init: hxfer upool D=512 η₀=0.1 @ 10k (v2stream last.ckpt).
#
#   bash scans/jets/submit_toptagging_ft_D512_upool_gputest.sh
#
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks

INIT_FROM="${INIT_FROM:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/init_ckpts/hxfer-upool-D512-eta0p1-v2stream-last.ckpt}"
FT_ROOT="${FT_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune}"
ETA0="${ETA0:-0.1}"
WIDTH=512
HEAD=16
DEPTH=4
NUM_PARTICLES=80
STAR_R="${STAR_R:-0.1}"
KNN_K="${KNN_K:-6}"
# ~0.5–0.7 s/step @ D=512 → ~5k steps fits comfortably in 59m after startup.
MAX_STEPS="${MAX_STEPS:-5000}"
BATCH="${BATCH:-128}"
N_GPUS="${N_GPUS:-4}"
UNIFORM_POOL="${UNIFORM_POOL:-1}"
RUN_TAG="${RUN_TAG:-ft-L4D512-upool-hxfer0p1-gputest-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-k${KNN_K}-R0p1}"

case "$FT_ROOT" in
  *jetclass_scaling_law*|*jetclass_output*)
    echo "ERROR: FT_ROOT=$FT_ROOT points at JetClass pretrain tree" >&2
    exit 1
    ;;
esac
if [[ ! -f "$INIT_FROM" ]]; then
  echo "ERROR: INIT_FROM missing: $INIT_FROM" >&2
  exit 1
fi

mkdir -p "$FT_ROOT/logs"

# Della routes time<=59m onto gputest; do NOT set partition=gputest.
out=$(sbatch \
  --job-name=tt-ft-D512-upool \
  --time=0:59:00 \
  --exclude=della-l02g6,della-l03g11 \
  --export=ALL,INIT_FROM="$INIT_FROM",FT_ROOT="$FT_ROOT",ETA0="$ETA0",WIDTH="$WIDTH",HEAD="$HEAD",DEPTH="$DEPTH",NUM_PARTICLES="$NUM_PARTICLES",STAR_R="$STAR_R",KNN_K="$KNN_K",MAX_STEPS="$MAX_STEPS",RUN_TAG="$RUN_TAG",BATCH="$BATCH",N_GPUS="$N_GPUS",UNIFORM_POOL="$UNIFORM_POOL" \
  scans/jets/finetune_toptagging_from_jetclass.slurm)
echo "$out"
echo "  INIT_FROM=$INIT_FROM"
echo "  FT_ROOT=$FT_ROOT"
echo "  L=$DEPTH D=$WIDTH H=$HEAD η₀=$ETA0 R=$STAR_R k=$KNN_K uniform_pool=$UNIFORM_POOL"
echo "  max_steps=$MAX_STEPS B=${BATCH}×${N_GPUS} run_tag=$RUN_TAG"
squeue -u "$USER" -o '%.12i %.9P %.40j %.2t %.10M %R' | head -20
