#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=capen-att-hier
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0

# CAPEN-Llama-att on hierarchical TopTagging caches
# (kNN 2-edges + DBSCAN hypers + virtual nodes/edges).
#
# Cache (ParT-int + L2 edges), already built:
#   /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging/processed/hier_k8_eps0p08_ms2_vn1_ve1_mdb32_pint/n128/
# Selected by --data-root + --hier-k/eps/vn/ve/mdb + --num-particles 128
# (tag suffix _pint comes from hierarchical_construction_tag).
#
# Model differs from default CAPEN-Llama:
#   * class-token MHSA readout: 1/(D σ) · MHSA(Q={QCD,top}, KV=[X;E])
#   * M_22 only among hyperedges (degree > 2)
#   * --hyperedge-only drops all pairwise 2-edges at train time
#     (kNN, vn_link, deg≤2; no cache rebuild). Virtual nodes stay in X.
#
# Attention γ_rs: estimated from the hierarchical cache at job start
# (same supports as training: all-to-all M11, incidence M12/M21,
# hyperedge-only M22, drop pairwise 2-edges when --hyperedge-only). Logged to this
# Slurm stdout and the run parquet. Opt out with --no-estimate-gamma-rs
# and explicit --gamma-11/12/21/22.
#
# Launch: 1 task, 4 GPUs, `python ... --n-gpus 4` → Lightning ddp-spawn.
# --partition=a100 + --time=0:59:00 is how Della accepts a gputest job
# (do not set --partition=gpu or --partition=gputest; both are rejected).
# Keep --constraint=gpu80 (avoids 40GB/MIG).
#
# Optional overrides:
#   sbatch --export=ALL,ETA0=0.25,WIDTH=256,EPOCHS=35 \
#     scans/jets/test_capen_llama_att_toptagging_hier.slurm

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "hier cache: /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging/processed/hier_k8_eps0p08_ms2_vn1_ve1_mdb32_pint/n128"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
props = torch.cuda.get_device_properties(0)
gib = props.total_memory / 1024**3
print(
    f"GPU0 {props.name} {gib:.1f} GiB  "
    f"count={torch.cuda.device_count()}  "
    f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
)
if gib < 70:
    sys.exit(
        f"Need A100 80GB (#SBATCH --constraint=gpu80); this node has {gib:.1f} GiB"
    )
PY

python test_cpen_toptagging.py \
  --model capen-llama-att \
  --operators incidence \
  --hier-k 8 \
  --hier-eps 0.08 \
  --hier-min-samples 2 \
  --hier-vn 1 \
  --hier-ve 1 \
  --hier-mdb 32 \
  --mode sweep_lr \
  --optimizer adam \
  --residual-structure transformer-like \
  --etas "1" \
  --epochs "2" \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging \
  --depth "4" --width "192" --heads "6" \
  --dropout 0.1 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs \
  --batch-size 128 \
  --n-gpus 4 \
  --num-workers 8 \
  --normalization uniform \
  --num-particles 128 \
  --log-every-n-steps 100 \
  --attention-normalization gamma \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --no-torch-compile \
  --hyperedge-only
