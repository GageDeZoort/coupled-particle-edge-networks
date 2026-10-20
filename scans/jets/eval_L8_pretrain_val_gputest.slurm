#!/bin/bash
# Quick validation pass on finished L8/D1024 pretrain ckpt (no training).
# Rebuilds the Lit module via the same CLI path as training (ckpt has no
# hyper_parameters), loads state_dict, estimates part-interaction preprocess,
# then trainer.validate on n_val jets.
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-L8-val
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/logs/jc_L8_val_gt_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/logs/jc_L8_val_gt_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain
CKPT="$ROOT/jetclass_output/adam/sweep_lr/capen-llama-att_8_1024_0p1_ln_op-incidence_onorm-degree_res-transformer-like_h32_ntr10240k_nval0_ntst0_nf7_gstar0p1_stream-kin7-partint-p80-knn7-noloop-sk7-D10p24M-s80k-seed0-pretrain-L8D1024-b32-g4-s80000-dh32-k7-R0p1-eta0p1-s80k/last.ckpt"
N_VAL="${N_VAL:-25000}"
BATCH="${BATCH:-32}"

echo "Host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "ckpt=$CKPT n_val=$N_VAL batch=$BATCH"
test -f "$CKPT" || { echo "missing ckpt"; exit 1; }

python - <<PY
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import torch

from cpen.lit_models.lit_cpen_jetclass import LitCPENJetClass
from scans.common.sweep_common import (
    _datamodule_factory_ctx,
    build_model,
    configure_graph_args,
    create_jetclass_datamodule,
    jetclass_live_graph_kwargs,
    resolve_part_interaction_preprocess,
)
from scans.jets.test_cpen_jetclass import parse_args

ckpt = Path("${CKPT}")
n_val = int("${N_VAL}")
batch = int("${BATCH}")
root = Path("${ROOT}")

# Same architecture / live-graph flags as L8 const-LR pretrain.
sys.argv = [
    "eval_L8_pretrain_val",
    "--model", "capen-llama-att",
    "--operators", "incidence",
    "--jetclass-stream",
    "--jetclass-raw-root", "/projects/j/jdezoort/jetclass",
    "--jetclass-features", "kin7",
    "--jetclass-edge-features", "part-interaction",
    "--live-knn-k", "7",
    "--star-radius", "0.1",
    "--live-star-kin7",
    "--mode", "sweep_lr",
    "--optimizer", "adam",
    "--scheduler", "none",
    "--residual-structure", "transformer-like",
    "--etas", "0.1",
    "--epochs", "1",
    "--n-train", "10240000",
    "--n-test", "0",
    "--n-val", str(n_val),
    "--data-root", "/scratch/gpfs/BHANIN/jdezoort/datasets/jetclass",
    "--depth", "8",
    "--width", "1024",
    "--heads", "32",
    "--dropout", "0.0",
    "--root", str(root),
    "--batch-size", str(batch),
    "--n-gpus", "1",
    "--num-workers", "4",
    "--normalization", "uniform",
    "--attention-normalization", "gamma",
    "--num-particles", "80",
    "--seed", "0",
    "--layer-norm",
    "--no-torch-compile",
]
args = parse_args()
args.dataset = "jetclass"
configure_graph_args(args)

payload = torch.load(ckpt, map_location="cpu", weights_only=False)
step = int(payload.get("global_step", 0) or 0)
sd = payload["state_dict"]
# Freeze γ to the trained values (skip re-estimation).
args.gamma_11 = float(sd["model.attn_gamma_11"])
args.gamma_12 = float(sd["model.attn_gamma_12"])
args.gamma_21 = float(sd["model.attn_gamma_21"])
args.gamma_22 = float(sd["model.attn_gamma_22"])
print(f"[val] ckpt={ckpt} global_step={step}")
print(
    f"[val] gammas from ckpt: "
    f"{args.gamma_11:.4f} {args.gamma_12:.4f} {args.gamma_21:.4f} {args.gamma_22:.4f}"
)

def _dm_factory():
    return create_jetclass_datamodule(args, data_root=args.data_root or args.root)

token = _datamodule_factory_ctx.set(_dm_factory)
try:
    # Match train_one_run: estimate part-interaction Θ(1) on live graph.
    resolve_part_interaction_preprocess(args, args.data_root or args.root)
finally:
    _datamodule_factory_ctx.reset(token)

dm = _dm_factory()
dm.corr_adam = float(getattr(args, "corr", 1.0) or 1.0)
dm.corr_sgd = dm.corr_adam

n_feat, out_dim = dm.get_dims()
model = build_model(
    args,
    n_features=n_feat,
    n_edge_features=dm.get_edge_dim(),
    out_dim=out_dim,
    depth=int(args.depth),
    width=int(args.width),
    heads=int(args.heads),
)
live = jetclass_live_graph_kwargs(args)
lit = LitCPENJetClass(
    model=model,
    eta_0=0.1,
    model_name="capen-llama-att",
    optimizer="adam",
    scheduler="none",
    dataset="jetclass",
    depth=8,
    width=1024,
    heads=32,
    corr=float(getattr(args, "corr", 1.0) or 1.0),
    **live,
)
missing, unexpected = lit.load_state_dict(sd, strict=False)
print(f"[val] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
if missing:
    print("[val] missing sample:", missing[:8])
if unexpected:
    print("[val] unexpected sample:", unexpected[:8])
lit.eval()

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    precision="32-true",
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
)
print("[val] running trainer.validate …")
metrics = trainer.validate(model=lit, datamodule=dm, verbose=True)
out = {
    "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
    "ckpt": str(ckpt),
    "global_step": step,
    "n_val": n_val,
    "batch_size": batch,
    "gammas": {
        "11": args.gamma_11,
        "12": args.gamma_12,
        "21": args.gamma_21,
        "22": args.gamma_22,
    },
    "metrics": metrics,
}
out_path = root / "manifests" / "L8D1024_val_at_80k.json"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(out, indent=2, default=str) + "\n")
print(json.dumps(out, indent=2, default=str))
print(f"[val] wrote {out_path}")
PY

echo "done L8 val"
