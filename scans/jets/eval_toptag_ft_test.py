#!/usr/bin/env python3
"""Full TopTagging test-set eval of a fine-tuned CAPEN ckpt (no training).

Reports test AUC + bg rejection @50% / @30% (ParT Table 5 metrics).
Invoked by ``eval_toptag_ft_test_gputest.slurm`` (must be a real .py file so
DataLoader workers can spawn).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.lit_models.lit_cpen_toptagging import LitCPENTopTagging
from scans.common.sweep_common import (
    _datamodule_factory_ctx,
    build_model,
    configure_graph_args,
    create_toptagging_datamodule,
    jetclass_live_graph_kwargs,
    resolve_part_interaction_preprocess,
)
from scans.jets.test_cpen_toptagging_ft import parse_args


def main() -> None:
    ckpt = Path(os.environ["CKPT"])
    n_test = int(os.environ.get("N_TEST", "400000"))
    batch = int(os.environ.get("BATCH", "256"))
    eta0 = float(os.environ.get("ETA0", "0.025"))
    ft_root = Path(
        os.environ.get(
            "FT_ROOT",
            "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune",
        )
    )

    sys.argv = [
        "eval_toptag_ft_test",
        "--model",
        "capen-llama-att",
        "--operators",
        "incidence",
        "--toptagging-live",
        "--star-radius",
        "0.1",
        "--live-knn-k",
        "7",
        "--live-star-kin7",
        "--jetclass-edge-features",
        "part-interaction",
        "--init-from",
        str(ckpt),  # satisfy CLI; FT weights loaded from CKPT below
        "--mode",
        "sweep_lr",
        "--optimizer",
        "adam",
        "--scheduler",
        "none",
        "--residual-structure",
        "transformer-like",
        "--etas",
        str(eta0),
        "--epochs",
        "1",
        "--n-test",
        str(n_test),
        "--n-val",
        "0",
        "--data-root",
        "/scratch/gpfs/BHANIN/jdezoort/datasets/toptagging",
        "--depth",
        "4",
        "--width",
        "256",
        "--heads",
        "8",
        "--dropout",
        "0.0",
        "--root",
        str(ft_root),
        "--batch-size",
        str(batch),
        "--n-gpus",
        "1",
        "--num-workers",
        "4",
        "--normalization",
        "uniform",
        "--attention-normalization",
        "gamma",
        "--num-particles",
        "80",
        "--seed",
        "0",
        "--layer-norm",
        "--no-torch-compile",
    ]
    args = parse_args()
    args.dataset = "toptagging"
    args.toptagging_live = True
    configure_graph_args(args)

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    step = int(payload.get("global_step", 0) or 0)
    epoch = int(payload.get("epoch", -1) or -1)
    sd = payload["state_dict"]
    args.gamma_11 = float(sd["model.attn_gamma_11"])
    args.gamma_12 = float(sd["model.attn_gamma_12"])
    args.gamma_21 = float(sd["model.attn_gamma_21"])
    args.gamma_22 = float(sd["model.attn_gamma_22"])
    print(f"[test] ckpt={ckpt} global_step={step} epoch={epoch}")
    print(
        f"[test] gammas from ckpt: "
        f"{args.gamma_11:.4f} {args.gamma_12:.4f} "
        f"{args.gamma_21:.4f} {args.gamma_22:.4f}"
    )

    def _dm_factory():
        return create_toptagging_datamodule(
            args, data_root=args.data_root or args.root, dense_pairs=False
        )

    token = _datamodule_factory_ctx.set(_dm_factory)
    try:
        resolve_part_interaction_preprocess(args, args.data_root or args.root)
    finally:
        _datamodule_factory_ctx.reset(token)

    dm = _dm_factory()
    dm.setup("test")
    n_test_actual = len(dm._test)
    print(f"[test] n_test={n_test_actual} batch={batch}")

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
    lit = LitCPENTopTagging(
        model=model,
        eta_0=eta0,
        model_name="capen-llama-att",
        optimizer="adam",
        scheduler="none",
        dataset="toptagging",
        depth=4,
        width=256,
        heads=8,
        corr=float(getattr(args, "corr", 1.0) or 1.0),
        **live,
    )
    missing, unexpected = lit.load_state_dict(sd, strict=False)
    print(f"[test] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[test] missing sample:", missing[:12])
    if unexpected:
        print("[test] unexpected sample:", unexpected[:12])
    lit.eval()

    p = torch.cuda.get_device_properties(0)
    gib = p.total_memory / 1024**3
    print(f"[test] GPU0={p.name} {gib:.1f} GiB")
    if gib < 70.0:
        raise SystemExit(f"Need A100 80GB; got {gib:.1f} GiB")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    print("[test] running trainer.test …")
    metrics = trainer.test(model=lit, datamodule=dm, verbose=True)

    flat: dict = {}
    if isinstance(metrics, list):
        for d in metrics:
            if isinstance(d, dict):
                flat.update(d)
    elif isinstance(metrics, dict):
        flat = metrics

    out = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ckpt": str(ckpt),
        "global_step": step,
        "epoch": epoch,
        "n_test": n_test,
        "n_test_actual": n_test_actual,
        "batch_size": batch,
        "eta0": eta0,
        "gammas": {
            "11": args.gamma_11,
            "12": args.gamma_12,
            "21": args.gamma_21,
            "22": args.gamma_22,
        },
        "metrics": flat,
        "part_reference": {
            "ParT-f.t. AUC": 0.9877,
            "ParT-f.t. Rej50": 691,
            "ParT-f.t. Rej30": 2766,
            "LorentzNet AUC": 0.9868,
            "LorentzNet Rej50": 498,
            "LorentzNet Rej30": 2195,
        },
        "note_val_rej_discreteness": (
            "Val Rej₃₀% on n_val=20k (n_bg≈9925) only takes values "
            "n_bg/{4,5,6,…} ≈ {2481,1985,1654,…}; not a logging bug."
        ),
    }
    manifest_name = os.environ.get(
        "MANIFEST_NAME", "ft_L4D256_full_eta0p025_best_test.json"
    )
    out_path = ft_root / "manifests" / manifest_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps(out, indent=2, default=str))
    print(f"[test] wrote {out_path}")

    auc = flat.get("test_roc_auc")
    r50 = flat.get("test_bg_rejection")
    r30 = flat.get("test_bg_rejection_0p3")
    print(
        f"[test] SUMMARY  AUC={auc}  Rej50%={r50}  Rej30%={r30}  "
        f"(ParT-f.t. 0.9877 / 691 / 2766)"
    )


if __name__ == "__main__":
    main()
