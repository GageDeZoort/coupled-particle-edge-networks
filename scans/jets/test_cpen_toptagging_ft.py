#!/usr/bin/env python
"""
JetClass → TopTagging fine-tune (weights-only warm start).

Requires ``--init-from`` (a JetClass pretrain Lightning ``.ckpt``) and
``--toptagging-live`` so graphs match the JetClass live star+$R$/kNN recipe.

Checkpoints write under ``--root`` (default study tree
``…/toptagging_finetune``), never into the JetClass pretrain directory.

Example:
  python scans/jets/test_cpen_toptagging_ft.py \\
    --init-from /path/to/jetclass/last.ckpt \\
    --toptagging-live --star-radius 0.2 --live-knn-k 6 \\
    --jetclass-edge-features part-interaction \\
    --num-particles 80 --etas 0.5 \\
    --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune \\
    --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging \\
    ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.lit_models.lit_cpen_toptagging import LitCPENTopTagging
from cpen.training.init_from import assert_ft_root_not_pretrain, read_ckpt_eta0
from scans.common.sweep_common import (
    _default_operators,
    add_common_args,
    configure_graph_args,
    create_toptagging_datamodule,
    run_sweep,
)

DEFAULT_FT_ROOT = "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune"
DEFAULT_DATA_ROOT = "/scratch/gpfs/BHANIN/jdezoort/datasets/toptagging"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune CAPEN on TopTagging from a JetClass pretrain ckpt"
    )
    add_common_args(parser)
    # Distinguish "user omitted --etas" from bytecode's multi-η default list.
    for action in parser._actions:
        if getattr(action, "dest", None) == "etas":
            action.default = None
            break
    parser.set_defaults(
        root=DEFAULT_FT_ROOT,
        data_root=DEFAULT_DATA_ROOT,
        toptagging_live=True,
        jetclass_edge_features="part-interaction",
        star_radius=0.2,
        live_knn_k=6,
        num_particles=80,
        checkpoint_monitor="val_roc_auc",
        checkpoint_mode="max",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = "toptagging"
    args.toptagging_live = True

    if not getattr(args, "init_from", None):
        raise SystemExit(
            "TopTagging fine-tune requires --init-from <jetclass.ckpt> "
            "(weights-only warm start from JetClass pretrain)."
        )
    assert_ft_root_not_pretrain(args.root)

    # Default FT η₀ to the pretrain η₀ when the user did not pass --etas.
    raw_etas = getattr(args, "etas", None)
    etas_unset = raw_etas is None or (
        isinstance(raw_etas, str) and not str(raw_etas).strip()
    )
    if etas_unset:
        pre_eta = read_ckpt_eta0(args.init_from)
        if pre_eta is None:
            raise SystemExit(
                "Could not read eta_0 from --init-from ckpt; pass --etas explicitly."
            )
        args.etas = str(pre_eta)
        print(f"[ft] defaulting --etas to pretrain eta_0={pre_eta:g}")

    configure_graph_args(args)
    data_root = args.data_root or args.root
    dense_pairs = args.model == "cpen" and _default_operators(args) == "pe-basis"

    def datamodule_factory():
        return create_toptagging_datamodule(
            args,
            data_root=data_root,
            dense_pairs=dense_pairs,
        )

    run_sweep(
        args,
        dataset="toptagging",
        datamodule_factory=datamodule_factory,
        lit_factory=LitCPENTopTagging,
    )


if __name__ == "__main__":
    main()
