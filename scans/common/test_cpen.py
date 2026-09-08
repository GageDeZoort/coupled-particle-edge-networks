#!/usr/bin/env python
"""Unified CPEN/particle-only transfer sweeps across supported datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.lit_models.lit_cpen_mnist import LitCPENMNIST
from cpen.lit_models.lit_cpen_pascal import LitCPENPascal
from cpen.lit_models.lit_cpen_qm9 import LitCPENQM9
from cpen.lit_models.lit_cpen_toptagging import LitCPENTopTagging
from scans.common.sweep_common import (
    _default_operators,
    add_common_args,
    configure_graph_args,
    create_mnist_datamodule,
    create_pascal_datamodule,
    create_qm9_datamodule,
    create_toptagging_datamodule,
    run_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_graph_args(args)
    data_root = args.data_root or args.root

    if args.dataset in {"mnist-sp", "mnist"}:
        def lit_factory(model, **kwargs):
            return LitCPENMNIST(
                model,
                readout_mode=str(getattr(args, "readout_mode", "node") or "node"),
                **kwargs,
            )

        run_sweep(
            args,
            dataset="mnist",
            datamodule_factory=lambda: create_mnist_datamodule(
                args,
                data_root=data_root,
            ),
            lit_factory=lit_factory,
        )
        return

    if args.dataset == "qm9":
        from cpen.apps.qm9.qm9_graph_cache import qm9_target_stats, resolve_target

        target_name, _, target_unit = resolve_target(getattr(args, "qm9_target", None))
        target_mean, target_std = qm9_target_stats(data_root, target_name)

        def lit_factory(model, **kwargs):
            return LitCPENQM9(
                model,
                target_mean=target_mean,
                target_std=target_std,
                target_name=target_name,
                target_unit=target_unit,
                loss=str(getattr(args, "qm9_loss", "l1") or "l1"),
                readout_mode=str(getattr(args, "readout_mode", "graph") or "graph"),
                **kwargs,
            )

        run_sweep(
            args,
            dataset="qm9",
            datamodule_factory=lambda: create_qm9_datamodule(
                args,
                data_root=data_root,
            ),
            lit_factory=lit_factory,
        )
        return

    if args.dataset == "pascalvoc-sp":
        def lit_factory(model, **kwargs):
            return LitCPENPascal(
                model,
                edge_loss_weight=float(getattr(args, "edge_loss_weight", 1.0) or 1.0),
                edge_aux=(
                    "hard-ce"
                    if str(getattr(args, "readout_mode", "node") or "node") == "node+edge"
                    else "none"
                ),
                **kwargs,
            )

        run_sweep(
            args,
            dataset="pascal",
            datamodule_factory=lambda: create_pascal_datamodule(
                args,
                data_root=data_root,
            ),
            lit_factory=lit_factory,
        )
        return

    dense_pairs = args.model == "cpen" and _default_operators(args) == "pe-basis"
    run_sweep(
        args,
        dataset="toptagging",
        datamodule_factory=lambda: create_toptagging_datamodule(
            args,
            data_root=data_root,
            dense_pairs=dense_pairs,
        ),
        lit_factory=LitCPENTopTagging,
    )


if __name__ == "__main__":
    main()
