#!/usr/bin/env python
"""CPEN semisupervised sweeps on stellar-stream PyG graphs.

Example (all MOCK graphs; one optimizer step per graph):
  python scans/streams/test_cpen_stream.py \\
    --model cpen --operators incidence --mode sweep_lr --etas 0.1 \\
    --epochs 50 --batch-size 1 --n-gpus 1 --depth 4 --width 64 \\
    --stream-name all \\
    --data-root /projects/BHANIN/adri_gage_gnn_streams/gnn_streams/mocks/\\
streams_per_region_iter3_pm_stepping2_k4_edges_pmcoh/pyg_data_pmcoh_dir0p97_dlog0p25 \\
    --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs \\
    --no-torch-compile

Include real streams with ``--include-real-streams``.
Edge aux: ``--edge-aux product-consistency`` (BCE on stopgrad(p_u p_v)).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.lit_models.lit_cpen_stream import LitCPENStream
from cpen.utils.stream_graph_cache import DEFAULT_STREAM_DATA_DIR
from scans.common.sweep_common import (
    add_common_args,
    configure_graph_args,
    create_stream_datamodule,
    run_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(
        dataset="stream",
        batch_size=1,
        n_gpus=1,
        num_workers=0,
        normalization="uniform",
        operators="incidence",
        model="cpen",
        data_root=str(DEFAULT_STREAM_DATA_DIR),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = "stream"
    configure_graph_args(args)
    data_root = args.data_root or str(DEFAULT_STREAM_DATA_DIR)

    dm_box: dict = {}

    def datamodule_factory():
        dm = create_stream_datamodule(args, data_root=data_root)
        dm_box["dm"] = dm
        return dm

    def lit_factory(model, **kwargs):
        dm = dm_box.get("dm")
        weights = None if dm is None else dm.class_weights
        edge_weights = None if dm is None else getattr(dm, "edge_class_weights", None)
        return LitCPENStream(
            model,
            class_weights=weights,
            edge_class_weights=edge_weights,
            edge_loss_weight=float(getattr(args, "edge_loss_weight", 1.0) or 1.0),
            edge_aux=str(getattr(args, "edge_aux", "hard-ce") or "hard-ce"),
            **kwargs,
        )

    run_sweep(
        args,
        dataset="stream",
        datamodule_factory=datamodule_factory,
        lit_factory=lit_factory,
    )


if __name__ == "__main__":
    main()
