#!/usr/bin/env python
"""CPEN node+edge training on stage-4 stellar-stream blob graphs.

One GMM blob per optimizer step, minimizing the multi-objective

    L = node_CE(is_mock_stream) + edge_loss_weight * edge_CE(edge_y)

Train is rebalanced (mock-rich hosts vs size-matched empty blobs) because the
raw pool is ~97% empty; val/test stay at natural occupancy.

Example::

    python scans/streams/test_cpen_blob.py \\
      --mode sweep_lr --etas 0.05 --epochs 30 --depth 4 --width 256 \\
      --blob-rich-min-mock 100 --blob-empty-per-rich 1.0 \\
      --data-root /scratch/gpfs/BHANIN/jgdezoort/streams/galaxies \\
      --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/blobs --no-torch-compile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.apps.streams.blob_datamodule import BlobStreamDatamodule
from cpen.apps.streams.blob_graph_cache import DEFAULT_BLOB_DATA_DIR
from cpen.lit_models.lit_cpen_stream import LitCPENStream
from scans.common.sweep_common import add_common_args, configure_graph_args, run_sweep


def add_blob_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--blob-rich-min-mock",
        type=int,
        default=100,
        help="Mock-rich threshold: a train graph is a positive host when "
        "n_mock >= this. Default: 100.",
    )
    parser.add_argument(
        "--blob-empty-per-rich",
        type=float,
        default=1.0,
        help="Empty (n_mock=0) train graphs per rich graph. 1.0 = 50/50. "
        "Negatives are size-matched to the rich blobs. Default: 1.0.",
    )
    parser.add_argument(
        "--blob-include-dilute",
        action="store_true",
        help="Also train on graphs with 1..rich_min_mock-1 mock stars "
        "(weak signal; excluded by default).",
    )
    parser.add_argument(
        "--blob-min-stars",
        type=int,
        default=16,
        help="Drop blobs smaller than this from every split (kNN k=8 is "
        "degenerate below ~9 stars). Default: 16.",
    )
    parser.add_argument(
        "--blob-val-graphs",
        type=int,
        default=2000,
        help="Cap on val graphs, sampled at the natural empty/host ratio. "
        "<=0 uses every val graph. Default: 2000.",
    )
    parser.add_argument(
        "--blob-test-graphs",
        type=int,
        default=0,
        help="Cap on test graphs (natural ratio). <=0 uses all. Default: 0.",
    )
    parser.add_argument(
        "--blob-node-frame",
        type=str,
        default="local",
        choices=("local", "raw"),
        help="'local' (default) replaces absolute DES sky position with "
        "blob-centered tangent-plane coords; 'raw' keeps the stored columns.",
    )
    parser.add_argument(
        "--blob-drop-features",
        type=str,
        default="parallax",
        help="Node features to withhold. Parallax is dropped by default: mock "
        "stars are injected with essentially noiseless Gaia parallax, which "
        "alone scores ~0.91 AUROC with no learning. Proper motion is kept — "
        "streams really are cold in PM. Aliases: parallax, pm, pmra, pmdec, "
        "mag, sky. Edge features derived from a dropped column go too. "
        "Pass 'none' to keep everything.",
    )
    parser.add_argument(
        "--blob-no-hyperedges",
        action="store_true",
        help="Train on kNN 2-edges only (skip sky/PM hyperedges).",
    )
    parser.add_argument(
        "--blob-galaxies",
        type=str,
        default=None,
        help="Galaxy ids (comma- or space-separated) to restrict every split "
        "to, e.g. 0001,0002. Default: all galaxies present.",
    )
    parser.add_argument(
        "--blob-split-seed",
        type=int,
        default=0,
        help="Seed for empty-negative sampling and eval subsampling. Default: 0.",
    )
    parser.add_argument(
        "--blob-max-cache",
        type=int,
        default=4096,
        help="Max payloads held in memory per dataset. Default: 4096.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_blob_args(parser)
    parser.set_defaults(
        dataset="stream",
        batch_size=1,
        n_gpus=1,
        num_workers=0,
        normalization="uniform",
        operators="incidence",
        model="cpen",
        data_root=str(DEFAULT_BLOB_DATA_DIR),
        edge_aux="hard+product",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 'stream' drives readout_mode=node+edge, sparse incidence, and the
    # node+edge checkpoint defaults in sweep_common; blob_run keeps the run dir
    # from colliding with per-MOCK stream runs at the same depth/width.
    args.dataset = "stream"
    args.blob_run = True
    configure_graph_args(args)
    data_root = args.data_root or str(DEFAULT_BLOB_DATA_DIR)

    dm_box: dict = {}

    def datamodule_factory():
        dm = BlobStreamDatamodule(
            data_root=data_root,
            batch_size=1,
            num_workers=args.num_workers if args.num_workers is not None else 0,
            node_frame=args.blob_node_frame,
            drop_features=args.blob_drop_features,
            rich_min_mock=int(args.blob_rich_min_mock),
            empty_per_rich=float(args.blob_empty_per_rich),
            include_dilute=bool(args.blob_include_dilute),
            min_stars=int(args.blob_min_stars),
            max_val_graphs=int(args.blob_val_graphs) or None,
            max_test_graphs=int(args.blob_test_graphs) or None,
            galaxy_ids=args.blob_galaxies,
            split_seed=int(args.blob_split_seed),
            include_incidence=True,
            include_edge_features=True,
            include_hyperedges=not bool(args.blob_no_hyperedges),
            max_cache=int(args.blob_max_cache),
        )
        dm_box["dm"] = dm
        return dm

    def lit_factory(model, **kwargs):
        dm = dm_box.get("dm")
        return LitCPENStream(
            model,
            class_weights=None if dm is None else dm.class_weights,
            edge_class_weights=None if dm is None else dm.edge_class_weights,
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
