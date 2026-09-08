#!/usr/bin/env python
"""
TopTagging hyperparameter transfer sweeps for CPEN.

Examples:
  # kNN graphs (pre-build with build_toptagging_graphs.py)
  python scans/jets/test_cpen_toptagging.py \\
    --model cpen --operators incidence --graph-construction 8-NN \\
    --mode sweep_lr --etas 0.3 --data-root /path/to/toptagging ...

  # star-R hypergraphs (pre-build with build_toptagging_star_graphs.py)
  python scans/jets/test_cpen_toptagging.py \\
    --model cpen --operators incidence --star-radius 0.25 \\
    --mode sweep_lr --etas 0.3 --data-root /path/to/toptagging ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without editable install when launched from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.lit_models.lit_cpen_toptagging import LitCPENTopTagging
from scans.common.sweep_common import (
    _default_operators,
    add_common_args,
    configure_graph_args,
    create_toptagging_datamodule,
    run_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPEN TopTagging transfer sweeps")
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
