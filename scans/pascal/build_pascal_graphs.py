#!/usr/bin/env python
"""Build mmap-friendly CPEN incidence caches for PascalVOC-SP (incidence-v2).

By default bakes WIRE spectral coordinates into the primary ``{split}.pt``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.utils.pascal_graph_cache import (
    DEFAULT_WIRE_COORDINATE_DIM,
    PASCAL_SPLITS,
    build_pascal_graph_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--wire-coordinate-dim",
        type=int,
        default=DEFAULT_WIRE_COORDINATE_DIM,
        help="Laplacian eigenvector count baked into the primary cache (0 to disable)",
    )
    parser.add_argument(
        "--wire-combinatorial-laplacian",
        dest="wire_normalized_laplacian",
        action="store_false",
    )
    parser.set_defaults(wire_normalized_laplacian=True)
    args = parser.parse_args()

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    unknown = set(splits) - set(PASCAL_SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    wire_dim = None if args.wire_coordinate_dim <= 0 else args.wire_coordinate_dim
    for split in splits:
        build_pascal_graph_cache(
            data_root=args.data_root,
            split=split,
            rebuild=args.rebuild,
            wire_coordinate_dim=wire_dim,
            wire_normalized_laplacian=args.wire_normalized_laplacian,
        )


if __name__ == "__main__":
    main()
