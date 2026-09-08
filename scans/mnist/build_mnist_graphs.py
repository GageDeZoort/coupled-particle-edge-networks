#!/usr/bin/env python
"""Build mmap CPEN incidence caches for MNISTSuperpixels (incidence-v1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scans.mnist.mnist_graph_cache import MNIST_SPLITS, build_mnist_graph_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    unknown = set(splits) - set(MNIST_SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    for split in splits:
        build_mnist_graph_cache(
            data_root=args.data_root,
            split=split,
            rebuild=args.rebuild,
        )


if __name__ == "__main__":
    main()
