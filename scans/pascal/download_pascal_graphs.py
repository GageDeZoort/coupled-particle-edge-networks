#!/usr/bin/env python
"""Download PascalVOC-SP raw LRGB splits (requires outbound network)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.utils.pascal_graph_cache import PASCAL_NAME, PASCAL_SPLITS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", default="train,val,test")
    args = parser.parse_args()

    try:
        from torch_geometric.datasets import LRGBDataset
    except ImportError as exc:
        raise ImportError("Downloading PascalVOC-SP requires torch-geometric") from exc

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    unknown = set(splits) - set(PASCAL_SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")

    for split in splits:
        dataset = LRGBDataset(args.data_root, PASCAL_NAME, split=split)
        print(f"[pascal-download] {split}: {len(dataset)} graphs at {args.data_root}", flush=True)


if __name__ == "__main__":
    main()
