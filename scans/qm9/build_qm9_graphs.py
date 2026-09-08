#!/usr/bin/env python
"""Build mmap CPEN incidence caches for QM9 (incidence-v1).

Build ``train`` before ``val`` / ``test``: features are standardized with
train-split statistics, which the train build persists to ``meta.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scans.qm9.qm9_graph_cache import QM9_SPLITS, build_qm9_graph_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--pyg-root",
        default=None,
        help=(
            "Existing torch_geometric QM9 root to read instead of "
            "<data-root>/QM9 (skips the download)."
        ),
    )
    args = parser.parse_args()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    unknown = set(splits) - set(QM9_SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    # Train carries the standardization statistics the other splits reuse.
    splits.sort(key=lambda split: QM9_SPLITS.index(split))
    for split in splits:
        build_qm9_graph_cache(
            data_root=args.data_root,
            split=split,
            rebuild=args.rebuild,
            pyg_root=args.pyg_root,
        )


if __name__ == "__main__":
    main()
