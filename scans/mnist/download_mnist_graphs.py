#!/usr/bin/env python
"""Download PyG MNISTSuperpixels (same source as hp-transfer-gts)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    try:
        from torch_geometric.datasets import MNISTSuperpixels
    except ImportError as exc:
        raise ImportError("Downloading MNISTSuperpixels requires torch-geometric") from exc

    root = str(Path(args.data_root) / "MNISTSuperpixels")
    for train in (True, False):
        dataset = MNISTSuperpixels(root=root, train=train)
        split = "train" if train else "test"
        print(f"[mnist-download] {split}: {len(dataset)} graphs at {root}", flush=True)


if __name__ == "__main__":
    main()
