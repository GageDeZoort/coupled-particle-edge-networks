#!/usr/bin/env python
"""
JetClassLite hyperparameter transfer sweeps for CPEN / CAPEN.

Uses ParT-full 17-D particle features and pre-built star-$R$ caches.

Example:
  python scans/jets/test_cpen_jetclass.py \\
    --model capen --operators incidence --star-radius 0.15 \\
    --mode sweep_lr --etas 0.1 --epochs 50 --batch-size 64 --n-gpus 4 \\
    --num-particles 128 \\
    --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \\
    --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.lit_models.lit_cpen_jetclass import LitCPENJetClass
from cpen.utils.jetclass import DEFAULT_NUM_PARTICLES
from scans.common.sweep_common import (
    add_common_args,
    configure_graph_args,
    create_jetclass_datamodule,
    run_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPEN/CAPEN JetClassLite transfer sweeps")
    add_common_args(parser)
    parser.set_defaults(num_particles=DEFAULT_NUM_PARTICLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = "jetclass"
    configure_graph_args(args)
    data_root = args.data_root or args.root

    def datamodule_factory():
        return create_jetclass_datamodule(args, data_root=data_root)

    run_sweep(
        args,
        dataset="jetclass",
        datamodule_factory=datamodule_factory,
        lit_factory=LitCPENJetClass,
    )


if __name__ == "__main__":
    main()
