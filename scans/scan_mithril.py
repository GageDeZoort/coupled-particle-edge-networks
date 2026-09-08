"""
Optional YAML/config-driven batch scanner (legacy path).

Not required for day-to-day sweeps; use scans/testing/test_*.py instead.
This stub documents the intended interface for future SLURM array jobs.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="YAML-driven CPEN sweep launcher (stub)")
    parser.add_argument("--config", required=True, help="Path to sweep YAML config")
    parser.add_argument("--array-index", type=int, default=0, help="SLURM array task id")
    args = parser.parse_args()
    raise NotImplementedError(
        f"scan_mithril is not implemented yet (config={args.config}, index={args.array_index}). "
        "Use scans/testing/test_*.py for explicit sweeps."
    )


if __name__ == "__main__":
    main()
