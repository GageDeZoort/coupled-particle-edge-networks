#!/usr/bin/env python3
"""Stage C head-to-head: mock-only vs fine-tuned checkpoints (mask-holdout).

Scores each RUN_DIR (or mock ckpt via a synthetic run wrapper), writes
holdout recovery tables under notebooks/streams/figures/real_label_holdout_eval/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]  # scripts/ → repo root
sys.path[:0] = [str(REPO), str(REPO / "src")]

from cpen.apps.streams.preprocess.config import DEFAULT_REAL_PIPELINE_ROOT, TEST_REAL_STREAMS
from cpen.apps.streams.real_discovery import (
    DiscoveryConfig,
    apply_threshold,
    choose_threshold_supervised,
    fit_stats_for_cfg,
    holdout_recovery_breakdown,
    load_lit_from_run,
    pick_ckpt,
    rebuild_cell_split,
    score_manifest,
)


def _score_run(
    *,
    name: str,
    run_dir: Path,
    cfg: DiscoveryConfig,
    cell_split,
    stats,
    device: str,
    max_graphs: int | None,
    holdout: set[str],
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    lit, ckpt, hp, device = load_lit_from_run(run_dir, cfg=cfg, device=device)
    print(f"[{name}] ckpt={ckpt} hp={hp}")
    scored = score_manifest(
        cfg,
        lit=lit,
        stats=stats,
        cell_split=cell_split,
        device=device,
        max_graphs=max_graphs,
        progress=True,
    )
    t_star, _ = choose_threshold_supervised(
        scored, holdout_streams=holdout, roles=["fit_val", "val"]
    )
    stars = apply_threshold(scored, t_star)
    rec = holdout_recovery_breakdown(
        stars, holdout_streams=holdout, occupancy=cell_split.occupancy
    )
    rec.insert(0, "model", name)
    print(f"[{name}] t*={t_star:.3f} stars={len(stars):,}")
    display_cols = ["model", "stream_label", "subset", "n_members", "recall", "mean_p"]
    print(rec[display_cols].to_string(index=False))
    return stars, t_star, rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mock-run-dir", type=Path, required=True, help="Mock pretrain run dir (best.ckpt)")
    ap.add_argument("--ft-run-dir", type=Path, default=None, help="Full FT maskholdout run dir")
    ap.add_argument("--lastblock-run-dir", type=Path, default=None, help="Last-block FT run dir")
    ap.add_argument("--max-graphs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    out = REPO / "notebooks" / "streams" / "figures" / "real_label_holdout_eval"
    out.mkdir(parents=True, exist_ok=True)

    cfg = DiscoveryConfig(
        real_root=Path(DEFAULT_REAL_PIPELINE_ROOT),
        galaxy_id="0000",
        drop_features="parallax",
        rich_min_mock=10,
        mask_holdout=True,
    )
    holdout = set(cfg.holdout_streams) | set(TEST_REAL_STREAMS)
    cell_split = rebuild_cell_split(cfg)
    stats, _, _ = fit_stats_for_cfg(cfg)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    runs: list[tuple[str, Path]] = [("mock", args.mock_run_dir)]
    if args.ft_run_dir is not None:
        runs.append(("full_ft", args.ft_run_dir))
    if args.lastblock_run_dir is not None:
        runs.append(("lastblock_ft", args.lastblock_run_dir))

    all_rec = []
    meta = {"device": device, "holdout": sorted(holdout), "runs": {}}
    for name, run_dir in runs:
        assert pick_ckpt(run_dir).is_file(), run_dir
        stars, t_star, rec = _score_run(
            name=name,
            run_dir=run_dir,
            cfg=cfg,
            cell_split=cell_split,
            stats=stats,
            device=device,
            max_graphs=args.max_graphs,
            holdout=holdout,
        )
        all_rec.append(rec)
        stars.to_parquet(out / f"stars_{name}_t{t_star:.2f}.parquet", index=False)
        meta["runs"][name] = {"run_dir": str(run_dir), "t_star": t_star}

    tab = pd.concat(all_rec, ignore_index=True)
    tab.to_parquet(out / "holdout_recovery_compare.parquet", index=False)
    (out / "compare_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Bar chart: all-subset recall by model × stream
    sub = tab[tab.subset == "all"].copy()
    streams = sorted(sub.stream_label.unique())
    models = list(dict.fromkeys(sub.model))
    x = np.arange(len(streams))
    w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, m in enumerate(models):
        vals = [
            float(sub[(sub.model == m) & (sub.stream_label == s)].recall.iloc[0])
            if len(sub[(sub.model == m) & (sub.stream_label == s)])
            else 0.0
            for s in streams
        ]
        ax.bar(x + (i - 0.5 * (len(models) - 1)) * w, vals, width=w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(streams, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall @ t*")
    ax.set_title("Holdout recovery: mock vs fine-tunes")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "holdout_recall_compare.png", dpi=140)
    fig.savefig(out / "holdout_recall_compare.pdf")
    print("wrote", out)


if __name__ == "__main__":
    main()
