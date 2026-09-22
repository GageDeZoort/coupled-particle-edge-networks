#!/usr/bin/env python3
"""η0-transfer look at the 1M JetClass capen-llama-att scan (all epochs)."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jetclass_output/adam/sweep_lr"
)
FIG_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "figures"
GLOB = "capen-llama-att_*ntr1M*.parquet"

_SAVE_DPI = 500
_LINEWIDTH = 0.85
_MARKERSIZE = 2.5

Y_SPECS = (
    ("train_loss", "train loss"),
    ("val_acc", "val accuracy"),
    ("val_roc_auc", "val ROC AUC"),
    ("val_bg_rejection", r"val $1/\varepsilon_B$ ($\varepsilon_S=0.5$)"),
)

_EPOCH_STYLE = {0: "-", 1: "--", 2: ":"}


def _setup_style() -> None:
    try:
        import scienceplots  # noqa: F401
    except ImportError:
        return
    plt.style.use(["science", "nature", "no-latex"])


def load_epochs(*, include_rope: bool = False) -> pd.DataFrame:
    frames = []
    for path in sorted(ROOT.glob(GLOB)):
        is_rope = "rope" in path.stem
        if is_rope and not include_rope:
            continue
        if (not is_rope) and include_rope:
            continue
        df = pd.read_parquet(path)
        ep = df.loc[df["record_type"].eq("epoch")].copy()
        if ep.empty:
            continue
        for _, row in ep.iterrows():
            frames.append(
                {
                    "eta0": float(row["eta_0"]),
                    "depth": int(row["depth"]),
                    "width": int(row["width"]),
                    "heads": int(row["heads"]),
                    "epoch": int(row["epoch"]),
                    "global_step": int(row["global_step"]),
                    "train_loss": float(row["train_loss"]),
                    "val_acc": float(row["val_acc"]),
                    "val_roc_auc": float(row["val_roc_auc"]),
                    "val_bg_rejection": float(row["val_bg_rejection"]),
                    "rope": is_rope,
                    "run": path.stem,
                }
            )
    if not frames:
        raise SystemExit(f"No epoch rows under {ROOT}/{GLOB}")
    return pd.DataFrame(frames).sort_values(["depth", "width", "epoch", "eta0"])


def _arch_label(depth: int, width: int, heads: int) -> str:
    return rf"$L={depth}$, $D={width}$, $h={heads}$"


def plot(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    archs = (
        df[["depth", "width", "heads"]]
        .drop_duplicates()
        .sort_values(["depth", "width"])
        .itertuples(index=False, name=None)
    )
    markers = ("o", "s", "D", "^")
    handles: list[Line2D] = []
    epoch_handles: list[Line2D] = []
    seen_epochs: set[int] = set()
    for i, (depth, width, heads) in enumerate(archs):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                markersize=_MARKERSIZE,
                linewidth=_LINEWIDTH,
                label=_arch_label(depth, width, heads),
            )
        )
        sub_arch = df[
            (df["depth"] == depth) & (df["width"] == width) & (df["heads"] == heads)
        ]
        for epoch, sub in sub_arch.groupby("epoch"):
            epoch = int(epoch)
            sub = sub.sort_values("eta0")
            ls = _EPOCH_STYLE.get(epoch, "-.")
            if epoch not in seen_epochs:
                seen_epochs.add(epoch)
                epoch_handles.append(
                    Line2D(
                        [],
                        [],
                        color="0.35",
                        linestyle=ls,
                        linewidth=_LINEWIDTH,
                        label=rf"epoch {epoch}",
                    )
                )
            for ax, (col, _) in zip(axes.ravel(), Y_SPECS):
                ax.plot(
                    sub["eta0"],
                    sub[col],
                    color=color,
                    marker=marker,
                    linestyle=ls,
                    markersize=_MARKERSIZE,
                    linewidth=_LINEWIDTH,
                )

    for ax, (_, ylabel) in zip(axes.ravel(), Y_SPECS):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\eta_0$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)

    fig.legend(
        handles=handles + epoch_handles,
        loc="upper center",
        ncol=5,
        fontsize=6.5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_SAVE_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup_style()
    df = load_epochs()
    csv_path = FIG_DIR / "jetclass_att_1M_transfer.csv"
    fig_path = FIG_DIR / "jetclass_att_1M_transfer.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    plot(df, fig_path)
    print(f"Wrote {len(df)} rows to {csv_path}")
    print(f"Wrote {fig_path}")
    stars = (
        df.sort_values("val_roc_auc", ascending=False)
        .groupby(["depth", "width", "epoch"], as_index=False)
        .first()
    )
    for _, row in stars.iterrows():
        print(
            f"L={int(row['depth'])} D={int(row['width'])} ep={int(row['epoch'])}  "
            f"η0*={row['eta0']:g}  roc={row['val_roc_auc']:.4f}  "
            f"n={int(((df.depth == row['depth']) & (df.width == row['width']) & (df.epoch == row['epoch'])).sum())}"
        )


if __name__ == "__main__":
    main()
