#!/usr/bin/env python3
"""1-epoch 1M JetClass: BaselineTransformer (M11) vs capen-llama-att (with edges).

Default comparison is L=3, D=256, h=8 after the first completed epoch (epoch 0).
"""
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

_SAVE_DPI = 500
_LINEWIDTH = 0.85
_MARKERSIZE = 2.5

Y_SPECS = (
    ("train_loss", "train loss"),
    ("val_acc", "val accuracy"),
    ("val_roc_auc", "val ROC AUC"),
    ("val_bg_rejection", r"val $1/\varepsilon_B$ ($\varepsilon_S=0.5$)"),
)

_MODEL_STYLE = {
    "att": {"color": "C0", "marker": "o", "label": r"with edges ($M_{11,12,21,22}$)"},
    "m11": {"color": "C1", "marker": "s", "label": r"M11 only (no edges)"},
}


def _setup_style() -> None:
    try:
        import scienceplots  # noqa: F401
    except ImportError:
        return
    plt.style.use(["science", "nature", "no-latex"])


def _load_glob(glob: str, *, model: str, skip_rope: bool = True) -> pd.DataFrame:
    frames = []
    for path in sorted(ROOT.glob(glob)):
        if skip_rope and "rope" in path.stem:
            continue
        df = pd.read_parquet(path)
        ep = df.loc[df["record_type"].eq("epoch")].copy()
        if ep.empty:
            continue
        for _, row in ep.iterrows():
            frames.append(
                {
                    "model": model,
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
                    "run": path.stem,
                }
            )
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)


def load_comparison(*, epoch: int = 0, depth: int = 3, width: int = 256) -> pd.DataFrame:
    att = _load_glob("capen-llama-att_*ntr1M*.parquet", model="att")
    m11 = _load_glob("baseline-transformer_*ntr1M*.parquet", model="m11")
    df = pd.concat([att, m11], ignore_index=True)
    if df.empty:
        raise SystemExit(f"No epoch rows under {ROOT}")
    out = df[
        (df["epoch"] == int(epoch))
        & (df["depth"] == int(depth))
        & (df["width"] == int(width))
    ].copy()
    if out.empty:
        raise SystemExit(
            f"No rows for epoch={epoch} L={depth} D={width}. "
            f"available: {df.groupby(['model','depth','width','epoch']).size().to_dict()}"
        )
    return out.sort_values(["model", "eta0"])


def plot(df: pd.DataFrame, output_path: Path, *, depth: int, width: int, epoch: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6))
    handles: list[Line2D] = []
    for model, style in _MODEL_STYLE.items():
        sub = df[df["model"] == model].sort_values("eta0")
        if sub.empty:
            continue
        handles.append(
            Line2D(
                [],
                [],
                color=style["color"],
                marker=style["marker"],
                markersize=_MARKERSIZE,
                linewidth=_LINEWIDTH,
                label=style["label"],
            )
        )
        for ax, (col, _) in zip(axes.ravel(), Y_SPECS):
            ax.plot(
                sub["eta0"],
                sub[col],
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                markersize=_MARKERSIZE,
                linewidth=_LINEWIDTH,
            )

    for ax, (_, ylabel) in zip(axes.ravel(), Y_SPECS):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\eta_0$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        rf"JetClass 1M, 1 epoch  ($L={depth}$, $D={width}$)",
        fontsize=9,
        y=1.02,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 1.00),
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_SAVE_DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup_style()
    depth, width, epoch = 3, 256, 0
    df = load_comparison(epoch=epoch, depth=depth, width=width)
    csv_path = FIG_DIR / "jetclass_m11_vs_att_1M_ep1.csv"
    fig_path = FIG_DIR / "jetclass_m11_vs_att_1M_ep1.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    plot(df, fig_path, depth=depth, width=width, epoch=epoch)
    print(f"Wrote {len(df)} rows to {csv_path}")
    print(f"Wrote {fig_path}")
    for model, sub in df.groupby("model"):
        best = sub.sort_values("val_roc_auc", ascending=False).iloc[0]
        print(
            f"{model:3s}  n={len(sub)}  η0*={best['eta0']:g}  "
            f"roc={best['val_roc_auc']:.4f}  acc={best['val_acc']:.4f}  "
            f"loss={best['train_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
