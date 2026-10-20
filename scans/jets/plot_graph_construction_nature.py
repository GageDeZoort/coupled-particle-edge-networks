#!/usr/bin/env python3
"""Nature-ready three-panel graph-construction figure (iteratively updated).

Panels
  (a) JetClass val ROC vs kNN degree k  at fixed R★=0.1 (noloop CompletenessP v2)
  (b) JetClass val ROC vs star radius R★ at fixed k★=7
  (c) edge ablation: η₀ transfer; color=P, dashed=no edges, solid=edges

Canonical graph: all-seed star-R + ΔR-kNN with kin7 star aggregates (sk7).

Live collection from scratch run dirs + optional cached CSVs. Re-run anytime::

  python scans/jets/plot_graph_construction_nature.py
  python scans/jets/plot_graph_construction_nature.py --no-collect   # CSV only

Outputs
  paper/figures/fig_graph_construction.{png,pdf,eps}
  paper/figures/fig_graph_construction_caption.txt
  paper/data/graph_k_val_from_trainlog.csv
  paper/data/graph_R_val_from_trainlog.csv   (when R cells exist)
  paper/data/m11_eta_transfer.csv   # panel (c) M11 live; edge arm pending
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from paper.style.nature_plots import nature_color, save_paper_figure, use_nature

SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
DATA = REPO / "paper" / "data"
FIG_DIR = REPO / "paper" / "figures"
FIG_STEM = "fig_graph_construction"

K_TAG = "v6gt"  # gpu-test B=96×4, B_global=384 (supersedes v5a80)
K_TAG_LAM = "v6lam"  # Lambda 8×A100 B=384×1 (R★ variants, e.g. 0.075)
K_TAG_V1 = "v1lam"  # Lambda 8×A100 B=64×8=512 @ R★=0.1 (refined panel a)
K_CSV = DATA / "graph_k_val_from_trainlog.csv"
K_CSV_R075 = DATA / "graph_k_R0p075_val_from_trainlog.csv"
K_STAR_DEFAULT = 0.1  # panel (a) fixed R★
# Downstream / panel (c) operating point from noloop CompletenessP v2 (a)+(b):
OP_K_STAR = 7
OP_R_STAR = 0.1
R_TAG = "v6a80"  # Plot-2: 1×A100-80GB B=384 (legacy; prefer v6gt @ k★=12)
R_TAG_GT = "v6gt"  # Plot-2 matched: 4×A100 gpu-test B=96→384
R_TAG_GT2 = "v6gt2"  # re-run after streaming/skip fix (true disjoint shards)
R_TAG_MIG = "v6mig"  # deprecated probe tag (1×MIG B=32)
R_TAG_V1 = "v1lam"  # Lambda R★ @ k★=6, B=64×8
R_CSV = DATA / "graph_R_val_from_trainlog.csv"
M11_TAG = "m11-b96-g4-s2500-dh32"  # panel (c) redo @ 2500 after encoder-scaling fix
M11_TAG_LEGACY = "m11-b96-g4-s4000-dh32"  # archived k★=12 campaign
M11_TAG_A80 = "m11-b384-g1-s4000-dh32-a80"  # D=768 extension on 1×A100-80GB
M11_CSV = DATA / "m11_eta_transfer.csv"
# Edge η₀ @ locked OP (k★, R★)=(7, 0.1), L4/D256, B=64×4, 40k steps (gputest).
EDGE_TAG = "edge-b64-g4-s40000-dh32-k7-R0p1"
EDGE_TAG_LEGACY = "edge-b64-g4-s20000-dh32-k7-R0p1"  # first 20k pass before extend
EDGE_CSV = DATA / "edge_eta_transfer.csv"
EDGE_TARGET = 40000

K_GRID = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # noloop CompletenessP v2 panel (a)
R_GRID = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.2]  # noloop panel (b) at k★=7
K_STAR_R = OP_K_STAR  # fixed k for panel (b) / Plot 2
ETA_GRID = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
WIDTHS = [256, 384, 512, 768]
# Parameter counts for panel (c) color.
#
# Edge CAPEN (`capen-llama-att`): raw trainable P is valid — M11/M12/M21/M22 +
# edge MLP/encoder/pool are all used.
#
# M11-only (`baseline-transformer`): the module still *allocates* the full CAPEN
# weight set (same raw P as edge), but `_forward_m11_only` only uses particle
# M11 + node MLP/pool. Edge pathway weights are kept in the graph via `* 0`
# (`_unused_edge_modules`, `output_pool._unused_edge_pool`, zeroed decoders)
# so DDP sees them — they must NOT count toward panel-(c) P. Count by
# subtracting unused name prefixes from a real M11 `last.ckpt` (see
# `fig_graph_construction.md`). Never trust parquet `n_trainable_params` for M11.
P_EDGE_BY_WIDTH: dict[int, int] = {
    # 128: 2_759_040,  # wiped 2026-10-01 (suspicious vs D=256); re-queue later
    256: 11_023_104,
    384: 24_792_192,
    512: 44_066_304,
    768: 99_129_600,  # raw parquet; confirm if/when D=768 edge runs
}
P_M11_EFFECTIVE_BY_WIDTH: dict[int, int] = {
    # From baseline-transformer last.ckpt: total − {encoder_e, attn_12/21/22,
    # mlp_e_*, output_pool.attn_e, decoder_x, decoder_e, attn_gamma_12/21/22}.
    256: 4_463_361,
    384: 10_037_377,
    512: 17_839_617,
}
# Back-compat alias (prefer kind-specific maps above).
P_BY_WIDTH = P_EDGE_BY_WIDTH
M11_TARGET = 2500


def _fmt_params(n: int) -> str:
    """Format trainable-param counts for legends / colorbar ticks."""
    if n >= 1_000_000:
        m = n / 1_000_000
        return f"{m:.1f}M" if m < 100 else f"{m:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)

# Panel-(a) style k-sweeps: capture R★ + tag so R=0.075 (v6lam) coexists with R=0.125 (v6gt).
NAME_K = re.compile(
    r"knn(?P<k>\d+).*graph-k-.*R(?P<R>[0-9p]+).*shard(?P<shard>\d+).*seed(?P<seed>\d+).*(?P<tag>v1lam|v6gt2|v6gt|v6lam)"
)
# Plot-2 dirs: ...graph-R-...-R{tag}-shard{S}-seed{E}-v6a80|v6gt|v6mig|v1lam
NAME_R = re.compile(
    r"knn(?P<k>\d+).*graph-R-.*R(?P<R>[0-9p]+)-shard(?P<shard>\d+)-seed(?P<seed>\d+)-(?P<tag>v1lam|v6a80|v6gt2|v6gt|v6mig|v6lam)"
)
NAME_ETA = re.compile(
    r"^(?P<model>baseline-transformer|capen-llama-att)_"
    r"(?P<L>\d+)_(?P<D>\d+)_(?P<eta>[0-9p]+)_"
)
SEED_IN_NAME = re.compile(r"(?:^|[^0-9])seed(?P<seed>\d+)(?:-|$)")
SHARD_IN_NAME = re.compile(r"(?:^|[^0-9])shard(?P<shard>\d+)(?:-|$)")
SKIP_IN_NAME = re.compile(r"(?:^|[^0-9])skip(?P<skip>[0-9pM]+)(?:-|$)")


def _seed_from_run(name: str) -> int:
    matches = list(SEED_IN_NAME.finditer(name))
    if not matches:
        return 0
    return int(matches[-1].group("seed"))


def _shard_from_run(name: str, *, n_train: int = 1_000_000) -> int:
    """Shard index from explicit ``shardN`` tag, else from ``skip`` / n_train."""
    matches = list(SHARD_IN_NAME.finditer(name))
    if matches:
        return int(matches[-1].group("shard"))
    sm = SKIP_IN_NAME.search(name)
    if sm is None:
        return 0
    raw = sm.group("skip").replace("p", ".")
    if raw.endswith("M"):
        skip = int(float(raw[:-1]) * 1_000_000)
    elif raw.endswith("k"):
        skip = int(float(raw[:-1]) * 1_000)
    else:
        skip = int(float(raw))
    if n_train <= 0 or skip <= 0:
        return 0
    return int(skip // n_train)


def _eta_tag(s: str) -> float:
    return float(s.replace("p", "."))


def _r_tag(s: str) -> float:
    return float(s.replace("p", "."))


def _roc_at_step(pqpath: Path, min_step: int) -> tuple[float, int] | None:
    if not pqpath.is_file():
        return None
    try:
        import pyarrow.parquet as pq

        df = pq.read_table(pqpath, columns=["global_step", "val_roc_auc"]).to_pandas()
    except Exception:
        return None
    sub = df.dropna(subset=["val_roc_auc"])
    if sub.empty:
        return None
    near = sub[sub["global_step"] >= min_step]
    row = (
        near.loc[near["global_step"].idxmin()]
        if len(near)
        else sub.loc[sub["global_step"].idxmax()]
    )
    step = int(row["global_step"])
    if step < min_step:
        return None
    return float(row["val_roc_auc"]), step


def collect_k(
    min_step: int = 1000,
    *,
    star_r: float | None = K_STAR_DEFAULT,
    tags: tuple[str, ...] = (K_TAG_V1, K_TAG, K_TAG_LAM),
) -> pd.DataFrame:
    """Collect kNN-k sweep cells. Default prefers Lambda v1lam @ R★=0.1."""
    rows = []
    seen: set[Path] = set()
    for tag in tags:
        for d in sorted(SWEEP.glob(f"*{tag}*")):
            if not d.is_dir() or d in seen:
                continue
            seen.add(d)
            m = NAME_K.search(d.name)
            if not m:
                continue
            r_val = _r_tag(m["R"])
            if star_r is not None and abs(r_val - float(star_r)) > 1e-9:
                continue
            hit = _roc_at_step(Path(str(d) + ".parquet"), min_step)
            if hit is None:
                continue
            roc, step = hit
            rows.append(
                dict(
                    k=int(m["k"]),
                    star_r=r_val,
                    shard=int(m["shard"]),
                    seed=int(m["seed"]),
                    step=step,
                    val_roc_auc=roc,
                    tag=m["tag"],
                    run=d.name,
                )
            )
    df = pd.DataFrame(rows)
    if len(df):
        # Prefer streaming-fixed v6gt2 over collapsed v6gt copies.
        pref = {"v1lam": 4, "v6gt2": 3, "v6gt": 2, "v6lam": 1}
        df["_pref"] = df["tag"].map(lambda t: pref.get(t, 0))
        df = df.sort_values(["k", "star_r", "shard", "seed", "_pref"]).drop_duplicates(
            subset=["k", "star_r", "shard", "seed"], keep="last"
        )
        df = df.drop(columns=["_pref"])
        # Drop byte-identical shard clones (ram-subset skip bug).
        df = df.sort_values(["k", "star_r", "seed", "shard"]).drop_duplicates(
            subset=["k", "star_r", "seed", "val_roc_auc"], keep="first"
        )
        # Panel-(a) grid: exclude k=1 (degenerate 1-NN; not a meaningful graph).
        df = df[df["k"].isin(K_GRID)]
    return df.reset_index(drop=True)


def collect_R(min_step: int = 1000, k_star: int | None = None) -> pd.DataFrame:
    """Collect Plot-2 R★ sweeps (v6gt / v6lam / v6a80; v6mig probe ignored last)."""
    rows = []
    seen: set[Path] = set()
    for pat in (
        f"*graph-R*{R_TAG_V1}*",
        f"*graph-R*{R_TAG_GT2}*",
        f"*graph-R*{R_TAG_GT}*",
        f"*graph-R*v6lam*",
        f"*graph-R*{R_TAG}*",
        f"*graph-R*{R_TAG_MIG}*",
    ):
        for d in sorted(SWEEP.glob(pat)):
            if not d.is_dir() or d in seen:
                continue
            seen.add(d)
            m = NAME_R.search(d.name)
            if not m:
                continue
            k = int(m.group("k"))
            r_raw = m.group("R")
            shard = int(m.group("shard"))
            seed = int(m.group("seed"))
            if k_star is not None and k != k_star:
                continue
            hit = _roc_at_step(Path(str(d) + ".parquet"), min_step)
            if hit is None:
                continue
            roc, step = hit
            rows.append(
                dict(
                    k=k,
                    star_r=_r_tag(r_raw),
                    shard=shard,
                    seed=seed,
                    step=step,
                    val_roc_auc=roc,
                    tag=m.group("tag"),
                    run=d.name,
                )
            )
    df = pd.DataFrame(rows)
    if len(df):
        # Prefer streaming-fixed v6gt2 over collapsed v6gt copies.
        pref = {"v1lam": 5, "v6gt2": 4, "v6gt": 3, "v6lam": 2, "v6a80": 1, "v6mig": 0}
        df["_pref"] = df["tag"].map(pref).fillna(0)
        df = df.sort_values(["star_r", "shard", "seed", "_pref"]).drop_duplicates(
            subset=["k", "star_r", "shard", "seed"], keep="last"
        ).drop(columns=["_pref"])
        # Drop byte-identical shard clones (same seed + identical ROC → ram-subset skip bug).
        df = df.sort_values(["star_r", "seed", "shard"]).drop_duplicates(
            subset=["k", "star_r", "seed", "val_roc_auc"], keep="first"
        )
    return df.reset_index(drop=True)


def collect_eta(tag: str, target: int) -> pd.DataFrame:
    rows = []
    for p in sorted(SWEEP.glob(f"*{tag}*")):
        name = p.name.replace(".parquet", "")
        m = NAME_ETA.match(name)
        if m is None or tag not in name:
            continue
        if p.suffix == ".parquet":
            continue
        pqpath = Path(str(p) + ".parquet")
        if not pqpath.is_file():
            continue
        try:
            import pyarrow.parquet as pq

            df = pq.read_table(
                pqpath,
                columns=["global_step", "val_roc_auc", "val_acc", "val_loss"],
            ).to_pandas()
        except Exception:
            continue
        if df.empty:
            continue
        step = int(df["global_step"].max())
        sub = df.dropna(subset=["val_roc_auc"])
        if sub.empty:
            continue
        last = sub.loc[sub["global_step"].idxmax()]
        model = m.group("model")
        kind = "m11" if model == "baseline-transformer" else "edge"
        rows.append(
            dict(
                kind=kind,
                model=model,
                width=int(m.group("D")),
                eta0=_eta_tag(m.group("eta")),
                seed=_seed_from_run(name),
                shard=_shard_from_run(name),
                step=step,
                done=step >= target,
                val_roc_auc=float(last["val_roc_auc"]),
                val_acc=float(last["val_acc"]) if pd.notna(last.get("val_acc")) else None,
                val_loss=(
                    float(last["val_loss"]) if pd.notna(last.get("val_loss")) else None
                ),
                run=name,
            )
        )
    out = pd.DataFrame(rows)
    if len(out):
        # Keep seed×shard replicates (error bars); do not collapse shards.
        out = out.sort_values(
            ["kind", "width", "eta0", "seed", "shard"]
        ).drop_duplicates(
            subset=["kind", "width", "eta0", "seed", "shard"], keep="last"
        )
    return out.reset_index(drop=True)


GEOM_K_CSV = DATA / "graph_geometry_knn.csv"
GEOM_S_CSV = DATA / "graph_geometry_star.csv"
GEOM_COLOR = "0.28"


def _style_twin(ax_r, ylabel: str) -> None:
    """Muted right-hand axis; label sits inside — no tick labels (avoids panel clash)."""
    ax_r.set_ylabel("")
    ax_r.set_yticklabels([])
    ax_r.tick_params(axis="y", colors=GEOM_COLOR, length=2, labelleft=False, labelright=False)
    ax_r.spines["right"].set_color(GEOM_COLOR)
    ax_r.spines["right"].set_linewidth(0.6)
    ax_r.text(
        0.98, 0.98, ylabel,
        transform=ax_r.transAxes, ha="right", va="top",
        color=GEOM_COLOR, fontsize=5.5,
    )


def _geom_errorbar(ax_in, x, y, yerr) -> None:
    """Inset mean±std: numpy arrays + bold whiskers (thin grey vanishes at figure scale)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    err = None if yerr is None else np.asarray(yerr, dtype=np.float64)
    ax_in.errorbar(
        x,
        y,
        yerr=err,
        fmt="o-",
        color=GEOM_COLOR,
        lw=1.0,
        ms=3.0,
        capsize=2.5,
        capthick=1.0,
        elinewidth=1.1,
        zorder=3,
        clip_on=False,
    )


def inset_edges_vs_k(ax) -> None:
    """Small edges/jet subplot in the bottom-right of panel (a)."""
    if not GEOM_K_CSV.is_file():
        return
    g = pd.read_csv(GEOM_K_CSV).sort_values("k")
    if "n_edges_mean" not in g.columns or g.empty:
        return

    # Axes-fraction rect: [left, bottom, width, height] — compact lower-right.
    ax_in = ax.inset_axes([0.55, 0.12, 0.42, 0.36])
    yerr = g["n_edges_std"] if "n_edges_std" in g.columns else None
    _geom_errorbar(ax_in, g["k"], g["n_edges_mean"], yerr)
    ax_in.set_xlabel(r"$k$", fontsize=5.5, labelpad=0.5)
    ax_in.set_ylabel(r"edges / jet", fontsize=5.5, labelpad=0.5, color=GEOM_COLOR)
    ax_in.set_xticks([2, 8, 16])
    ax_in.tick_params(axis="both", labelsize=5, length=2, pad=0.5, colors=GEOM_COLOR)
    ax_in.tick_params(axis="x", colors="0.25")
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("0.5")
    ax_in.set_facecolor("1.0")
    y0 = 0.0
    y1 = float(g["n_edges_mean"].max())
    if yerr is not None:
        y1 = float((g["n_edges_mean"] + g["n_edges_std"].fillna(0.0)).max())
    ax_in.set_ylim(y0, y1 * 1.08)
    ax_in.set_xlim(1.5, 16.5)


def inset_support_vs_R(ax) -> None:
    """Small |S|/jet subplot in the bottom-right of panel (b)."""
    if not GEOM_S_CSV.is_file():
        return
    g = pd.read_csv(GEOM_S_CSV).sort_values("star_r")
    if "mean_support" not in g.columns or g.empty:
        return
    # Prefer the Plot-2 R★ grid; keep nearby study points if grid rows missing.
    g_plot = g[g["star_r"].isin(R_GRID)].copy()
    if g_plot.empty:
        g_plot = g
    ax_in = ax.inset_axes([0.55, 0.12, 0.42, 0.36])
    yerr = g_plot["support_std"] if "support_std" in g_plot.columns else None
    _geom_errorbar(ax_in, g_plot["star_r"], g_plot["mean_support"], yerr)
    ax_in.set_xlabel(r"$R_\star$", fontsize=5.5, labelpad=0.5)
    ax_in.set_ylabel(r"$|S|$ / jet", fontsize=5.5, labelpad=0.5, color=GEOM_COLOR)
    ax_in.set_xticks([0.05, 0.1, 0.2])
    ax_in.tick_params(axis="both", labelsize=5, length=2, pad=0.5, colors=GEOM_COLOR)
    ax_in.tick_params(axis="x", colors="0.25")
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("0.5")
    ax_in.set_facecolor("1.0")
    y1 = float(g_plot["mean_support"].max())
    if yerr is not None:
        y1 = float((g_plot["mean_support"] + g_plot["support_std"].fillna(0.0)).max())
    ax_in.set_ylim(0.0, y1 * 1.12)
    ax_in.set_xlim(0.03, 0.22)


def agg_mean_std(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = (
        df.groupby(by, as_index=False)
        .agg(n=("val_roc_auc", "count"), mean=("val_roc_auc", "mean"), std=("val_roc_auc", "std"))
    )
    g["std"] = g["std"].fillna(0.0)
    return g


def panel_a(ax, k_df: pd.DataFrame) -> None:
    ax.set_title(r"a  $k$ at $R_\star{=}0.1$", loc="left", pad=3)
    ax.set_xlabel(r"kNN degree $k$")
    ax.set_ylabel("Val. ROC-AUC")
    ax.set_xticks(K_GRID)
    ax.set_xlim(1.5, 10.5)

    # Edges/jet as an inset (own axes) so ROC error bars stay readable.
    inset_edges_vs_k(ax)

    # Fixed window for ROC; inset occupies lower half.
    ax.set_ylim(0.9335, 0.9425)

    if k_df.empty:
        ax.text(0.5, 0.72, "awaiting Plot-1 cells", transform=ax.transAxes,
                ha="center", va="center", color="0.5", fontsize=7)
        return

    agg = agg_mean_std(k_df, "k")
    c = nature_color(0)
    ax.errorbar(
        agg["k"], agg["mean"], yerr=agg["std"],
        fmt="o-", color=c, ms=3.2, lw=0.9, capsize=2.0, elinewidth=0.8,
        zorder=3,
    )


def panel_b(ax, r_df: pd.DataFrame, k_star: int | None) -> None:
    title = r"b  $R_\star$"
    if k_star is not None:
        title += rf" at $k^\star={k_star}$"
    else:
        title += r" at $k^\star$"
    ax.set_title(title, loc="left", pad=3)
    ax.set_xlabel(r"star radius $R_\star$")
    ax.set_ylabel("Val. ROC-AUC")
    ax.set_xticks(R_GRID)
    ax.set_xticklabels([f"{r:g}" for r in R_GRID], fontsize=5.5)
    ax.set_xlim(0.02, 0.22)

    inset_support_vs_R(ax)

    if r_df.empty:
        ax.text(
            0.5, 0.58,
            "Plot 2 pending",
            transform=ax.transAxes, ha="center", va="center",
            color="0.5", fontsize=7,
        )
        ax.set_ylim(0.915, 0.945)
        return

    agg = agg_mean_std(r_df, "star_r")
    c = nature_color(0)
    # Match panel (a) ROC window so (a)/(b) are visually comparable.
    ax.set_ylim(0.9335, 0.9425)
    ax.errorbar(
        agg["star_r"], agg["mean"], yerr=agg["std"],
        fmt="s-", color=c, ms=3.2, lw=0.9, capsize=2.5, elinewidth=1.0,
        capthick=1.0, zorder=3,
    )


def panel_c(
    ax,
    eta_df: pd.DataFrame,
    *,
    fig=None,
    metric: str = "val_roc_auc",
    ylabel: str | None = None,
    title: str = r"c  edge ablation",
    legend: bool = True,
    colorbar: bool = True,
    y_pad: float | None = None,
) -> None:
    """Edge ablation: η₀ transfer, color = trainable P, ls = edges vs none."""
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    if ylabel is None:
        ylabel = {
            "val_roc_auc": "Val. ROC-AUC",
            "val_acc": "Val. accuracy",
            "val_loss": "Val. loss",
        }.get(metric, metric)
    if y_pad is None:
        y_pad = 0.02 if metric == "val_loss" else 0.003

    ax.set_title(title, loc="left", pad=3)
    ax.set_xlabel(r"base learning rate $\eta_0$")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_xticks(ETA_GRID)
    ax.set_xticklabels([str(e) for e in ETA_GRID])

    if eta_df.empty or metric not in eta_df.columns:
        ax.text(
            0.5,
            0.5,
            "M11 filling\n(edge pending)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="0.5",
            fontsize=7,
        )
        return

    # Color encodes *effective* parameter count; linestyle encodes edge presence.
    # Dashed = M11-only (effective P); solid = edges (raw trainable P).
    kind_style = {
        "m11": dict(ls="--", zorder=2),
        "edge": dict(ls="-", zorder=3),
    }
    def _p_for(kind: str, w: int) -> int:
        if kind == "m11":
            return P_M11_EFFECTIVE_BY_WIDTH[w]
        return P_EDGE_BY_WIDTH[w]

    # Stable color scale over both arms' P values (incl. pending D=768 edge raw).
    all_widths = [w for w in WIDTHS if w in P_EDGE_BY_WIDTH]
    p_all = np.array(
        [P_EDGE_BY_WIDTH[w] for w in all_widths]
        + [P_M11_EFFECTIVE_BY_WIDTH[w] for w in all_widths if w in P_M11_EFFECTIVE_BY_WIDTH],
        dtype=float,
    )
    widths_present = sorted(
        {int(w) for w in eta_df["width"].unique() if int(w) in P_EDGE_BY_WIDTH}
    )
    # Log-normalize so ~4.5M→99M spreads evenly on viridis.
    norm = mpl.colors.LogNorm(vmin=float(p_all.min()), vmax=float(p_all.max()))
    cmap = mpl.colormaps["viridis"]

    plotted_kinds: set[str] = set()
    y_vals: list[float] = []
    for kind, ksty in kind_style.items():
        subk = eta_df[eta_df["kind"] == kind]
        if subk.empty:
            continue
        plotted_kinds.add(kind)
        for w in widths_present:
            if kind == "m11" and w not in P_M11_EFFECTIVE_BY_WIDTH:
                continue
            sub = subk[subk["width"] == w].dropna(subset=[metric])
            if sub.empty:
                continue
            color = cmap(norm(_p_for(kind, w)))
            # Aggregate finished seed×shard replicates; n=1 → std=0 (no whisker).
            # Pending cells are shown hollow so in-flight shards don't hide finished means.
            finished = sub[sub["done"]] if "done" in sub.columns else sub
            pending = (
                sub[~sub["done"]]
                if "done" in sub.columns
                else sub.iloc[0:0]
            )
            if len(finished):
                agg = (
                    finished.groupby("eta0", as_index=False)
                    .agg(
                        n=(metric, "count"),
                        mean=(metric, "mean"),
                        std=(metric, "std"),
                    )
                    .sort_values("eta0")
                )
                agg["std"] = agg["std"].fillna(0.0)
                yerr = agg["std"].to_numpy()
                yerr = np.where(agg["n"].to_numpy() >= 2, yerr, 0.0)
                ax.errorbar(
                    agg["eta0"], agg["mean"], yerr=yerr,
                    linestyle=ksty["ls"], color=color,
                    marker="o", ms=3.0, lw=0.95,
                    capsize=2.0, elinewidth=0.8, zorder=ksty["zorder"],
                )
                y_vals.extend(agg["mean"].astype(float).tolist())
            if len(pending):
                pend = (
                    pending.groupby("eta0", as_index=False)
                    .agg(mean=(metric, "mean"))
                    .sort_values("eta0")
                )
                ax.plot(
                    pend["eta0"], pend["mean"],
                    linestyle="none", color=color,
                    marker="o", ms=3.2, mfc="white", mew=0.8,
                    zorder=ksty["zorder"],
                )
                y_vals.extend(pend["mean"].astype(float).tolist())

    # Linestyle legend only (P lives on the colorbar).
    handles = [
        Line2D([0], [0], color="0.25", ls="--", lw=1.0, marker="o", ms=3.0,
               label="No edges"),
        Line2D([0], [0], color="0.25", ls="-", lw=1.0, marker="o", ms=3.0,
               label="Edges"),
    ]
    if "edge" not in plotted_kinds:
        # Keep the Edges entry so the encoding is clear before data arrives.
        pass
    if legend:
        loc = "upper left" if metric == "val_loss" else "lower left"
        ax.legend(handles=handles, loc=loc, fontsize=5.5, ncol=1,
                  handlelength=2.2)

    # Colorbar: parameter count (kind-dependent; see P_* maps).
    if fig is not None and colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("# parameters", fontsize=6)
        cbar.ax.tick_params(labelsize=5, length=2, which="both")
        tick_ps = np.array([5_000_000, 10_000_000, 25_000_000, 50_000_000], dtype=float)
        cbar.set_ticks(tick_ps)
        cbar.set_ticklabels(["5M", "10M", "25M", "50M"])
        cbar.minorticks_off()
        cbar.ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    if y_vals:
        y0 = float(min(y_vals) - y_pad)
        y1 = float(max(y_vals) + y_pad)
        ax.set_ylim(y0, y1)


def write_caption(path: Path, k_df: pd.DataFrame, r_df: pd.DataFrame, eta_df: pd.DataFrame) -> None:
    k_complete = (
        len(k_df) > 0
        and set(int(k) for k in k_df["k"].unique()) >= set(K_GRID)
        and bool((k_df.groupby("k").size() == 6).all())
    )
    r_n = int(len(r_df)) if len(r_df) else 0
    m11_done = int(((eta_df["kind"] == "m11") & eta_df["done"]).sum()) if len(eta_df) else 0
    m11_tot = int((eta_df["kind"] == "m11").sum()) if len(eta_df) else 0
    edge_done = int(((eta_df["kind"] == "edge") & eta_df["done"]).sum()) if len(eta_df) else 0
    edge_tot = int((eta_df["kind"] == "edge").sum()) if len(eta_df) else 0

    text = f"""\
Figure. Graph construction and η₀-transfer on JetClass.
Canonical graph: all-seed star-R hyperedges with kin7 aggregate features plus
ΔR-kNN (noloop) part-interaction edges (tag sk7 / v2cpNL).
(a) Holdout validation ROC–AUC after 2500 Adam steps versus kNN degree k at fixed
star radius R★ = 0.1 (CompletenessP v2: B_global=128, D=256, L=4). Each point is
the mean over n = 6 independent training cells (three disjoint 320k-jet shards ×
two seeds). Error bars denote the sample standard deviation (s.d.) across those
cells, not the s.e.m. k ∈ {{2,…,10}}. Peak at k★ = 7 (mean ROC ≈ 0.9391).
Inset: mean directed ΔR-kNN edges per jet.
{"Panel (a) complete (all k with n = 6)." if k_complete else "Panel (a) still filling."}
(b) Same protocol versus star radius R★ at k★ = 7. ROC cells: {r_n}/42.
Performance is flat for R ∈ [0.08, 0.14] (≈ 0.9391); operating point R★ = 0.1.
(c) Edge ablation η₀×D at (k★, R★)=({OP_K_STAR}, {OP_R_STAR}): pending re-run at
the frozen OP (prior cells used k★=6). Dashed = M11; solid = edges.
M11: {m11_done}/{max(m11_tot, 18)}; edges: {edge_done}/{max(edge_tot, 18)}.
"""
    path.write_text(text)
    print(f"wrote {path}")


def _load_or_collect(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.no_collect:
        k_df = pd.read_csv(K_CSV) if K_CSV.is_file() else pd.DataFrame()
        r_df = pd.read_csv(R_CSV) if R_CSV.is_file() else pd.DataFrame()
        frames = []
        if M11_CSV.is_file():
            frames.append(pd.read_csv(M11_CSV))
        if EDGE_CSV.is_file():
            frames.append(pd.read_csv(EDGE_CSV))
        eta_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return k_df, r_df, eta_df

    k_df = collect_k(args.k_min_step, star_r=K_STAR_DEFAULT)
    k_r075 = collect_k(args.k_min_step, star_r=0.075)
    r_df = collect_R(args.k_min_step, k_star=K_STAR_R)
    # Panel (c): M11 @2500 (legacy) + edge @20k at (k★,R★)=(7,0.1).
    m11_df = collect_eta(M11_TAG, args.m11_target)
    edge_target = int(getattr(args, "edge_target", EDGE_TARGET) or EDGE_TARGET)
    edge_df = collect_eta(EDGE_TAG, edge_target)
    frames = [d for d in (m11_df, edge_df) if len(d)]
    eta_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(k_df):
        k_df.to_csv(K_CSV, index=False)
        print(f"wrote {K_CSV} ({len(k_df)} rows)")
    if len(k_r075):
        k_r075.to_csv(K_CSV_R075, index=False)
        print(f"wrote {K_CSV_R075} ({len(k_r075)} rows)")
    if len(r_df):
        r_df.to_csv(R_CSV, index=False)
        print(f"wrote {R_CSV} ({len(r_df)} rows)")
    if len(m11_df):
        m11_df.to_csv(M11_CSV, index=False)
        print(f"wrote {M11_CSV} ({len(m11_df)} rows)")
    if len(edge_df):
        edge_df.to_csv(EDGE_CSV, index=False)
        print(f"wrote {EDGE_CSV} ({len(edge_df)} rows)")
    if not len(eta_df):
        print(f"panel (c): no {M11_TAG} / {EDGE_TAG} cells yet")
    return k_df, r_df, eta_df


def plot_panel_c_metrics(eta_df: pd.DataFrame, stem: str) -> None:
    """Diagnostic: same panel-(c) encoding for ROC / accuracy / loss."""
    use_nature(fontsize=8)
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.15),
        gridspec_kw={"wspace": 0.45},
    )
    specs = [
        ("val_roc_auc", "Val. ROC-AUC", r"a  val ROC"),
        ("val_acc", "Val. accuracy", r"b  val accuracy"),
        ("val_loss", "Val. loss", r"c  val loss"),
    ]
    for i, (metric, ylabel, title) in enumerate(specs):
        panel_c(
            axes[i],
            eta_df,
            fig=fig if i == 2 else None,
            metric=metric,
            ylabel=ylabel,
            title=title,
            legend=(i == 0),
            colorbar=(i == 2),
        )
    written = save_paper_figure(fig, stem, formats=("png", "pdf"))
    for p in written:
        print(f"wrote {p}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-collect", action="store_true",
                    help="Skip live scratch scan; read CSVs only")
    ap.add_argument("--k-min-step", type=int, default=1000)
    ap.add_argument("--m11-target", type=int, default=M11_TARGET)
    ap.add_argument("--edge-target", type=int, default=EDGE_TARGET)
    ap.add_argument("--stem", default=FIG_STEM)
    ap.add_argument(
        "--panel-c-metrics",
        action="store_true",
        help="Also write diagnostic fig with panel-(c) val_roc / val_acc / val_loss",
    )
    ap.add_argument(
        "--panel-c-metrics-only",
        action="store_true",
        help="Only write the panel-(c) ROC/acc/loss diagnostic (skip full 3-panel fig)",
    )
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    k_df, r_df, eta_df = _load_or_collect(args)

    if len(k_df) and "k" in k_df.columns:
        k_df = k_df[k_df["k"].isin(K_GRID)].reset_index(drop=True)

    metrics_stem = "fig_graph_construction_panel_c_metrics"
    if args.panel_c_metrics or args.panel_c_metrics_only:
        plot_panel_c_metrics(eta_df, metrics_stem)
        # Peak η₀ summary for each metric (finished cells only).
        if len(eta_df):
            done = eta_df[eta_df["done"]] if "done" in eta_df.columns else eta_df
            for metric in ("val_roc_auc", "val_acc", "val_loss"):
                if metric not in done.columns:
                    continue
                print(f"\n=== peak η₀ by width ({metric}, finished) ===")
                for kind in ("m11", "edge"):
                    sub = done[done["kind"] == kind].dropna(subset=[metric])
                    if sub.empty:
                        continue
                    agg = (
                        sub.groupby(["width", "eta0"], as_index=False)[metric]
                        .mean()
                    )
                    for w, g in agg.groupby("width"):
                        if metric == "val_loss":
                            row = g.loc[g[metric].idxmin()]
                        else:
                            row = g.loc[g[metric].idxmax()]
                        print(
                            f"  {kind} D={int(w)}: η₀={row['eta0']:g} "
                            f"→ {metric}={row[metric]:.5f}"
                        )

    if args.panel_c_metrics_only:
        return

    # Panel (b) uses fixed k★ for the R★ sweep (Plot 2).
    k_star = K_STAR_R

    use_nature(fontsize=8)
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.15),
        gridspec_kw={"wspace": 0.45},
    )
    panel_a(axes[0], k_df)
    panel_b(axes[1], r_df, k_star)
    panel_c(axes[2], eta_df, fig=fig)

    written = save_paper_figure(fig, args.stem, formats=("png", "pdf", "eps"))
    for p in written:
        print(f"wrote {p}")
    plt.close(fig)

    write_caption(FIG_DIR / f"{args.stem}_caption.txt", k_df, r_df, eta_df)


if __name__ == "__main__":
    main()
