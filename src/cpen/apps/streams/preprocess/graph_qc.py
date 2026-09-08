"""Scan pipeline graph manifests and plot population statistics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cpen.apps.streams.preprocess.config import DEFAULT_PIPELINE_ROOT, SPLITS
from cpen.apps.streams.preprocess.gmm import BLOB_N_MAX
from cpen.apps.streams.preprocess.pipeline import (
    BLOBS_SUBDIR,
    CELLS_SUBDIR,
    GMM_SUBDIR,
    GRAPHS_SUBDIR,
    PHYS_SUBDIR,
)

_STAGES = (
    ("cells", CELLS_SUBDIR),
    ("phys", PHYS_SUBDIR),
    ("gmm", GMM_SUBDIR),
    ("blobs", BLOBS_SUBDIR),
    ("graphs", GRAPHS_SUBDIR),
)
_SPLIT_COLORS = {"train": "#4c78a8", "val": "#e45756", "test": "#72b7b2"}


def _done(path: Path) -> bool:
    return (path / ".done").is_file()


def list_galaxy_out_dirs(out_root=DEFAULT_PIPELINE_ROOT) -> list[tuple[str, str, Path]]:
    """``(split, galaxy_id, galaxy_dir)`` for every ``{out_root}/{split}/{id}``."""
    root = Path(out_root)
    rows = []
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for d in sorted(split_dir.iterdir()):
            if d.is_dir() and d.name.isdigit():
                rows.append((split, d.name, d))
    return rows


def pipeline_coverage(out_root=DEFAULT_PIPELINE_ROOT) -> pd.DataFrame:
    """One row per galaxy directory: which stages have ``.done``."""
    rows = []
    for split, gid, d in list_galaxy_out_dirs(out_root):
        row = {"split": split, "galaxy_id": gid, "path": str(d)}
        for name, sub in _STAGES:
            row[name] = _done(d / sub)
        man = d / GRAPHS_SUBDIR / "manifest.parquet"
        row["n_graphs"] = int(len(pd.read_parquet(man))) if man.is_file() else 0
        rows.append(row)
    cols = ["split", "galaxy_id", "cells", "phys", "gmm", "blobs", "graphs", "n_graphs", "path"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def load_graph_manifests(out_root=DEFAULT_PIPELINE_ROOT) -> pd.DataFrame:
    """Concatenate every ``graphs/manifest.parquet``, joined with blob width when present."""
    frames = []
    for split, gid, d in list_galaxy_out_dirs(out_root):
        gman = d / GRAPHS_SUBDIR / "manifest.parquet"
        if not gman.is_file():
            continue
        tab = pd.read_parquet(gman)
        tab["split"] = split
        tab["galaxy_id"] = gid
        bman = d / BLOBS_SUBDIR / "manifest.parquet"
        if bman.is_file():
            blobs = pd.read_parquet(bman)
            key = ["cell_id", "gmm_label", "blob_part"]
            if all(c in blobs.columns for c in key) and "width" in blobs.columns:
                tab = tab.merge(
                    blobs[key + ["width"]], on=key, how="left", suffixes=("", "_blob"),
                )
        frames.append(tab)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "n_stars" in out.columns and "n_mock" in out.columns:
        out["mock_frac"] = np.where(out["n_stars"] > 0, out["n_mock"] / out["n_stars"], np.nan)
        out["is_empty"] = out["n_mock"] == 0
    if "n_knn" in out.columns and "n_knn_pos" in out.columns:
        out["knn_pos_frac"] = np.where(out["n_knn"] > 0, out["n_knn_pos"] / out["n_knn"], np.nan)
    if "n_hyper" in out.columns and "n_hyper_pos" in out.columns:
        out["hyper_pos_frac"] = np.where(out["n_hyper"] > 0, out["n_hyper_pos"] / out["n_hyper"], np.nan)
    return out


def print_graph_population(tab: pd.DataFrame, coverage: pd.DataFrame | None = None) -> None:
    """Stdout summary of graph-pool size / occupancy / labels."""
    if coverage is not None and len(coverage):
        print("===== Pipeline coverage =====")
        for split, sub in coverage.groupby("split", sort=False):
            done = int(sub["graphs"].sum())
            print(
                f"  {split:5s}  galaxies on disk {len(sub):2d}  "
                f"graphs-done {done:2d}  n_graphs {int(sub.n_graphs.sum()):,}"
            )
        pending = coverage[~coverage["graphs"]]
        if len(pending):
            print("  still running / not started:")
            print(
                pending[["split", "galaxy_id", "cells", "phys", "gmm", "blobs", "graphs"]]
                .to_string(index=False)
            )
        print()
    if tab is None or not len(tab):
        print("No graph manifests yet.")
        return
    print(f"===== Graphs: {len(tab):,}  galaxies {tab.groupby(['split','galaxy_id']).ngroups} =====")
    print(
        f"  N: median {tab.n_stars.median():,.0f}  "
        f"p90 {tab.n_stars.quantile(0.9):,.0f}  "
        f"max {int(tab.n_stars.max()):,}  "
        f"(cap {BLOB_N_MAX:,})"
    )
    n_over = int((tab.n_stars > BLOB_N_MAX).sum())
    print(f"  N>{BLOB_N_MAX:,}: {n_over}")
    empty = tab[tab["is_empty"]] if "is_empty" in tab.columns else tab.iloc[0:0]
    print(
        f"  empty (0 mock): {len(empty):,} ({100 * len(empty) / len(tab):.1f}%)  "
        f"with ≥1 mock: {len(tab) - len(empty):,}"
    )
    if "n_knn" in tab.columns:
        print(
            f"  kNN+ {100 * tab.n_knn_pos.sum() / max(tab.n_knn.sum(), 1):.2f}%  "
            f"hyper+ {100 * tab.n_hyper_pos.sum() / max(tab.n_hyper.sum(), 1):.2f}%"
        )
    print()
    print("  by split:")
    for split, sub in tab.groupby("split", sort=False):
        n_e = int(sub["is_empty"].sum()) if "is_empty" in sub.columns else 0
        print(
            f"    {split:5s}  graphs {len(sub):5,}  "
            f"median N={sub.n_stars.median():,.0f}  "
            f"empty {100 * n_e / len(sub):.0f}%  "
            f"max N={int(sub.n_stars.max()):,}"
        )


def plot_graph_population(
    tab: pd.DataFrame,
    coverage: pd.DataFrame | None = None,
    *,
    n_max: int = BLOB_N_MAX,
    show: bool = True,
    savepath=None,
    dpi: int = 140,
):
    """Size, occupancy, and edge-label population plots for the graph pool."""
    import matplotlib.pyplot as plt

    if tab is None or not len(tab):
        print("No graphs to plot yet.")
        return None

    splits = [s for s in SPLITS if s in set(tab["split"])]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), dpi=dpi)
    ax_size, ax_mock, ax_frac = axes[0]
    ax_knn, ax_hyp, ax_sc = axes[1]

    bins_n = np.linspace(0, max(float(n_max), float(tab.n_stars.max())), 41)
    for split in splits:
        sub = tab[tab["split"] == split]
        color = _SPLIT_COLORS.get(split, "#444")
        ax_size.hist(
            sub.n_stars, bins=bins_n, histtype="step", lw=1.8,
            color=color, label=f"{split} ({len(sub):,})",
        )
    ax_size.axvline(n_max, color="0.3", ls="--", lw=1.0, label=rf"$N_{{\max}}={n_max:,}$")
    ax_size.set_xlabel("stars / graph")
    ax_size.set_ylabel("# graphs")
    ax_size.set_title("Graph size")
    ax_size.legend(fontsize=8, framealpha=0.9)
    ax_size.grid(True, axis="y", ls=":", alpha=0.4)

    mock_hi = max(float(tab.n_mock.max()), 1.0)
    bins_m = np.concatenate([[-0.5, 0.5], np.linspace(1, mock_hi, 32)])
    for split in splits:
        sub = tab[tab["split"] == split]
        ax_mock.hist(
            sub.n_mock, bins=bins_m, histtype="step", lw=1.8,
            color=_SPLIT_COLORS.get(split, "#444"), label=split,
        )
    ax_mock.set_xlabel("mock stars / graph")
    ax_mock.set_ylabel("# graphs")
    ax_mock.set_title("Mock occupancy")
    ax_mock.set_yscale("log")
    ax_mock.grid(True, axis="y", ls=":", alpha=0.4)

    counts = []
    labels = []
    colors = []
    for split in splits:
        sub = tab[tab["split"] == split]
        n_empty = int(sub["is_empty"].sum()) if "is_empty" in sub.columns else 0
        n_host = len(sub) - n_empty
        counts.append([n_empty, n_host])
        labels.append(split)
        colors.append(_SPLIT_COLORS.get(split, "#444"))
    counts = np.asarray(counts, dtype=float)
    x = np.arange(len(labels))
    ax_frac.bar(x, counts[:, 0] / counts.sum(axis=1), color="#9aa5b1", label="empty")
    ax_frac.bar(
        x, counts[:, 1] / counts.sum(axis=1), bottom=counts[:, 0] / counts.sum(axis=1),
        color="#c0392b", label="≥1 mock",
    )
    ax_frac.set_xticks(x, labels)
    ax_frac.set_ylim(0, 1.05)
    ax_frac.set_ylabel("fraction of graphs")
    ax_frac.set_title("Empty vs mock-hosting")
    ax_frac.legend(fontsize=8, framealpha=0.9)
    ax_frac.grid(True, axis="y", ls=":", alpha=0.4)

    if "knn_pos_frac" in tab.columns:
        for split in splits:
            sub = tab[tab["split"] == split]
            ax_knn.hist(
                sub.knn_pos_frac.dropna(), bins=np.linspace(0, 1, 21),
                histtype="step", lw=1.8, color=_SPLIT_COLORS.get(split, "#444"), label=split,
            )
        ax_knn.set_xlabel(r"kNN $y{+} / E$")
        ax_knn.set_ylabel("# graphs")
        ax_knn.set_title("kNN positive fraction")
        ax_knn.grid(True, axis="y", ls=":", alpha=0.4)
    if "hyper_pos_frac" in tab.columns:
        for split in splits:
            sub = tab[tab["split"] == split]
            ax_hyp.hist(
                sub.hyper_pos_frac.dropna(), bins=np.linspace(0, 1, 21),
                histtype="step", lw=1.8, color=_SPLIT_COLORS.get(split, "#444"), label=split,
            )
        ax_hyp.set_xlabel(r"hyper $y{+} / H$")
        ax_hyp.set_ylabel("# graphs")
        ax_hyp.set_title("Hyperedge positive fraction")
        ax_hyp.grid(True, axis="y", ls=":", alpha=0.4)

    rng = np.random.default_rng(0)
    take = tab if len(tab) <= 8000 else tab.sample(8000, random_state=0)
    jitter = 0.15 * rng.uniform(-1, 1, len(take))
    ax_sc.scatter(
        take.n_stars + jitter, take.n_mock + 0.15 * rng.uniform(-1, 1, len(take)),
        s=8, alpha=0.25, c=take["split"].map(_SPLIT_COLORS).fillna("#444"),
        linewidths=0,
    )
    ax_sc.axvline(n_max, color="0.3", ls="--", lw=1.0)
    ax_sc.set_xlabel("stars / graph")
    ax_sc.set_ylabel("mock stars")
    ax_sc.set_title("Size vs mock count")
    ax_sc.grid(True, ls=":", alpha=0.4)

    n_gal = int(tab.groupby(["split", "galaxy_id"]).ngroups)
    fig.suptitle(
        f"Graph population  {len(tab):,} graphs  {n_gal} galaxies  "
        f"empty {100 * tab.is_empty.mean():.0f}%",
        fontsize=12, fontweight="medium",
    )
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, axes


def plot_graphs_per_galaxy(
    tab: pd.DataFrame,
    *,
    show: bool = True,
    savepath=None,
    dpi: int = 140,
):
    """Graphs per galaxy, stacked empty vs mock-hosting."""
    import matplotlib.pyplot as plt

    if tab is None or not len(tab):
        print("No graphs to plot yet.")
        return None
    g = (
        tab.groupby(["split", "galaxy_id"], sort=False)
        .agg(n=("n_stars", "size"), n_empty=("is_empty", "sum"), n_mock_stars=("n_mock", "sum"))
        .reset_index()
    )
    g["n_host"] = g["n"] - g["n_empty"]
    g["label"] = g["split"] + "/" + g["galaxy_id"]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.35 * len(g) + 2.0), 3.6), dpi=dpi)
    x = np.arange(len(g))
    ax.bar(x, g.n_empty, color="#9aa5b1", label="empty")
    ax.bar(x, g.n_host, bottom=g.n_empty, color="#c0392b", label="≥1 mock")
    ax.set_xticks(x, g.label, rotation=90, fontsize=8)
    ax.set_ylabel("# graphs")
    ax.set_title("Graphs per galaxy")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, ax


def plot_graph_coverage(
    coverage: pd.DataFrame,
    *,
    show: bool = True,
    savepath=None,
    dpi: int = 140,
):
    """Per-galaxy stage completion (useful while Slurm arrays are still running)."""
    import matplotlib.pyplot as plt

    if coverage is None or not len(coverage):
        print("No galaxy output directories yet.")
        return None
    stages = ["cells", "phys", "gmm", "blobs", "graphs"]
    cov = coverage.sort_values(["split", "galaxy_id"]).reset_index(drop=True)
    labels = [f"{r.split}/{r.galaxy_id}" for r in cov.itertuples(index=False)]
    mat = cov[stages].to_numpy(bool).astype(float)
    fig_h = max(3.4, 0.28 * len(cov) + 1.2)
    fig, ax = plt.subplots(figsize=(6.4, fig_h), dpi=dpi)
    ax.imshow(mat, aspect="auto", cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(len(stages)), stages)
    ax.set_yticks(np.arange(len(cov)), labels, fontsize=8)
    ax.set_title("Pipeline stage done (.done)")
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, ax
