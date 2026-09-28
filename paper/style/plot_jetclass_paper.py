"""Nature-style JetClass paper figures."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from paper.style.nature_plots import nature_color, save_paper_figure, use_nature

DATA = REPO / "paper" / "data"
FIG = REPO / "paper" / "figures"


def plot_ablation_graph_1m():
    """a: edges vs M11; b: kNN k (+ no-star); c: star radius R★."""
    use_nature()
    edges = pd.read_csv(DATA / "edges_vs_m11_1m.csv")
    graph = pd.read_csv(DATA / "graph_ablation_1m.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05),
                             gridspec_kw={"wspace": 0.40})
    ax0, ax1, ax2 = axes

    # --- a: edges help ---
    for model, color, marker, label in [
        ("edges", nature_color(0), "o", "with edges"),
        ("m11", nature_color(1), "s", "M11 only"),
    ]:
        sub = edges[edges.model == model].sort_values("eta0")
        ax0.plot(sub.eta0, sub.val_roc_auc, marker=marker, color=color, ms=2.6, lw=0.9, label=label)
    ax0.set_xscale("log")
    ax0.set_xlabel(r"$\eta_0$")
    ax0.set_ylabel("Validation ROC AUC")
    ax0.set_title("a  Edges help", loc="left", pad=2)
    ax0.legend(loc="lower center", fontsize=5.5, handlelength=1.4)
    ax0.set_ylim(0.895, 0.932)
    ax0.text(0.04, 0.04, r"1M, 1 ep; $L{=}3$, $D{=}256$", transform=ax0.transAxes,
             fontsize=5, color="0.35", va="bottom")

    # --- b: kNN saturation; k=6 chosen ---
    matched = graph[graph.panel == "graph_matched"].copy()
    knn = matched[matched.variant.str.startswith("kNN")].copy()
    knn["k"] = knn.variant.str.extract(r"k=(\d+)").astype(int)
    knn = knn.sort_values("k")
    nostar = matched[matched.variant == "no star"]
    ax1.plot(knn.k, knn.val_roc_auc, "o-", color=nature_color(0), ms=3.0, lw=0.9, label="with star")
    # highlight chosen k=6
    k6 = knn[knn.k == 6].iloc[0]
    ax1.plot(k6.k, k6.val_roc_auc, "o", color=nature_color(0), ms=5.0,
             markeredgecolor="k", markeredgewidth=0.45, zorder=4)
    if len(nostar):
        ax1.plot([6], [float(nostar.val_roc_auc.iloc[0])], marker="x", color=nature_color(1),
                 ms=5.0, mew=0.9, linestyle="None", label="no star ($k{=}6$)")
    ax1.set_xticks([2, 4, 6, 8])
    ax1.set_xlabel(r"kNN $k$")
    ax1.set_ylabel("Validation ROC AUC")
    ax1.set_title(r"b  Choose $k{=}6$", loc="left", pad=2)
    ax1.legend(loc="lower right", fontsize=5.5, handlelength=1.4)
    ax1.text(0.04, 0.04, r"2-ep matched; $L{=}4$", transform=ax1.transAxes,
             fontsize=5, color="0.35", va="bottom")

    # --- c: star radius → justify R★=0.2 ---
    star = graph[graph.panel == "star_radius"].sort_values("r_star")
    ax2.plot(star.r_star, star.val_roc_auc, "s-", color=nature_color(3), ms=3.0, lw=0.9)
    chosen = star[star.r_star == 0.2]
    if len(chosen):
        ax2.plot(chosen.r_star, chosen.val_roc_auc, "s", color=nature_color(3), ms=5.0,
                 markeredgecolor="k", markeredgewidth=0.45, zorder=4,
                 label=r"chosen $R_\star{=}0.2$")
        ax2.legend(loc="upper right", fontsize=5.5, handlelength=1.0)
    ax2.set_xticks([0.10, 0.15, 0.20, 0.30])
    ax2.set_xlabel(r"Star radius $R_\star$")
    ax2.set_ylabel("Validation ROC AUC")
    ax2.set_title(r"c  Star radius $R_\star$", loc="left", pad=2)
    ax2.text(0.04, 0.04, r"1-ep matched; $L{=}4$", transform=ax2.transAxes,
             fontsize=5, color="0.35", va="bottom")

    paths = save_paper_figure(fig, "fig_jetclass_ablation_1m")
    plt.close(fig)
    return paths


def _width_label(width: int, heads: int) -> str:
    return f"D={int(width)}, H={int(heads)}"


def plot_transfer_adam_45k():
    use_nature()
    df = pd.read_csv(DATA / "transfer_adam_45k.csv")

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2), sharex=True,
                             gridspec_kw={"wspace": 0.32})
    ax_loss, ax_roc = axes

    # Drop D384 η₀=2.5 (diverged) so both panels stay readable.
    series = [
        (256, "o-", nature_color(0), 2.2),
        (384, "s-", nature_color(1), 2.2),
        (512, "*", nature_color(2), 4.0),
        (768, "D", nature_color(3), 2.4),
    ]
    for width, style, color, ms in series:
        sub = df[df.width == width].sort_values("eta0")
        if width == 384:
            sub = sub[sub.eta0 <= 1.0]
        if len(sub) == 0:
            continue
        heads = int(sub.heads.iloc[0])
        label = _width_label(width, heads)
        if "-" in style:  # connected η₀ grid
            ax_loss.plot(sub.eta0, sub.train_loss, style, color=color, ms=ms, lw=0.9, label=label)
            ax_roc.plot(sub.eta0, sub.val_roc_auc, style, color=color, ms=ms, lw=0.9, label=label)
        else:  # single-η₀ marker
            ax_loss.plot(sub.eta0, sub.train_loss, marker=style, color=color, ms=ms,
                         linestyle="None", label=label)
            ax_roc.plot(sub.eta0, sub.val_roc_auc, marker=style, color=color, ms=ms,
                        linestyle="None", label=label)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\eta_0$")

    ax_loss.set_ylabel("Train loss")
    ax_loss.set_ylim(0.775, 0.820)
    ax_roc.set_ylabel("Validation ROC AUC")
    ax_roc.set_ylim(0.9556, 0.9605)

    ax_loss.legend(loc="upper right", fontsize=5.5, handlelength=1.4,
                   labelspacing=0.25, borderpad=0.2, handletextpad=0.35)

    paths = save_paper_figure(fig, "fig_jetclass_adam_transfer_45k")
    plt.close(fig)
    return paths


def plot_graph_geometry_k_R():
    """kNN edges vs k; star support vs R★; JetClass ROC plateau vs R★."""
    use_nature()
    knn = pd.read_csv(DATA / "graph_geometry_knn.csv")
    star = pd.read_csv(DATA / "graph_geometry_star.csv")
    roc = pd.read_csv(DATA / "graph_geometry_star_roc.csv").sort_values("radius")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05),
                             gridspec_kw={"wspace": 0.42})
    ax_k, ax_r, ax_roc = axes

    # --- kNN ---
    for cls, color, marker in [("QCD", nature_color(0), "o"), ("top", nature_color(1), "s")]:
        sub = knn[(knn["class"] == cls) & (knn.k <= 16)].sort_values("k")
        ax_k.plot(sub.k, sub.n_edges_mean, marker=marker, color=color, ms=2.6, lw=0.9, label=cls)
    ax_k.axvline(6, color="0.45", lw=0.75, ls="--", zorder=0)
    ax_k.plot([6], [float(knn[(knn["class"] == "QCD") & (knn.k == 6)].n_edges_mean.iloc[0])],
              "o", color=nature_color(0), ms=5.0, markeredgecolor="k", markeredgewidth=0.45, zorder=4)
    ax_k.plot([6], [float(knn[(knn["class"] == "top") & (knn.k == 6)].n_edges_mean.iloc[0])],
              "s", color=nature_color(1), ms=5.0, markeredgecolor="k", markeredgewidth=0.45, zorder=4)
    ax_k.set_xlabel(r"kNN $k$")
    ax_k.set_ylabel("Mean edges / jet")
    ax_k.set_xticks([1, 2, 4, 6, 8, 16])
    ax_k.legend(loc="upper left", fontsize=5.5, handlelength=1.3)
    ax_k.text(0.97, 0.04, r"chosen $k{=}6$", transform=ax_k.transAxes,
              fontsize=5.5, color="0.35", ha="right", va="bottom")

    # --- star support vs R (extended) ---
    for cls, color, marker in [("QCD", nature_color(0), "o"), ("top", nature_color(1), "s")]:
        sub = star[star["class"] == cls].sort_values("radius")
        ax_r.plot(sub.radius, sub.mean_support_size, marker=marker, color=color,
                  ms=2.6, lw=0.9, label=cls)
    ax_r.axhline(6.0, color="0.45", lw=0.75, ls=":", zorder=0)
    ax_r.axvline(0.2, color="0.45", lw=0.75, ls="--", zorder=0)
    for cls, color, marker in [("QCD", nature_color(0), "o"), ("top", nature_color(1), "s")]:
        hit = star[(star["class"] == cls) & (np.isclose(star.radius, 0.2))]
        if len(hit):
            ax_r.plot([0.2], [float(hit.mean_support_size.iloc[0])], marker=marker,
                      color=color, ms=5.0, markeredgecolor="k", markeredgewidth=0.45, zorder=4)
    ax_r.set_xlabel(r"Star radius $R_\star$")
    ax_r.set_ylabel(r"Mean star support $|S_c|$")
    ax_r.set_xticks([0.05, 0.1, 0.2, 0.3, 0.4])
    ax_r.legend(loc="upper left", fontsize=5.5, handlelength=1.3)
    ax_r.text(0.04, 0.04, r"dotted: $k{=}6$ scale", transform=ax_r.transAxes,
              fontsize=5.5, color="0.35", va="bottom")
    ax_r.text(0.97, 0.04, r"chosen $R_\star{=}0.2$", transform=ax_r.transAxes,
              fontsize=5.5, color="0.35", ha="right", va="bottom")

    # --- ROC plateau vs R (JetClass ablation) ---
    ax_roc.plot(roc.radius, roc.val_roc_auc, "D-", color=nature_color(3), ms=3.0, lw=0.9)
    chosen = roc[np.isclose(roc.radius, 0.2)]
    if len(chosen):
        ax_roc.plot(chosen.radius, chosen.val_roc_auc, "D", color=nature_color(3), ms=5.0,
                    markeredgecolor="k", markeredgewidth=0.45, zorder=4)
    ax_roc.axvline(0.2, color="0.45", lw=0.75, ls="--", zorder=0)
    # shade the broad high-ROC plateau [0.1, 0.2]
    ymin, ymax = float(roc.val_roc_auc.min()) - 0.0008, float(roc.val_roc_auc.max()) + 0.0008
    ax_roc.axvspan(0.1, 0.2, color=nature_color(3), alpha=0.10, zorder=0)
    ax_roc.set_ylim(ymin, ymax)
    ax_roc.set_xlabel(r"Star radius $R_\star$")
    ax_roc.set_ylabel("Validation ROC AUC")
    ax_roc.set_xticks([0.10, 0.15, 0.20, 0.30])
    ax_roc.text(0.04, 0.04, r"1M JetClass, 1 ep; $L{=}4$", transform=ax_roc.transAxes,
                fontsize=5.5, color="0.35", va="bottom")
    ax_roc.text(0.97, 0.96, "plateau", transform=ax_roc.transAxes,
                fontsize=5.5, color="0.35", ha="right", va="top")

    paths = save_paper_figure(fig, "fig_jetclass_graph_geometry")
    plt.close(fig)
    return paths


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("ablation:", *plot_ablation_graph_1m(), sep="\n  ")
    print("geometry:", *plot_graph_geometry_k_R(), sep="\n  ")
    print("transfer:", *plot_transfer_adam_45k(), sep="\n  ")


if __name__ == "__main__":
    main()
