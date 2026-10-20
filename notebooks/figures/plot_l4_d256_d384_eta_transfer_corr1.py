#!/usr/bin/env python3
"""L4 JetClass baseline: D=256 H=8 vs D=384 H=12 performance vs eta0 (corr=1).

Primary footing: matched @45k steps on shared eta0 grid {0.1, 0.25, 0.5}.
D384 also shows the wider eta0 sweep @45k as a light context curve.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "l4_d256_d384_eta_transfer_corr1"
CSV = Path(__file__).resolve().parent / "l4_d256_d384_eta_transfer_corr1_45k.csv"

# Shared matched grid @45k
etas = np.array([0.1, 0.25, 0.5])

# D256 H=8 @45k (jobs 14324683 / 14327810 / 14330528)
d256_45k_roc = np.array([0.959560, 0.959780, 0.959343])
d256_45k_acc = np.array([0.71996, 0.72156, 0.72128])

# D384 H=12 @45k — matched points (jobs 14244173 / 14243369 / 14264044)
d384_45k_roc = np.array([0.959846, 0.960082, 0.959896])
d384_45k_acc = np.array([0.72356, 0.72640, 0.72556])

# D384 H=12 @45k — full eta0 sweep (context)
etas_d384_wide = np.array([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5])
d384_wide_roc = np.array(
    [0.957354, 0.958865, 0.959564, 0.959846, 0.960082, 0.959896, 0.958599, 0.947667]
)
d384_wide_acc = np.array(
    [0.71316, 0.71960, 0.72132, 0.72356, 0.72640, 0.72556, 0.71744, 0.67760]
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

C_D256 = "#2c3e50"
C_D384 = "#1a7a6d"
C_D384_WIDE = "#7a9e98"

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)

for ax, ylabel, key in [
    (axes[0], "Val ROC AUC", "roc"),
    (axes[1], "Val Acc", "acc"),
]:
    y_wide = d384_wide_roc if key == "roc" else d384_wide_acc
    ax.plot(
        etas_d384_wide,
        y_wide,
        "D--",
        color=C_D384_WIDE,
        label="D384 H=12 @45k (wide)",
        markersize=5,
        linewidth=1.2,
        alpha=0.55,
        markerfacecolor="white",
        markeredgewidth=1.2,
        markeredgecolor=C_D384_WIDE,
        zorder=1,
    )

    for x, roc, acc, color, fmt, lw, label in [
        (etas, d256_45k_roc, d256_45k_acc, C_D256, "o-", 2.0, "D256 H=8 @45k"),
        (etas, d384_45k_roc, d384_45k_acc, C_D384, "s-", 2.0, "D384 H=12 @45k"),
    ]:
        y = roc if key == "roc" else acc
        ax.plot(
            x,
            y,
            fmt,
            color=color,
            label=label,
            markersize=8,
            linewidth=lw,
            alpha=1.0,
            markerfacecolor="white",
            markeredgewidth=1.8,
            markeredgecolor=color,
            zorder=3,
        )

    ax.set_xscale("log")
    xticks = etas_d384_wide
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{e:g}" for e in xticks])
    ax.set_xlabel(r"$\eta_0$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.65)
    ax.legend(frameon=False, loc="best")

fig.suptitle(
    r"L4 JetClass baseline corr=1 — $\eta_0$ transfer @45k",
    fontsize=13,
    fontweight="semibold",
    y=1.02,
)

png_path = OUT.with_suffix(".png")
pdf_path = OUT.with_suffix(".pdf")
fig.savefig(png_path)
fig.savefig(pdf_path)
plt.close(fig)

# CSV of all plotted points
lines = ["eta0,D,H,step,val_acc,val_roc_auc,series"]
for eta, acc, roc in zip(etas, d256_45k_acc, d256_45k_roc):
    lines.append(f"{eta:g},256,8,45000,{acc:.5f},{roc:.6f},matched")
for eta, acc, roc in zip(etas_d384_wide, d384_wide_acc, d384_wide_roc):
    series = "matched" if eta in set(etas.tolist()) else "wide"
    lines.append(f"{eta:g},384,12,45000,{acc:.5f},{roc:.6f},{series}")
CSV.write_text("\n".join(lines) + "\n")

print("Numbers used (@45k):")
print("eta\tD\tval_acc\tval_roc_auc")
for eta, acc, roc in zip(etas, d256_45k_acc, d256_45k_roc):
    print(f"{eta:g}\t256\t{acc:.5f}\t{roc:.6f}")
for eta, acc, roc in zip(etas_d384_wide, d384_wide_acc, d384_wide_roc):
    print(f"{eta:g}\t384\t{acc:.5f}\t{roc:.6f}")

eta_star_256 = float(etas[int(np.argmax(d256_45k_roc))])
eta_star_384 = float(etas_d384_wide[int(np.argmax(d384_wide_roc))])
print(f"η★ (max ROC @45k): D256={eta_star_256:g}  D384={eta_star_384:g}")

for p in (png_path, pdf_path, CSV):
    size = p.stat().st_size
    print(f"{p}  ({size:,} bytes / {size / 1024:.1f} KiB)")
