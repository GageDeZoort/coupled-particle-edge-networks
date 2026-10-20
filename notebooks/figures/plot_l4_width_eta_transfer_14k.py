#!/usr/bin/env python3
"""L4 JetClass: width × η₀ transfer at the common D512/D768 step (~14k).

D256 (H=8) and D384 (H=12) show their available η₀ grids; D512 (H=16) and
D768 (H=24) so far have only η₀=0.25. Metrics taken from the nearest val
checkpoint to step 14000 (14001 / 14002).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Prefer scratch (home quota is often full); fall back to this directory.
_SCRATCH = Path(
    "/scratch/gpfs/BHANIN/jdezoort/posttrain_scaling/plots/l4_width_eta_transfer_14k"
)
_LOCAL = Path(__file__).resolve().parent / "l4_width_eta_transfer_14k"
OUT = _SCRATCH if _SCRATCH.parent.is_dir() else _LOCAL
CSV = OUT.with_suffix(".csv")

STEP = 14000  # common footing (D768 max ≈14.1k; D512 already past)

# D256 H=8 — η₀ grid {0.1, 0.25, 0.5} @14001
etas_d256 = np.array([0.1, 0.25, 0.5])
d256_roc = np.array([0.955987, 0.956118, 0.956395])
d256_acc = np.array([0.70516, 0.70504, 0.70320])

# D384 H=12 — full η₀ sweep @14001
etas_d384 = np.array([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5])
d384_roc = np.array(
    [0.952438, 0.954722, 0.951330, 0.955771, 0.956328, 0.955997, 0.955354, 0.865466]
)
d384_acc = np.array(
    [0.69516, 0.70360, 0.69324, 0.70580, 0.70552, 0.70640, 0.70348, 0.47384]
)

# D512 H=16 / D768 H=24 — single η₀=0.25 point
eta_wide = 0.25
d512_roc, d512_acc, d512_step = 0.956482, 0.70808, 14001
d768_roc, d768_acc, d768_step = 0.956630, 0.70668, 14002

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 8.5,
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
C_D512 = "#c0392b"
C_D768 = "#8e44ad"

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), constrained_layout=True)

for ax, ylabel, key in [
    (axes[0], "Val ROC AUC", "roc"),
    (axes[1], "Val Acc", "acc"),
]:
    y384 = d384_roc if key == "roc" else d384_acc
    y256 = d256_roc if key == "roc" else d256_acc
    y512 = d512_roc if key == "roc" else d512_acc
    y768 = d768_roc if key == "roc" else d768_acc

    # D384 full η₀ sweep as context
    ax.plot(
        etas_d384,
        y384,
        "D--",
        color=C_D384_WIDE,
        label="D384 H=12 (η₀ grid)",
        markersize=5,
        linewidth=1.2,
        alpha=0.55,
        markerfacecolor="white",
        markeredgewidth=1.2,
        markeredgecolor=C_D384_WIDE,
        zorder=1,
    )
    # emphasize matched mid-grid on D384
    mid = np.isin(etas_d384, etas_d256)
    ax.plot(
        etas_d384[mid],
        y384[mid],
        "s-",
        color=C_D384,
        label="D384 H=12 (matched)",
        markersize=7,
        linewidth=1.8,
        markerfacecolor="white",
        markeredgewidth=1.6,
        markeredgecolor=C_D384,
        zorder=3,
    )
    ax.plot(
        etas_d256,
        y256,
        "o-",
        color=C_D256,
        label="D256 H=8 (η₀ grid)",
        markersize=7,
        linewidth=1.8,
        markerfacecolor="white",
        markeredgewidth=1.6,
        markeredgecolor=C_D256,
        zorder=3,
    )
    # single-η₀ larger widths
    ax.plot(
        [eta_wide],
        [y512],
        "P",
        color=C_D512,
        label=r"D512 H=16 ($\eta_0$=0.25 only)",
        markersize=11,
        markerfacecolor="white",
        markeredgewidth=2.0,
        markeredgecolor=C_D512,
        zorder=4,
    )
    ax.plot(
        [eta_wide],
        [y768],
        "X",
        color=C_D768,
        label=r"D768 H=24 ($\eta_0$=0.25 only)",
        markersize=11,
        markerfacecolor="white",
        markeredgewidth=2.0,
        markeredgecolor=C_D768,
        zorder=4,
    )

    ax.set_xscale("log")
    ax.set_xticks(etas_d384)
    ax.set_xticklabels([f"{e:g}" for e in etas_d384])
    ax.set_xlabel(r"$\eta_0$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.65)
    ax.legend(frameon=False, loc="lower left" if key == "roc" else "best")

    # Zoom y to the competitive band (omit collapsed η₀=2.5 from scale)
    good = y384[:-1]  # drop 2.5
    ys = np.concatenate([good, y256, [y512, y768]])
    pad = 0.0025 if key == "roc" else 0.004
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    # annotate collapsed η₀=2.5
    ax.annotate(
        rf"D384 $\eta_0$=2.5: {y384[-1]:.3f}",
        xy=(2.5, ax.get_ylim()[0]),
        xytext=(-2, 8),
        textcoords="offset points",
        fontsize=7.5,
        color=C_D384_WIDE,
        ha="right",
    )

fig.suptitle(
    rf"L4 JetClass corr=1 — width $\times$ $\eta_0$ @~{STEP // 1000}k "
    rf"(common D512/D768 footing)",
    fontsize=12.5,
    fontweight="semibold",
    y=1.02,
)

png_path = OUT.with_suffix(".png")
pdf_path = OUT.with_suffix(".pdf")
fig.savefig(png_path)
fig.savefig(pdf_path)
plt.close(fig)

lines = ["eta0,D,H,step,val_acc,val_roc_auc,series"]
for eta, acc, roc in zip(etas_d256, d256_acc, d256_roc):
    lines.append(f"{eta:g},256,8,14001,{acc:.5f},{roc:.6f},grid")
for eta, acc, roc in zip(etas_d384, d384_acc, d384_roc):
    series = "matched" if eta in set(etas_d256.tolist()) else "wide"
    lines.append(f"{eta:g},384,12,14001,{acc:.5f},{roc:.6f},{series}")
lines.append(f"{eta_wide:g},512,16,{d512_step},{d512_acc:.5f},{d512_roc:.6f},single")
lines.append(f"{eta_wide:g},768,24,{d768_step},{d768_acc:.5f},{d768_roc:.6f},single")
CSV.write_text("\n".join(lines) + "\n")

print(f"Common footing: ~{STEP} steps (D512 val@{d512_step}, D768 val@{d768_step})")
print("eta\tD\tval_acc\tval_roc_auc")
for eta, acc, roc in zip(etas_d256, d256_acc, d256_roc):
    print(f"{eta:g}\t256\t{acc:.5f}\t{roc:.6f}")
for eta, acc, roc in zip(etas_d384, d384_acc, d384_roc):
    print(f"{eta:g}\t384\t{acc:.5f}\t{roc:.6f}")
print(f"{eta_wide:g}\t512\t{d512_acc:.5f}\t{d512_roc:.6f}")
print(f"{eta_wide:g}\t768\t{d768_acc:.5f}\t{d768_roc:.6f}")

for p in (png_path, pdf_path, CSV):
    size = p.stat().st_size
    print(f"{p}  ({size:,} bytes / {size / 1024:.1f} KiB)")
