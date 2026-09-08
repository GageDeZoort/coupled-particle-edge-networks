"""Visualization helpers for star-$R$ hypergraph studies in (eta, phi)."""

from __future__ import annotations

from typing import Any

import numpy as np

from cpen.utils.graph_radius import (
    HypergraphBuild,
    build_star_radius_hypergraph,
    eta_phi_from_four_vectors,
    hyperedge_centered_on_particle,
    hyperedges_incident_on_particle,
    particle_mask_from_four_vectors,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    LineCollection = None  # type: ignore[misc, assignment]

SUPPORT_COLORS: dict[int, str] = {
    1: "#984ea3",
    2: "#e41a1c",
    3: "#377eb8",
    4: "#4daf4a",
    5: "#ff7f00",
    6: "#a65628",
    7: "#f781bf",
    8: "#999999",
}
DEFAULT_EDGE_COLOR = "#bbbbbb"


def _require_matplotlib() -> Any:
    if plt is None:
        raise ImportError("matplotlib is required; install with: pip install matplotlib")
    return plt


def reference_particle_index(jet: np.ndarray, *, strategy: str = "max_energy") -> int:
    """Pick a representative active particle (default: highest energy)."""
    mask = particle_mask_from_four_vectors(jet)
    active = np.flatnonzero(mask)
    if active.size == 0:
        raise ValueError("jet has no active particles")
    if strategy == "max_energy":
        energies = jet[active, 0]
        return int(active[int(np.argmax(energies))])
    if strategy == "max_pt":
        px, py = jet[active, 1], jet[active, 2]
        pt = np.hypot(px, py)
        return int(active[int(np.argmax(pt))])
    raise ValueError(f"Unknown strategy {strategy!r}")


def _jet_eta_phi_coords(jet: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta_all, phi_all = eta_phi_from_four_vectors(jet)
    eta = eta_all[active]
    phi = phi_all[active]
    energies = jet[active, 0]
    return eta, phi, energies


def _color_for_support(m: int) -> str:
    return SUPPORT_COLORS.get(m, DEFAULT_EDGE_COLOR)


def _global_to_active_map(active: np.ndarray) -> dict[int, int]:
    return {int(g): i for i, g in enumerate(active.tolist())}


def _star_spoke_segments(
    build: HypergraphBuild,
    hyperedge_row: int,
    pos: np.ndarray,
    global_to_active: dict[int, int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    center = int(build.center_particles[hyperedge_row])
    if center not in global_to_active:
        return []
    center_pos = pos[global_to_active[center]]
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for member in np.flatnonzero(build.incidence[hyperedge_row] > 0):
        m = int(member)
        if m == center or m not in global_to_active:
            continue
        segments.append((center_pos, pos[global_to_active[m]]))
    return segments


def plot_jet_particles_eta_phi(
    jet: np.ndarray,
    build: HypergraphBuild,
    *,
    ax: Any | None = None,
    label: str | None = None,
    size_scale: float = 350.0,
    ref_particle: int | None = None,
) -> Any:
    """Scatter active particles in (eta, phi); marker size ~ energy."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5))

    active = build.active_indices
    eta, phi, energies = _jet_eta_phi_coords(jet, active)
    sizes = size_scale * energies / max(float(energies.max()), 1e-6)

    ax.scatter(
        eta,
        phi,
        s=sizes,
        c="#333333",
        alpha=0.75,
        edgecolors="k",
        linewidths=0.25,
        zorder=3,
    )

    if ref_particle is not None and ref_particle in active:
        ref_local = int(np.where(active == ref_particle)[0][0])
        ax.scatter(
            [eta[ref_local]],
            [phi[ref_local]],
            s=sizes[ref_local] * 1.4,
            c="#d62728",
            edgecolors="k",
            linewidths=0.8,
            zorder=5,
            marker="*",
            label="reference particle",
        )

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title(label or "particles (star-R)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    if ref_particle is not None:
        ax.legend(loc="upper right", fontsize=9)
    return ax


def plot_star_center_hyperedge(
    jet: np.ndarray,
    build: HypergraphBuild,
    particle_idx: int,
    *,
    ax: Any | None = None,
    linewidth: float = 2.0,
    title: str | None = None,
) -> Any:
    """Draw only the star hyperedge centered on ``particle_idx``."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5))

    plot_jet_particles_eta_phi(jet, build, ax=ax, ref_particle=particle_idx, label=None)

    row = hyperedge_centered_on_particle(build, particle_idx)
    if row is None:
        ax.set_title(f"particle {particle_idx} inactive")
        return ax

    active = build.active_indices
    global_to_active = _global_to_active_map(active)
    eta, phi, _ = _jet_eta_phi_coords(jet, active)
    pos = np.stack([eta, phi], axis=1)

    support = int(build.particle_support_sizes[row])
    color = _color_for_support(support)
    segments = _star_spoke_segments(build, row, pos, global_to_active)
    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors=[color],
                linewidths=linewidth,
                alpha=0.9,
                zorder=4,
            )
        )

    ax.plot([], [], color=color, linewidth=linewidth, label=f"|support|={support}")
    ax.legend(loc="upper left", fontsize=9)
    if title is None:
        title = f"star centered on particle {particle_idx} (R-hyperedge row {row})"
    ax.set_title(title)
    ax.autoscale()
    return ax


def plot_particle_hyperedge_fan(
    jet: np.ndarray,
    build: HypergraphBuild,
    particle_idx: int,
    *,
    ax: Any | None = None,
    max_hyperedges: int | None = 24,
    linewidth: float = 1.2,
    alpha: float = 0.85,
    title: str | None = None,
) -> Any:
    """
    Draw hyperedges incident on ``particle_idx``, colored by support size.

    The star centered on the reference particle is drawn with thicker spokes.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6))

    plot_jet_particles_eta_phi(jet, build, ax=ax, ref_particle=particle_idx, label=None)

    active = build.active_indices
    global_to_active = _global_to_active_map(active)
    eta, phi, _ = _jet_eta_phi_coords(jet, active)
    pos = np.stack([eta, phi], axis=1)

    incident_rows = hyperedges_incident_on_particle(build, particle_idx)
    if max_hyperedges is not None and len(incident_rows) > max_hyperedges:
        is_center = build.center_particles[incident_rows] == particle_idx
        order = np.lexsort(
            (
                -build.particle_support_sizes[incident_rows],
                -is_center.astype(int),
            )
        )
        incident_rows = incident_rows[order[:max_hyperedges]]

    seen_legend: set[int] = set()
    for row in incident_rows:
        support = int(build.particle_support_sizes[row])
        color = _color_for_support(support)
        center = int(build.center_particles[row])
        segments = _star_spoke_segments(build, int(row), pos, global_to_active)
        if not segments:
            continue
        lw = linewidth * (1.8 if center == particle_idx else 1.0)
        ax.add_collection(
            LineCollection(segments, colors=[color], linewidths=lw, alpha=alpha, zorder=4)
        )
        if support not in seen_legend:
            ax.plot([], [], color=color, linewidth=lw, label=f"|support|={support}")
            seen_legend.add(support)

    ax.legend(loc="upper left", fontsize=8, title="support size")
    if title is None:
        title = (
            f"incident star hyperedges on particle {particle_idx} "
            f"({len(incident_rows)} shown)"
        )
    ax.set_title(title)
    ax.autoscale()
    return ax


def plot_star_hyperedge_panel(
    jet: np.ndarray,
    *,
    radius: float,
    particle_idx: int | None = None,
) -> Any:
    """Two-panel view: star centered on reference vs all incident stars."""
    plt = _require_matplotlib()
    if particle_idx is None:
        particle_idx = reference_particle_index(jet)

    build = build_star_radius_hypergraph(jet, radius=radius)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    plot_star_center_hyperedge(
        jet,
        build,
        particle_idx,
        ax=axes[0],
        title=f"own star (R={radius:.2f})",
    )
    plot_particle_hyperedge_fan(
        jet,
        build,
        particle_idx,
        ax=axes[1],
        title=f"all stars incident on particle (R={radius:.2f})",
    )
    e_ref = jet[particle_idx, 0]
    fig.suptitle(
        f"reference particle {particle_idx} (E={e_ref:.2f})",
        y=1.02,
    )
    fig.tight_layout()
    return fig, particle_idx, build


def plot_jet_subsample_panels(
    jets: np.ndarray,
    labels: np.ndarray,
    *,
    radius: float,
    n_per_class: int = 2,
    seed: int = 0,
) -> Any:
    """Grid of star hyperedge fan plots for a small random subsample."""
    plt = _require_matplotlib()
    rng = np.random.default_rng(seed)
    n_rows = n_per_class * 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(6.5, 4.5 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    row = 0
    for class_label, class_name in [(1, "top"), (0, "QCD")]:
        pool = np.flatnonzero(labels == class_label)
        pick = rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)
        for idx in pick:
            jet = jets[idx]
            ref = reference_particle_index(jet)
            build = build_star_radius_hypergraph(jet, radius=radius)
            plot_particle_hyperedge_fan(
                jet,
                build,
                ref,
                ax=axes[row],
                title=f"{class_name} jet {idx} — star-R (ref E={jet[ref, 0]:.1f})",
            )
            row += 1

    fig.suptitle(
        f"Star-$R$ hyperedges (R={radius:.2f}) — reference = highest-$E$ particle",
        y=1.01,
    )
    fig.tight_layout()
    return fig
