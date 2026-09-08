"""Plotting helpers for studying jet graph / hypergraph construction fidelity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import torch

from cpen.utils.graphs import (
    build_knn_graph,
    lorentz_dot,
    particle_mask_from_four_vectors,
    spatial_coords_from_four_vectors,
)
from cpen.utils.graph_hypergraph import build_knn_graph_single, build_star_hypergraph
from cpen.utils.toptagging import (
    RawTopTaggingSplit,
    energy_weights,
    preprocess_particle_features,
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon
except ImportError:  # pragma: no cover - optional for training installs
    plt = None  # type: ignore[assignment]
    LineCollection = None  # type: ignore[misc, assignment]
    Polygon = None  # type: ignore[misc, assignment]

ConstructionName = Literal["knn", "star-hypergraph"]
Plane = Literal["px-py", "y-z", "eta-phi"]


@dataclass
class JetGraph:
    """Single-jet graph bundle for visualization and diagnostics."""

    x_raw: torch.Tensor
    mask: torch.Tensor
    edge_index: torch.Tensor
    edge_x: torch.Tensor
    incidence: torch.Tensor
    label: int | None = None
    construction: str = "8-NN"

    @property
    def n_particles(self) -> int:
        return int(self.x_raw.size(0))

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.size(0))

    def active_particles(self) -> torch.Tensor:
        return self.mask.nonzero(as_tuple=False).squeeze(-1)


def _require_matplotlib() -> Any:
    if plt is None:
        raise ImportError(
            "matplotlib is required for graph plotting. Install with: pip install matplotlib"
        )
    return plt


def load_jet_sample(
    *,
    data_root: str,
    split: str = "train",
    idx: int = 0,
    num_particles: int = 100,
) -> dict[str, torch.Tensor | int]:
    """Load one jet from local TopTagging HDF5 with raw + model features."""
    backend = RawTopTaggingSplit(
        data_root=data_root,
        split=split,
        num_particles=num_particles,
    )
    particles, jet = backend[idx]
    x_raw, x_model, mask = preprocess_particle_features(particles.float())
    return {
        "x_raw": x_raw,
        "x": x_model,
        "mask": mask,
        "z": energy_weights(x_raw, mask),
        "y": int(jet[0].item()),
    }


def build_jet_graph(
    sample: dict[str, torch.Tensor | int],
    *,
    k: int = 8,
    construction: ConstructionName = "knn",
    graph_construction: str | None = None,
) -> JetGraph:
    """Build a :class:`JetGraph` from a :func:`load_jet_sample` dict."""
    if graph_construction is not None:
        from cpen.utils.graphs import parse_graph_construction

        k = parse_graph_construction(graph_construction)

    x_raw = sample["x_raw"]
    mask = sample["mask"]
    label = sample.get("y")
    if construction == "knn":
        edge_index, edge_x, incidence = build_knn_graph_single(x_raw, k=k, mask=mask)
        tag = f"{k}-NN"
    elif construction == "star-hypergraph":
        edge_index, edge_x, incidence = build_star_hypergraph(x_raw, k=k, mask=mask)
        tag = f"star-{k}"
    else:
        raise ValueError(f"Unknown construction {construction!r}")

    return JetGraph(
        x_raw=x_raw,
        mask=mask,
        edge_index=edge_index,
        edge_x=edge_x,
        incidence=incidence,
        label=int(label) if label is not None else None,
        construction=tag,
    )


def _plane_coords(x_raw: torch.Tensor, mask: torch.Tensor, plane: Plane) -> tuple[np.ndarray, np.ndarray]:
    active = mask.bool()
    x = x_raw[active].detach().cpu().numpy()
    if plane == "px-py":
        return x[:, 1], x[:, 2]
    if plane == "y-z":
        p = np.sqrt(np.maximum(x[:, 1] ** 2 + x[:, 2] ** 2, 1e-12))
        return np.arctanh(np.clip(x[:, 2] / p, -0.999, 0.999)), x[:, 3]
    # eta-phi proxy from px, py
    px, py = x[:, 1], x[:, 2]
    phi = np.arctan2(py, px)
    pt = np.sqrt(np.maximum(px * px + py * py, 1e-12))
    eta = np.arcsinh(np.clip(pt, 1e-6, None))
    return eta, phi


def summarize_graph(graph: JetGraph) -> dict[str, float | int | str]:
    """Scalar diagnostics for one constructed graph."""
    active = graph.active_particles()
    n = int(active.numel())
    src = graph.edge_index[:, 0].long()
    dst = graph.edge_index[:, 1].long()
    mask = graph.mask.bool()

    valid_edge = mask[src] & mask[dst]
    coords = spatial_coords_from_four_vectors(graph.x_raw.unsqueeze(0))[0]
    dist = torch.linalg.norm(coords[src] - coords[dst], dim=-1)

    node_degree = graph.incidence.sum(dim=0)[active]
    edge_size = graph.incidence.sum(dim=1)
    mutual = 0
    edge_set = {(int(s), int(d)) for s, d in zip(src.tolist(), dst.tolist()) if s != d}
    for s, d in edge_set:
        if (d, s) in edge_set:
            mutual += 1

    return {
        "construction": graph.construction,
        "label": graph.label if graph.label is not None else -1,
        "n_particles_active": n,
        "n_edges": graph.n_edges,
        "n_edges_valid": int(valid_edge.sum().item()),
        "edges_per_particle": graph.n_edges / max(n, 1),
        "mean_node_degree": float(node_degree.mean().item()) if n else 0.0,
        "max_node_degree": float(node_degree.max().item()) if n else 0.0,
        "mean_hyperedge_size": float(edge_size.mean().item()) if graph.n_edges else 0.0,
        "max_hyperedge_size": float(edge_size.max().item()) if graph.n_edges else 0.0,
        "mean_spatial_edge_length": float(dist[valid_edge].mean().item()) if valid_edge.any() else 0.0,
        "max_spatial_edge_length": float(dist[valid_edge].max().item()) if valid_edge.any() else 0.0,
        "mutual_directed_pairs": mutual,
        "self_loop_edges": int((src == dst).sum().item()),
        "padding_edges": int((~valid_edge).sum().item()),
    }


def plot_particles(
    graph: JetGraph,
    *,
    plane: Plane = "px-py",
    ax: Any | None = None,
    show_label: bool = True,
    size_scale: float = 300.0,
) -> Any:
    """Scatter plot of real particles; marker size ~ energy fraction."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    x_coord, y_coord = _plane_coords(graph.x_raw, graph.mask, plane)
    z = energy_weights(graph.x_raw, graph.mask).detach().cpu().numpy()[graph.mask.bool().cpu().numpy()]
    colors = np.where(graph.label == 1, "#d62728", "#1f77b4") if graph.label is not None else "#333333"
    ax.scatter(x_coord, y_coord, s=size_scale * z / max(z.max(), 1e-6), c=colors, alpha=0.85, edgecolors="k", linewidths=0.3)
    ax.set_xlabel(plane.split("-")[0])
    ax.set_ylabel(plane.split("-")[1])
    title = f"particles ({plane})"
    if show_label and graph.label is not None:
        title += f" — {'top' if graph.label == 1 else 'QCD'}"
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    return ax


def plot_graph_overlay(
    graph: JetGraph,
    *,
    plane: Plane = "px-py",
    ax: Any | None = None,
    alpha: float = 0.35,
    max_edges: int | None = 500,
) -> Any:
    """Overlay directed edges (or hyperedge hulls) on particle scatter."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    plot_particles(graph, plane=plane, ax=ax, show_label=False)

    active_idx = graph.active_particles()
    idx_map = {int(i): k for k, i in enumerate(active_idx.tolist())}
    x_coord, y_coord = _plane_coords(graph.x_raw, graph.mask, plane)
    pos = np.stack([x_coord, y_coord], axis=1)

    src = graph.edge_index[:, 0].long()
    dst = graph.edge_index[:, 1].long()
    valid = graph.mask[src] & graph.mask[dst]

    if graph.construction.startswith("star"):
        for row in range(graph.n_edges):
            members = graph.incidence[row].nonzero(as_tuple=False).squeeze(-1).tolist()
            if len(members) < 2:
                continue
            poly_xy = np.stack(
                [_plane_coords(graph.x_raw[members], graph.mask[members], plane)[0],
                 _plane_coords(graph.x_raw[members], graph.mask[members], plane)[1]],
                axis=0,
            ).T
            if len(poly_xy) >= 3:
                ax.add_patch(
                    Polygon(
                        poly_xy,
                        closed=True,
                        fill=True,
                        facecolor="#ff7f0e",
                        edgecolor="#ff7f0e",
                        alpha=0.08,
                        linewidth=0.5,
                    )
                )
        ax.set_title(f"{graph.construction} hyperedges ({plane})")
        return ax

    segments = []
    for s, d, ok in zip(src.tolist(), dst.tolist(), valid.tolist()):
        if not ok or s == d:
            continue
        if s not in idx_map or d not in idx_map:
            continue
        segments.append([pos[idx_map[s]], pos[idx_map[d]]])
        if max_edges is not None and len(segments) >= max_edges:
            break

    if segments:
        lc = LineCollection(segments, colors="#444444", linewidths=0.6, alpha=alpha)
        ax.add_collection(lc)
    ax.set_title(f"{graph.construction} directed edges ({plane})")
    ax.autoscale()
    return ax


def plot_incidence_heatmap(graph: JetGraph, *, ax: Any | None = None, max_edges: int = 128) -> Any:
    """Heatmap of incidence matrix rows (edges) x columns (particles)."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    mat = graph.incidence[:max_edges, graph.mask.bool()].detach().cpu().numpy()
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="Blues")
    ax.set_xlabel("active particle index")
    ax.set_ylabel("edge / hyperedge index")
    ax.set_title(f"incidence S — {graph.construction}")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return ax


def plot_degree_distribution(graph: JetGraph, *, ax: Any | None = None) -> Any:
    """Histogram of node degree from incidence column sums."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    degree = graph.incidence.sum(dim=0)[graph.mask.bool()].detach().cpu().numpy()
    ax.hist(degree, bins=min(30, max(int(degree.max()) + 1, 2)), color="#1f77b4", alpha=0.85)
    ax.set_xlabel("node degree (incidence column sum)")
    ax.set_ylabel("count")
    ax.set_title(f"degree distribution — {graph.construction}")
    return ax


def plot_edge_length_distribution(graph: JetGraph, *, ax: Any | None = None) -> Any:
    """Histogram of spatial edge lengths in (px, py, pz)."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    coords = spatial_coords_from_four_vectors(graph.x_raw.unsqueeze(0))[0]
    src, dst = graph.edge_index[:, 0].long(), graph.edge_index[:, 1].long()
    valid = graph.mask[src] & graph.mask[dst]
    dist = torch.linalg.norm(coords[src] - coords[dst], dim=-1)[valid].detach().cpu().numpy()
    ax.hist(dist, bins=40, color="#2ca02c", alpha=0.85)
    ax.set_xlabel("||Δ(p_x,p_y,p_z)||")
    ax.set_ylabel("count")
    ax.set_title(f"spatial edge lengths — {graph.construction}")
    return ax


def plot_edge_feature_histograms(graph: JetGraph, *, ax: Any | None = None) -> Any:
    """Histograms of the four edge feature channels."""
    plt = _require_matplotlib()
    names = ["log-dot", "dpx", "dpy", "dpz"]
    if ax is None:
        _, axes = plt.subplots(1, 4, figsize=(12, 3))
    else:
        axes = [ax]

    feats = graph.edge_x.detach().cpu().numpy()
    for i, name in enumerate(names):
        axes[i].hist(feats[:, i], bins=40, color="#9467bd", alpha=0.85)
        axes[i].set_title(name)
        axes[i].set_xlabel("value")
    axes[0].figure.suptitle(f"edge features — {graph.construction}", y=1.02)
    return axes


def plot_knn_distance_profile(
    x_raw: torch.Tensor,
    mask: torch.Tensor,
    *,
    k: int = 8,
    ax: Any | None = None,
) -> Any:
    """
    For each active particle, plot distance to its r-th nearest neighbor (r=1..k).

    Useful for checking whether k is large enough to connect the jet.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    coords = spatial_coords_from_four_vectors(x_raw.unsqueeze(0))[0]
    invalid = ~mask.bool()
    dist2 = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).pow(2).squeeze(0)
    dist2 = dist2.masked_fill(invalid.unsqueeze(0), float("inf"))
    dist2 = dist2.masked_fill(invalid.unsqueeze(1), float("inf"))
    k_eff = min(k, x_raw.size(0))
    dists = dist2.topk(k_eff, largest=False).values.sqrt()
    active = mask.bool()
    profile = dists[active].detach().cpu().numpy()

    for r in range(k_eff):
        ax.plot(profile[:, r], alpha=0.08, color="#1f77b4")
    ax.plot(np.sort(profile, axis=0), alpha=0.4, color="#1f77b4", linewidth=2, label="sorted profiles")
    ax.set_xlabel("active particle index (sorted curves)")
    ax.set_ylabel("distance to r-th NN")
    ax.set_title(f"kNN distance profile (k={k_eff})")
    ax.legend(loc="upper left")
    return ax


def compare_constructions(
    sample: dict[str, torch.Tensor | int],
    *,
    k: int = 8,
    constructions: tuple[ConstructionName, ...] = ("knn", "star-hypergraph"),
) -> dict[str, dict[str, float | int | str]]:
    """Build multiple constructions on the same jet and return summary stats."""
    return {
        name: summarize_graph(build_jet_graph(sample, k=k, construction=name))
        for name in constructions
    }


def plot_construction_panel(
    sample: dict[str, torch.Tensor | int],
    *,
    k: int = 8,
    plane: Plane = "px-py",
) -> Any:
    """Side-by-side overlay plots for kNN vs star hypergraph on one jet."""
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, construction in zip(axes, ("knn", "star-hypergraph")):
        graph = build_jet_graph(sample, k=k, construction=construction)
        plot_graph_overlay(graph, plane=plane, ax=ax)
    fig.suptitle(f"graph construction comparison (k={k}, {plane})", y=1.02)
    fig.tight_layout()
    return fig


def plot_fidelity_dashboard(
    sample: dict[str, torch.Tensor | int],
    *,
    k: int = 8,
    construction: ConstructionName = "knn",
    plane: Plane = "px-py",
) -> Any:
    """Multi-panel figure: overlay, incidence, degrees, edge lengths, features, kNN profile."""
    plt = _require_matplotlib()
    graph = build_jet_graph(sample, k=k, construction=construction)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3)

    ax0 = fig.add_subplot(gs[0, 0])
    plot_graph_overlay(graph, plane=plane, ax=ax0)

    ax1 = fig.add_subplot(gs[0, 1:])
    plot_incidence_heatmap(graph, ax=ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    plot_degree_distribution(graph, ax=ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    plot_edge_length_distribution(graph, ax=ax3)

    ax4 = fig.add_subplot(gs[1, 2])
    plot_knn_distance_profile(sample["x_raw"], sample["mask"], k=k, ax=ax4)

    ax5 = fig.add_subplot(gs[2, :])
    plot_edge_feature_histograms(graph, ax=ax5)

    stats = summarize_graph(graph)
    fig.suptitle(
        f"{graph.construction} fidelity — "
        f"n_active={stats['n_particles_active']} edges={stats['n_edges']} "
        f"mean_deg={stats['mean_node_degree']:.1f}",
        y=1.01,
    )
    fig.tight_layout()
    return fig, graph, stats
