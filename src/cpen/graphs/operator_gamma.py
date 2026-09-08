"""Estimate operator scale factors gamma_rs for unnormalized incidence maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn

from cpen.utils.sparse_incidence import (
    BatchedIncidenceCOO,
    apply_t11_sts,
    apply_t11_sts_unnorm,
    apply_t12,
    apply_t12_unnorm,
    apply_t21,
    apply_t21_unnorm,
    apply_t22,
    apply_t22_unnorm,
    compute_incidence_degrees,
    coo_from_batch_tensors,
    coo_from_dense_incidence,
)

ActivationName = Literal["none", "gelu", "relu", "tanh"]


@dataclass(frozen=True)
class OperatorGammas:
    """Scale factors gamma_{rs} with gamma_{rs}^2 ~ E[||T_{rs} H||_F^2 / ||H||_F^2]."""

    gamma_11: float
    gamma_12: float
    gamma_21: float
    gamma_22: float

    @classmethod
    def ones(cls) -> OperatorGammas:
        return cls(1.0, 1.0, 1.0, 1.0)

    def as_dict(self) -> dict[str, float]:
        return {
            "gamma_11": self.gamma_11,
            "gamma_12": self.gamma_12,
            "gamma_21": self.gamma_21,
            "gamma_22": self.gamma_22,
        }

    def to_tensor(self, *, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        return {
            key: torch.tensor(value, device=device, dtype=torch.float32)
            for key, value in self.as_dict().items()
        }


def _activation_fn(name: ActivationName) -> Callable[[torch.Tensor], torch.Tensor] | None:
    if name == "none":
        return None
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unknown activation {name!r}")


def _frobenius_ratio(numerator: torch.Tensor, denominator: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Per-batch-element ||num||_F / ||den||_F, shape (B,)."""
    num = numerator.reshape(numerator.size(0), -1).norm(dim=-1)
    den = denominator.reshape(denominator.size(0), -1).norm(dim=-1).clamp_min(eps)
    return num / den


def _random_hidden(
    shape: tuple[int, ...],
    *,
    width: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    h = torch.randn(*shape, width, generator=generator)
    flat = h.reshape(h.size(0), -1)
    norm = flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    h = h / norm.view(h.size(0), *([1] * (h.dim() - 1)))
    return h


def _resolve_coo(
    incidence: torch.Tensor,
    *,
    incidence_node: torch.Tensor | None = None,
    incidence_edge: torch.Tensor | None = None,
    incidence_nnz: torch.Tensor | None = None,
) -> BatchedIncidenceCOO:
    if incidence_node is not None and incidence_edge is not None and incidence_nnz is not None:
        return coo_from_batch_tensors(
            incidence_node=incidence_node,
            incidence_edge=incidence_edge,
            incidence_nnz=incidence_nnz,
            num_edges=incidence.size(-2),
            num_nodes=incidence.size(-1),
        )
    return coo_from_dense_incidence(incidence)


def estimate_operator_gammas(
    incidence: torch.Tensor,
    *,
    incidence_node: torch.Tensor | None = None,
    incidence_edge: torch.Tensor | None = None,
    incidence_nnz: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    width: int = 64,
    activation: ActivationName = "none",
    eps: float = 1e-12,
    generator: torch.Generator | None = None,
) -> OperatorGammas:
    """Estimate gamma_{11..22} from one incidence batch using sparse operators."""
    if incidence.dim() != 3:
        raise ValueError(f"incidence must be (batch, n_edges, n_particles); got {tuple(incidence.shape)}")

    batch, n_edges, n_particles = incidence.shape
    coo = _resolve_coo(
        incidence,
        incidence_node=incidence_node,
        incidence_edge=incidence_edge,
        incidence_nnz=incidence_nnz,
    )
    act = _activation_fn(activation)

    h_x = _random_hidden((batch, n_particles), width=width, generator=generator)
    h_e = _random_hidden((batch, n_edges), width=width, generator=generator)

    if mask is not None:
        m = mask.to(h_x.dtype).unsqueeze(-1)
        h_x = h_x * m
        edge_active = (incidence.sum(dim=-1) > 0).to(h_x.dtype).unsqueeze(-1)
        h_e = h_e * edge_active

    if act is not None:
        h_x = act(h_x)
        h_e = act(h_e)

    t11 = apply_t11_sts_unnorm(h_x, coo)
    t12 = apply_t12_unnorm(h_x, coo)
    t21 = apply_t21_unnorm(h_e, coo)
    t22 = apply_t22_unnorm(h_e, coo)

    return OperatorGammas(
        gamma_11=float(_frobenius_ratio(t11, h_x, eps=eps).mean().item()),
        gamma_12=float(_frobenius_ratio(t12, h_x, eps=eps).mean().item()),
        gamma_21=float(_frobenius_ratio(t21, h_e, eps=eps).mean().item()),
        gamma_22=float(_frobenius_ratio(t22, h_e, eps=eps).mean().item()),
    )


def compare_degree_vs_unnorm_gammas(
    incidence: torch.Tensor,
    *,
    incidence_node: torch.Tensor | None = None,
    incidence_edge: torch.Tensor | None = None,
    incidence_nnz: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    width: int = 64,
    activation: ActivationName = "none",
    eps: float = 1e-12,
) -> dict[str, float]:
    """Return gamma estimates for unnormalized T and degree-normalized T (diagnostic)."""
    unnorm = estimate_operator_gammas(
        incidence,
        incidence_node=incidence_node,
        incidence_edge=incidence_edge,
        incidence_nnz=incidence_nnz,
        mask=mask,
        width=width,
        activation=activation,
        eps=eps,
    )
    batch, n_edges, n_particles = incidence.shape
    coo = _resolve_coo(
        incidence,
        incidence_node=incidence_node,
        incidence_edge=incidence_edge,
        incidence_nnz=incidence_nnz,
    )
    act = _activation_fn(activation)
    h_x = _random_hidden((batch, n_particles), width=width)
    h_e = _random_hidden((batch, n_edges), width=width)
    if mask is not None:
        m = mask.to(h_x.dtype).unsqueeze(-1)
        h_x = h_x * m
        edge_active = (incidence.sum(dim=-1) > 0).to(h_x.dtype).unsqueeze(-1)
        h_e = h_e * edge_active
    if act is not None:
        h_x = act(h_x)
        h_e = act(h_e)

    d_x_inv, d_e_inv = compute_incidence_degrees(incidence, eps=eps)
    if mask is not None:
        d_x_inv = d_x_inv * mask.to(d_x_inv.dtype)

    t11 = apply_t11_sts(h_x, coo, d_x_inv, d_e_inv)
    t12 = apply_t12(h_x, coo, d_e_inv)
    t21 = apply_t21(h_e, coo, d_x_inv)
    t22 = apply_t22(h_e, coo, d_x_inv, d_e_inv)

    return {
        **{f"unnorm_{k}": v for k, v in unnorm.as_dict().items()},
        "degree_gamma_11": float(_frobenius_ratio(t11, h_x, eps=eps).mean().item()),
        "degree_gamma_12": float(_frobenius_ratio(t12, h_x, eps=eps).mean().item()),
        "degree_gamma_21": float(_frobenius_ratio(t21, h_e, eps=eps).mean().item()),
        "degree_gamma_22": float(_frobenius_ratio(t22, h_e, eps=eps).mean().item()),
    }


def estimate_operator_gammas_from_batches(
    incidence_batches: list[torch.Tensor],
    *,
    incidence_node_batches: list[torch.Tensor | None] | None = None,
    incidence_edge_batches: list[torch.Tensor | None] | None = None,
    incidence_nnz_batches: list[torch.Tensor | None] | None = None,
    mask_batches: list[torch.Tensor | None] | None = None,
    width: int = 64,
    activation: ActivationName = "none",
    seed: int = 0,
) -> OperatorGammas:
    """Average gamma estimates over multiple incidence batches (e.g. training subsample)."""
    if not incidence_batches:
        raise ValueError("incidence_batches must be non-empty")

    n_batches = len(incidence_batches)
    if incidence_node_batches is None:
        incidence_node_batches = [None] * n_batches
    if incidence_edge_batches is None:
        incidence_edge_batches = [None] * n_batches
    if incidence_nnz_batches is None:
        incidence_nnz_batches = [None] * n_batches
    if mask_batches is None:
        mask_batches = [None] * n_batches

    gen = torch.Generator().manual_seed(seed)
    totals = {"gamma_11": 0.0, "gamma_12": 0.0, "gamma_21": 0.0, "gamma_22": 0.0}
    for inc, node, edge, nnz, m in zip(
        incidence_batches,
        incidence_node_batches,
        incidence_edge_batches,
        incidence_nnz_batches,
        mask_batches,
    ):
        est = estimate_operator_gammas(
            inc,
            incidence_node=node,
            incidence_edge=edge,
            incidence_nnz=nnz,
            mask=m,
            width=width,
            activation=activation,
            generator=gen,
        )
        for key, value in est.as_dict().items():
            totals[key] += value

    n = float(n_batches)
    return OperatorGammas(
        gamma_11=totals["gamma_11"] / n,
        gamma_12=totals["gamma_12"] / n,
        gamma_21=totals["gamma_21"] / n,
        gamma_22=totals["gamma_22"] / n,
    )


def estimate_operator_gammas_from_cache(
    *,
    data_root: str,
    split: str = "train",
    radius: float | None = None,
    graph_construction: str | None = None,
    num_particles: int = 100,
    n_jets: int = 512,
    batch_size: int = 64,
    seed: int = 0,
    width: int = 64,
    activation: ActivationName = "none",
) -> OperatorGammas:
    """
    Estimate dataset-level gamma_{rs} from a mmap graph cache subsample.

    Provide either ``radius`` (star-R cache) or ``graph_construction`` (kNN cache).
    """
    import numpy as np

    if (radius is None) == (graph_construction is None):
        raise ValueError("Provide exactly one of radius or graph_construction")

    if radius is not None:
        from cpen.utils.star_graph_cache import open_star_graph_cache

        payload, indices = open_star_graph_cache(
            data_root=data_root,
            split=split,
            radius=radius,
            num_particles=num_particles,
            max_jets=None,
            seed=None,
        )
    else:
        from cpen.utils.graph_cache import open_graph_cache

        payload, indices = open_graph_cache(
            data_root=data_root,
            split=split,
            graph_construction=graph_construction,
            num_particles=num_particles,
            max_jets=None,
            seed=seed,
        )

    n_total = payload["incidence"].size(0)
    rng = np.random.default_rng(seed)
    pool = np.arange(n_total) if indices is None else indices
    pick = rng.choice(pool, size=min(n_jets, len(pool)), replace=False)

    batches: list[torch.Tensor] = []
    node_batches: list[torch.Tensor | None] = []
    edge_batches: list[torch.Tensor | None] = []
    nnz_batches: list[torch.Tensor | None] = []
    mask_batches: list[torch.Tensor] = []
    has_coo = "incidence_node" in payload
    for start in range(0, len(pick), batch_size):
        idx = pick[start : start + batch_size]
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        batches.append(payload["incidence"][idx_t])
        mask_batches.append(payload["mask"][idx_t])
        if has_coo:
            node_batches.append(payload["incidence_node"][idx_t])
            edge_batches.append(payload["incidence_edge"][idx_t])
            nnz_batches.append(payload["incidence_nnz"][idx_t])
        else:
            node_batches.append(None)
            edge_batches.append(None)
            nnz_batches.append(None)

    return estimate_operator_gammas_from_batches(
        batches,
        incidence_node_batches=node_batches,
        incidence_edge_batches=edge_batches,
        incidence_nnz_batches=nnz_batches,
        mask_batches=mask_batches,
        width=width,
        activation=activation,
        seed=seed,
    )
