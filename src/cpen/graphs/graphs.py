"""Graph helpers (wrapper over ``graphs.pyc``).

Adds optional ``min_pt`` filtering for directed kNN edges: keep an edge only if
``max(pT_source, pT_target) >= min_pt``. Spec encoding for caches / CLI:

* ``8-NN`` — plain kNN (no pT cut)
* ``8-NN@pt1`` — k=8, require at least one endpoint with ``pT >= 1`` GeV
* ``8-NN@pt0p5`` — k=8, ``pT >= 0.5`` GeV

Directory tags become ``8nn`` / ``8nn_pt1`` / ``8nn_pt0p5``.
"""

from __future__ import annotations

import contextvars
import importlib.util
import re
import sys
from pathlib import Path

import torch

# Used when callers (legacy cache materialize) only pass ``k``.
_knn_min_pt_ctx: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "cpen_knn_min_pt", default=None
)


def knn_min_pt_context(min_pt: float | None):
    """Context manager / token setter for implicit ``min_pt`` in ``build_knn_graph``."""
    return _knn_min_pt_ctx.set(min_pt if min_pt is not None and min_pt > 0.0 else None)


def reset_knn_min_pt_context(token: contextvars.Token) -> None:
    _knn_min_pt_ctx.reset(token)

_PYC = Path(__file__).resolve().with_name("graphs.pyc")
if not _PYC.is_file():
    raise ImportError(f"Missing bytecode backend {_PYC}")

_spec = importlib.util.spec_from_file_location("_cpen_graphs_bc", _PYC)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_PYC}")
_bc = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_cpen_graphs_bc", _bc)
sys.modules.setdefault("cpen.utils._graphs_bc", _bc)
_spec.loader.exec_module(_bc)

_PUBLIC_MODULE = __name__
for _name, _value in vars(_bc).items():
    if _name.startswith("__"):
        continue
    if isinstance(_value, type) or callable(_value):
        try:
            _value.__module__ = _PUBLIC_MODULE
        except (AttributeError, TypeError):
            pass
    globals()[_name] = _value

_orig_build_knn_graph = _bc.build_knn_graph
_orig_parse_graph_construction = _bc.parse_graph_construction
_orig_graph_construction_tag = _bc.graph_construction_tag

_SPEC_RE = re.compile(
    r"^(\d+)-NN(?:@pt([0-9]+(?:p[0-9]+)?))?$",
    re.IGNORECASE,
)


def format_pt_token(min_pt: float) -> str:
    """Filename / spec token, e.g. ``1`` → ``1``, ``0.5`` → ``0p5``."""
    text = f"{min_pt:g}".replace(".", "p")
    return text


def parse_pt_token(token: str) -> float:
    return float(token.replace("p", "."))


def parse_knn_spec(spec: str) -> tuple[int, float | None]:
    """
    Parse ``8-NN`` or ``8-NN@pt1`` into ``(k, min_pt)``.

    ``min_pt`` is ``None`` when no pT cut is requested.
    """
    match = _SPEC_RE.match(spec.strip())
    if match is None:
        # Fall back to legacy parser for clearer errors on plain specs.
        k = int(_orig_parse_graph_construction(spec))
        return k, None
    k = int(match.group(1))
    if match.group(2) is None:
        return k, None
    return k, parse_pt_token(match.group(2))


def compose_knn_spec(k: int, min_pt: float | None = None) -> str:
    if min_pt is None or min_pt <= 0.0:
        return f"{k}-NN"
    return f"{k}-NN@pt{format_pt_token(min_pt)}"


def parse_graph_construction(spec: str) -> int:
    """Parse graph spec such as ``8-NN`` / ``8-NN@pt1`` into k."""
    k, _ = parse_knn_spec(spec)
    return k


def graph_construction_tag(spec: str) -> str:
    """Directory-safe tag, e.g. ``8-NN@pt1`` → ``8nn_pt1``."""
    k, min_pt = parse_knn_spec(spec)
    tag = f"{k}nn"
    if min_pt is not None and min_pt > 0.0:
        tag = f"{tag}_pt{format_pt_token(min_pt)}"
    return tag


def particle_pt_from_four_vectors(x_raw: torch.Tensor) -> torch.Tensor:
    """Transverse momentum from ``[E, px, py, pz]``."""
    return torch.hypot(x_raw[..., 1], x_raw[..., 2])


def apply_knn_min_pt_filter(
    edge_index: torch.Tensor,
    edge_x: torch.Tensor,
    incidence: torch.Tensor,
    x_raw: torch.Tensor,
    *,
    min_pt: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Zero out directed kNN edges whose endpoints both have ``pT < min_pt``.

    Keeps the fixed ``(B, N*k, ...)`` layout used by the cache (inactive edges
    already have empty incidence / zero ``edge_x``).
    """
    if min_pt <= 0.0:
        return edge_index, edge_x, incidence

    pt = particle_pt_from_four_vectors(x_raw)
    src = edge_index[..., 0].clamp(min=0, max=pt.size(-1) - 1)
    dst = edge_index[..., 1].clamp(min=0, max=pt.size(-1) - 1)
    pt_src = pt.gather(1, src)
    pt_dst = pt.gather(1, dst)
    keep = torch.maximum(pt_src, pt_dst) >= min_pt
    if mask is not None:
        # Never revive padded slots: require the edge already active.
        keep = keep & incidence.bool().any(dim=-1)

    keep_f = keep.to(dtype=edge_x.dtype).unsqueeze(-1)
    edge_x = edge_x * keep_f
    incidence = incidence * keep_f
    return edge_index, edge_x, incidence


def build_knn_graph(
    x_raw: torch.Tensor,
    *,
    k: int,
    mask: torch.Tensor | None = None,
    min_pt: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build directed kNN graphs from raw particle four-vectors.

    Optional ``min_pt`` (GeV) keeps an edge only if at least one endpoint has
    transverse momentum ``>= min_pt``.
    """
    if min_pt is None:
        min_pt = _knn_min_pt_ctx.get()
    edge_index, edge_x, incidence = _orig_build_knn_graph(x_raw, k=k, mask=mask)
    if min_pt is not None and min_pt > 0.0:
        edge_index, edge_x, incidence = apply_knn_min_pt_filter(
            edge_index,
            edge_x,
            incidence,
            x_raw,
            min_pt=float(min_pt),
            mask=mask,
        )
    return edge_index, edge_x, incidence


# Ensure in-module / cross-module callers see the wrapped API.
_bc.build_knn_graph = build_knn_graph
_bc.parse_graph_construction = parse_graph_construction
_bc.graph_construction_tag = graph_construction_tag
globals()["build_knn_graph"] = build_knn_graph
globals()["parse_graph_construction"] = parse_graph_construction
globals()["graph_construction_tag"] = graph_construction_tag
