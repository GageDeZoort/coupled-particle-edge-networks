"""TopTagging kNN graph caches (wrapper over ``graph_cache.pyc``).

Honours ``8-NN@pt1``-style specs from :mod:`cpen.utils.graphs` so pT-filtered
kNN caches land under ``processed/8nn_pt1/n128/`` instead of colliding with
plain ``8nn`` caches.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PYC = Path(__file__).resolve().with_name("graph_cache.pyc")
if not _PYC.is_file():
    raise ImportError(f"Missing bytecode backend {_PYC}")

# Ensure the graphs wrapper is loaded before the bytecode package imports it.
import cpen.utils.graphs as graphs  # noqa: E402

_spec = importlib.util.spec_from_file_location("_cpen_graph_cache_bc", _PYC)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_PYC}")
_bc = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_cpen_graph_cache_bc", _bc)
# Also expose under a package-local alias so spawn workers can resolve pickles
# if anything still points at the private bytecode module name.
sys.modules.setdefault("cpen.utils._graph_cache_bc", _bc)
_spec.loader.exec_module(_bc)

_PUBLIC_MODULE = __name__
for _name, _value in vars(_bc).items():
    if _name.startswith("__"):
        continue
    # DataLoader spawn pickles dataset classes by ``obj.__module__``. The raw
    # pyc lives as ``_cpen_graph_cache_bc``, which is not importable in workers
    # unless we rewrite the module to this public wrapper.
    if isinstance(_value, type) or callable(_value):
        try:
            _value.__module__ = _PUBLIC_MODULE
        except (AttributeError, TypeError):
            pass
    globals()[_name] = _value

# Prefer wrapped graphs symbols on the bytecode module globals (LOAD_GLOBAL).
_bc.build_knn_graph = graphs.build_knn_graph
_bc.parse_graph_construction = graphs.parse_graph_construction
_bc.graph_construction_tag = graphs.graph_construction_tag

_orig_cache_meta = _bc.cache_meta
_orig_build_or_load = _bc.build_or_load_graph_cache
_orig_build_all = _bc.build_all_graph_caches
_orig_missing = getattr(_bc, "missing_graph_cache_message", None)


def cache_meta(
    *,
    graph_construction: str,
    num_particles: int,
    max_jets: int | None,
    seed: int,
):
    """Like the bytecode helper, plus ``knn_min_pt`` when the spec requests it."""
    meta = dict(
        _orig_cache_meta(
            graph_construction=graph_construction,
            num_particles=num_particles,
            max_jets=max_jets,
            seed=seed,
        )
    )
    _k, min_pt = graphs.parse_knn_spec(graph_construction)
    meta["knn_min_pt"] = None if min_pt is None else float(min_pt)
    meta["graph_construction_tag"] = graphs.graph_construction_tag(graph_construction)
    return meta


def build_or_load_graph_cache(*args, **kwargs):
    """Materialize/load a split with the correct ``min_pt`` context for kNN."""
    construction = kwargs.get("graph_construction")
    if construction is None and len(args) >= 3:
        construction = args[2]
    _k, min_pt = graphs.parse_knn_spec(construction) if construction else (None, None)
    token = graphs.knn_min_pt_context(min_pt)
    try:
        return _orig_build_or_load(*args, **kwargs)
    finally:
        graphs.reset_knn_min_pt_context(token)


def build_all_graph_caches(*args, **kwargs):
    construction = kwargs.get("graph_construction")
    if construction is None and len(args) >= 2:
        construction = args[1]
    _k, min_pt = graphs.parse_knn_spec(construction) if construction else (None, None)
    token = graphs.knn_min_pt_context(min_pt)
    try:
        return _orig_build_all(*args, **kwargs)
    finally:
        graphs.reset_knn_min_pt_context(token)


def missing_graph_cache_message(*args, **kwargs):
    if _orig_missing is None:
        return "Graph cache missing."
    msg = _orig_missing(*args, **kwargs)
    return (
        f"{msg}\n"
        "For pT-filtered kNN caches use e.g. --graph-construction 8-NN@pt1 "
        "(or build_toptagging_knn_minpt_graphs.py)."
    )


_bc.cache_meta = cache_meta
_bc.build_or_load_graph_cache = build_or_load_graph_cache
_bc.build_all_graph_caches = build_all_graph_caches
_bc.missing_graph_cache_message = missing_graph_cache_message

globals()["cache_meta"] = cache_meta
globals()["build_or_load_graph_cache"] = build_or_load_graph_cache
globals()["build_all_graph_caches"] = build_all_graph_caches
globals()["missing_graph_cache_message"] = missing_graph_cache_message
globals()["parse_graph_construction"] = graphs.parse_graph_construction
globals()["graph_construction_tag"] = graphs.graph_construction_tag
globals()["build_knn_graph"] = graphs.build_knn_graph
