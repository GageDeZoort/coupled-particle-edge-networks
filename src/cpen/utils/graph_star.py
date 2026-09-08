"""Compatibility shim — prefer ``cpen.graphs.graph_star``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.graphs.graph_star")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
