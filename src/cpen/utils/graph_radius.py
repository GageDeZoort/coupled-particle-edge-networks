"""Compatibility shim — prefer ``cpen.graphs.graph_radius``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.graphs.graph_radius")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
