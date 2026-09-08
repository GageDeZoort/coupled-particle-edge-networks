"""Compatibility shim — prefer ``cpen.graphs.undirected``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.graphs.undirected")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
