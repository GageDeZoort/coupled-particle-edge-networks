"""Compatibility shim — prefer ``cpen.graphs.sparse_incidence``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.graphs.sparse_incidence")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
