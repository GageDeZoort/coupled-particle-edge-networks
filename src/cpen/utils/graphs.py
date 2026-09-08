"""Compatibility shim — prefer ``cpen.graphs.graphs``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.graphs.graphs")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
