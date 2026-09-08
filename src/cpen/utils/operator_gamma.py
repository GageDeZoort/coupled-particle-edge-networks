"""Compatibility shim — prefer ``cpen.graphs.operator_gamma``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.graphs.operator_gamma")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
