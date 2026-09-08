"""Compatibility shim — prefer ``cpen.graphs.wire_coordinates``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.graphs.wire_coordinates")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
