"""Compatibility shim — prefer ``cpen.apps.jets.graph_plotting``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.apps.jets.graph_plotting")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
