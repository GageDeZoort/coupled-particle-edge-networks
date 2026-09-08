"""Compatibility shim — prefer ``cpen.training.base_lit_cpen``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.training.base_lit_cpen")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
