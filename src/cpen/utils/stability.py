"""Compatibility shim — prefer ``cpen.training.stability``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.training.stability")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
