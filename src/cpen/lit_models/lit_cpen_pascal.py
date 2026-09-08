"""Compatibility shim — prefer ``cpen.apps.pascal.lit_cpen_pascal``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.apps.pascal.lit_cpen_pascal")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
