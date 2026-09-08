"""Compatibility shim — prefer ``cpen.apps.jets.lit_cpen_toptagging``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.apps.jets.lit_cpen_toptagging")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
