"""Compatibility shim — prefer ``cpen.apps.jets.toptagging``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.apps.jets.toptagging")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
