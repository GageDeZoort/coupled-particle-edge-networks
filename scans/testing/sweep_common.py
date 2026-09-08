"""Compatibility shim — prefer ``scans.common.sweep_common``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("scans.common.sweep_common")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
