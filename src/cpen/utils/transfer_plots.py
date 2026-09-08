"""Compatibility shim — prefer ``cpen.training.transfer_plots``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.training.transfer_plots")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
