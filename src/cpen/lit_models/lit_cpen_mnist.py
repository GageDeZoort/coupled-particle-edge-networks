"""Compatibility shim — prefer ``cpen.apps.mnist.lit_cpen_mnist``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.apps.mnist.lit_cpen_mnist")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
