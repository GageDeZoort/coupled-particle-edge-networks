"""Load sibling ``star_graph_cache.pyc`` bytecode as this module."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PYC = Path(__file__).resolve().with_name("star_graph_cache.pyc")
if not _PYC.is_file():
    raise ImportError(f"Missing bytecode backend {_PYC}")

_spec = importlib.util.spec_from_file_location("_cpen_star_graph_cache_bc", _PYC)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_PYC}")
_bc = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_cpen_star_graph_cache_bc", _bc)
_spec.loader.exec_module(_bc)

_PUBLIC = __name__
for _name, _value in vars(_bc).items():
    if _name.startswith("__"):
        continue
    if isinstance(_value, type) or callable(_value):
        try:
            _value.__module__ = _PUBLIC
        except (AttributeError, TypeError):
            pass
    globals()[_name] = _value
