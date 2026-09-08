"""Load sibling ``callbacks.pyc`` bytecode as this module."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PYC = Path(__file__).resolve().with_name("callbacks.pyc")
if not _PYC.is_file():
    raise ImportError(f"Missing bytecode backend {_PYC}")

_spec = importlib.util.spec_from_file_location("_cpen_callbacks_bc", _PYC)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_PYC}")
_bc = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_cpen_callbacks_bc", _bc)
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

# Regression metrics (QM9) postdate the bytecode parquet logger.
_REGRESSION_METRIC_KEYS = ("train_mae", "val_mae", "test_mae")
_logger = globals().get("ParquetLoggerCallback")
if _logger is not None and hasattr(_logger, "METRIC_KEYS"):
    _logger.METRIC_KEYS = tuple(_logger.METRIC_KEYS) + tuple(
        key for key in _REGRESSION_METRIC_KEYS if key not in _logger.METRIC_KEYS
    )
