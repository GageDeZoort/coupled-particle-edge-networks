"""Lightning datamodules for jet benchmarks.

Keep this package import-light: importing ``toptagging_datamodule`` must not
pull JetClass (or other) modules via package ``__init__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cpen.datamodules.base_datamodule import BaseDatamodule

if TYPE_CHECKING:
    from cpen.datamodules.jetclass_datamodule import JetClassLiteStarDatamodule
    from cpen.datamodules.toptagging_datamodule import TopTaggingDatamodule
    from cpen.datamodules.toptagging_star_datamodule import TopTaggingStarDatamodule

__all__ = [
    "BaseDatamodule",
    "TopTaggingDatamodule",
    "TopTaggingStarDatamodule",
    "JetClassLiteStarDatamodule",
    "JetClassDatamodule",
]


def __getattr__(name: str):
    if name in {"TopTaggingDatamodule"}:
        from cpen.datamodules.toptagging_datamodule import TopTaggingDatamodule

        return TopTaggingDatamodule
    if name in {"TopTaggingStarDatamodule"}:
        from cpen.datamodules.toptagging_star_datamodule import TopTaggingStarDatamodule

        return TopTaggingStarDatamodule
    if name in {"JetClassLiteStarDatamodule", "JetClassDatamodule"}:
        from cpen.datamodules.jetclass_datamodule import JetClassLiteStarDatamodule

        return JetClassLiteStarDatamodule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
