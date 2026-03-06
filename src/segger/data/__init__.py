"""Lazy exports for :mod:`segger.data`.

Avoid importing heavy runtime dependencies (GPU stack, polars, etc.) unless the
corresponding symbols are actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .atlas import AtlasReference
    from .data_module import ISTDataModule
    from .writer import ISTSegmentationWriter


__all__ = [
    "ISTDataModule",
    "ISTSegmentationWriter",
    "AtlasReference",
    "fetch_reference",
]


def __getattr__(name: str):
    if name == "ISTDataModule":
        from .data_module import ISTDataModule
        globals()["ISTDataModule"] = ISTDataModule
        return ISTDataModule
    if name == "ISTSegmentationWriter":
        from .writer import ISTSegmentationWriter
        globals()["ISTSegmentationWriter"] = ISTSegmentationWriter
        return ISTSegmentationWriter
    if name in ("AtlasReference", "fetch_reference"):
        from .atlas import AtlasReference, fetch_reference
        globals()["AtlasReference"] = AtlasReference
        globals()["fetch_reference"] = fetch_reference
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
