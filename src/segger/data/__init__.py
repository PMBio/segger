from .data_module import ISTDataModule
from .writer import ISTSegmentationWriter


def __getattr__(name: str):
    if name in ("AtlasReference", "fetch_reference"):
        from .atlas import AtlasReference, fetch_reference
        globals()["AtlasReference"] = AtlasReference
        globals()["fetch_reference"] = fetch_reference
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")