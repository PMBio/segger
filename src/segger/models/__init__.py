"""Public model exports with lazy loading to avoid circular imports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "LitISTEncoder",
    "AlignmentLoss",
    "compute_me_gene_edges",
]


def __getattr__(name: str) -> Any:
    if name == "LitISTEncoder":
        from .lightning_model import LitISTEncoder
        return LitISTEncoder
    if name == "AlignmentLoss":
        from .alignment_loss import AlignmentLoss
        return AlignmentLoss
    if name == "compute_me_gene_edges":
        from .alignment_loss import compute_me_gene_edges
        return compute_me_gene_edges
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
