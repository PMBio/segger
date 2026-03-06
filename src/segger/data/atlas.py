"""Fetch and cache scRNA-seq references from CellxGENE Census.

This module provides a high-level API for downloading tissue-specific
scRNA-seq references, subsampling them, and caching locally so that
Segger's alignment loss and validation metrics can use them without
manual data wrangling.

Usage
-----
>>> from segger.data.atlas import fetch_reference
>>> ref = fetch_reference("colon")
>>> ref.h5ad_path
PosixPath('/Users/.../.segger/references/homo_sapiens/colon/colon.h5ad')

The Census dependency (``cellxgene-census``) is optional.  Functions that
need it will raise ``ImportError`` with installation instructions if it
is missing.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    import anndata as ad

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = Path.home() / ".segger" / "references"

_TISSUE_ALIASES: dict[str, str] = {
    # colorectal
    "crc": "large intestine",
    "colon": "large intestine",
    "colorectal": "large intestine",
    "rectum": "large intestine",
    # breast
    "breast": "breast",
    "mammary": "breast",
    # brain
    "brain": "brain",
    "cerebral cortex": "brain",
    "cerebellum": "brain",
    # lung
    "lung": "lung",
    "pulmonary": "lung",
    # liver
    "liver": "liver",
    "hepatic": "liver",
    # kidney
    "kidney": "kidney",
    "renal": "kidney",
    # pancreas
    "pancreas": "pancreas",
    "pancreatic": "pancreas",
    # skin
    "skin": "skin",
    "dermis": "skin",
    "epidermis": "skin",
    # heart
    "heart": "heart",
    "cardiac": "heart",
    # eye
    "eye": "eye",
    "retina": "eye",
    # stomach
    "stomach": "stomach",
    "gastric": "stomach",
    # prostate
    "prostate": "prostate",
    # bladder
    "bladder": "bladder",
    "urinary bladder": "bladder",
    # blood
    "blood": "blood",
    "pbmc": "blood",
    # bone marrow
    "bone marrow": "bone marrow",
    # lymph node
    "lymph node": "lymph node",
    # spleen
    "spleen": "spleen",
    # thymus
    "thymus": "thymus",
    # small intestine
    "small intestine": "small intestine",
    "ileum": "small intestine",
    "jejunum": "small intestine",
    "duodenum": "small intestine",
}

# All known input names (aliases + canonical Census tissue_general values)
_ALL_KNOWN_NAMES: list[str] = sorted(set(_TISSUE_ALIASES.keys()) | set(_TISSUE_ALIASES.values()))

IMMUNE_KEYWORDS: tuple[str, ...] = (
    "t cell",
    "b cell",
    "nk",
    "monocyte",
    "macrophage",
    "dendritic",
    "neutrophil",
    "mast",
    "plasma",
    "lymphocyte",
)

NON_IMMUNE_KEYWORDS: tuple[str, ...] = (
    "epithelial",
    "endothelial",
    "fibroblast",
    "stromal",
    "hepatocyte",
    "neuron",
    "astrocyte",
    "oligodendrocyte",
    "acinar",
    "ductal",
    "tumor",
    "cancer",
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtlasReference:
    """Metadata for a cached scRNA-seq reference."""

    h5ad_path: Path
    metadata_path: Path
    tissue: str
    organism: str
    census_version: str
    n_obs: int
    n_cell_types: int
    cell_type_column: str  # always "cell_type"
    immune_only: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_cache_dir(cache_dir: Path | None = None) -> Path:
    """Return the cache directory, respecting env-var override."""
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get("SEGGER_REFERENCE_CACHE_DIR")
    if env:
        return Path(env)
    return _DEFAULT_CACHE_DIR


def _normalize_tissue(tissue: str) -> str:
    """Map user-friendly tissue names to Census ``tissue_general`` values.

    Raises ``ValueError`` with fuzzy-match suggestions if the name is not
    recognized.
    """
    key = tissue.strip().lower()
    if key in _TISSUE_ALIASES:
        return _TISSUE_ALIASES[key]

    # Not a known alias — try fuzzy matching
    close = difflib.get_close_matches(key, _ALL_KNOWN_NAMES, n=5, cutoff=0.6)
    valid_tissues = sorted(set(_TISSUE_ALIASES.values()))

    msg = f"Unknown tissue type: '{tissue}'."
    if close:
        suggestions = ", ".join(f"'{s}'" for s in close)
        msg += f"\n  Did you mean: {suggestions}?"
    msg += f"\n  Valid tissue types: {', '.join(valid_tissues)}"
    msg += f"\n  Valid aliases: {', '.join(sorted(_TISSUE_ALIASES.keys()))}"
    raise ValueError(msg)


def _normalize_tissue_lenient(tissue: str) -> str:
    """Like ``_normalize_tissue`` but passes through unknown names instead of raising."""
    key = tissue.strip().lower()
    return _TISSUE_ALIASES.get(key, key)


def _immune_only_guess(cell_types: list[str]) -> bool:
    """Heuristic: True when all cell types look immune-specific."""
    if not cell_types:
        return False
    lowered = [ct.lower() for ct in cell_types]
    has_non_immune = any(
        any(tok in ct for tok in NON_IMMUNE_KEYWORDS) for ct in lowered
    )
    if has_non_immune:
        return False
    has_immune = any(
        any(tok in ct for tok in IMMUNE_KEYWORDS) for ct in lowered
    )
    return has_immune


def _require_census():
    """Import cellxgene_census or raise with install instructions."""
    from segger.utils.optional_deps import require_cellxgene_census
    return require_cellxgene_census()


def _open_census(census_module, census_version: str):
    """Open a Census handle across API variants.

    Recent cellxgene-census releases use ``open_soma``; older versions use
    ``open``. This helper supports both.
    """
    open_soma = getattr(census_module, "open_soma", None)
    if callable(open_soma):
        return open_soma(census_version=census_version)

    open_legacy = getattr(census_module, "open", None)
    if callable(open_legacy):
        return open_legacy(census_version=census_version)

    raise AttributeError(
        "cellxgene_census does not expose a supported opener "
        "(expected open_soma or open)."
    )


@contextmanager
def _progress_spinner(message: str, *, enabled: bool, interval_sec: float = 5.0):
    """Print periodic progress heartbeats while a long call is running."""
    if not enabled:
        yield
        return

    stop_event = threading.Event()
    start = time.monotonic()

    def _worker():
        frames = "|/-\\"
        idx = 0
        while not stop_event.wait(interval_sec):
            elapsed = int(time.monotonic() - start)
            frame = frames[idx % len(frames)]
            idx += 1
            print(f"[atlas] {message} {frame} ({elapsed}s)", flush=True)

    print(f"[atlas] {message} ...", flush=True)
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        yield
        elapsed = int(time.monotonic() - start)
        print(f"[atlas] {message} done ({elapsed}s)", flush=True)
    finally:
        stop_event.set()
        thread.join(timeout=interval_sec + 1.0)


def _get_anndata_compat(
    census_module,
    census_handle,
    *,
    organism: str,
    obs_value_filter: str,
):
    """Call census.get_anndata() across argument-name variants."""
    try:
        return census_module.get_anndata(
            census_handle,
            organism=organism,
            obs_value_filter=obs_value_filter,
            obs_column_names=[
                "cell_type",
                "tissue_general",
                "dataset_id",
                "assay",
                "donor_id",
            ],
            var_column_names=["feature_name"],
        )
    except TypeError:
        return census_module.get_anndata(
            census_handle,
            organism=organism,
            obs_value_filter=obs_value_filter,
            column_names={
                "obs": [
                    "cell_type",
                    "tissue_general",
                    "dataset_id",
                    "assay",
                    "donor_id",
                ],
                "var": ["feature_name"],
            },
        )


def _tissue_dir(cache_dir: Path, organism: str, tissue: str) -> Path:
    """Return ``<cache_dir>/<organism>/<tissue>/``."""
    safe_tissue = tissue.replace(" ", "_")
    safe_organism = organism.replace(" ", "_")
    return cache_dir / safe_organism / safe_tissue


def _h5ad_path(cache_dir: Path, organism: str, tissue: str) -> Path:
    safe_tissue = tissue.replace(" ", "_")
    return _tissue_dir(cache_dir, organism, tissue) / f"{safe_tissue}.h5ad"


def _metadata_path(cache_dir: Path, organism: str, tissue: str) -> Path:
    safe_tissue = tissue.replace(" ", "_")
    return _tissue_dir(cache_dir, organism, tissue) / f"{safe_tissue}.json"


def _load_metadata(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _write_metadata(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _build_reference_from_metadata(meta_path: Path) -> AtlasReference:
    """Reconstruct an ``AtlasReference`` from the JSON sidecar."""
    meta = _load_metadata(meta_path)
    return AtlasReference(
        h5ad_path=Path(meta["h5ad_path"]),
        metadata_path=meta_path,
        tissue=meta["tissue"],
        organism=meta["organism"],
        census_version=meta.get("census_version", "unknown"),
        n_obs=meta["n_obs"],
        n_cell_types=meta["n_cell_types"],
        cell_type_column=meta.get("cell_type_column", "cell_type"),
        immune_only=meta.get("immune_only", False),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_reference(
    tissue: str,
    *,
    cache_dir: Path | None = None,
    organism: str = "homo_sapiens",
    census_version: str = "stable",
    max_cells_per_type: int = 2000,
    min_cell_types: int = 5,
    force: bool = False,
    progress: bool = False,
) -> AtlasReference:
    """Download a tissue-specific scRNA-seq reference from CellxGENE Census.

    Parameters
    ----------
    tissue
        Tissue name (e.g. ``"colon"``, ``"breast"``, ``"brain"``).
        Common aliases are resolved automatically.
    cache_dir
        Local cache directory.  Defaults to ``~/.segger/references``
        or ``$SEGGER_REFERENCE_CACHE_DIR``.
    organism
        Census organism string.
    census_version
        Census release version (``"stable"`` recommended).
    max_cells_per_type
        Maximum cells sampled per cell type (stratified).
    min_cell_types
        Emit a warning if fewer unique cell types are found.
    force
        Re-download even if a cached reference exists.
    progress
        If True, print periodic progress heartbeats during long Census calls.

    Returns
    -------
    AtlasReference
        Metadata about the cached reference, including ``h5ad_path``.
    """
    resolved_cache = _get_cache_dir(cache_dir)
    normalized = _normalize_tissue(tissue)
    h5ad = _h5ad_path(resolved_cache, organism, normalized)
    meta = _metadata_path(resolved_cache, organism, normalized)

    # Use cache if present and not forcing
    if not force and h5ad.exists() and meta.exists():
        if progress:
            print(f"[atlas] Using cached reference: {h5ad}", flush=True)
        return _build_reference_from_metadata(meta)

    census = _require_census()

    # Query Census
    import anndata as _ad
    import numpy as np

    with _progress_spinner("Opening CellxGENE Census", enabled=progress):
        census_handle = _open_census(census, census_version=census_version)

    with census_handle as c:
        # Value filter for Census query
        value_filter = (
            f"tissue_general == '{normalized}' "
            f"and is_primary_data == True "
            f"and disease == 'normal'"
        )

        with _progress_spinner("Querying and downloading reference", enabled=progress):
            adata: _ad.AnnData = _get_anndata_compat(
                census,
                c,
                organism=organism,
                obs_value_filter=value_filter,
            )

    if adata.n_obs == 0:
        raise ValueError(
            f"No cells found in CellxGENE Census for tissue_general='{normalized}'. "
            f"Original query tissue: '{tissue}'. "
            f"Try `segger atlas fetch --help` or check CellxGENE Census documentation."
        )

    # Standardize cell_type column
    adata.obs["cell_type"] = (
        adata.obs["cell_type"].astype("string").fillna("Unknown").astype("category")
    )
    cell_types = sorted(adata.obs["cell_type"].cat.categories.tolist())

    if len(cell_types) < min_cell_types:
        import warnings
        warnings.warn(
            f"Only {len(cell_types)} unique cell types for tissue '{normalized}'; "
            f"expected at least {min_cell_types}.",
            UserWarning,
            stacklevel=2,
        )

    # Stratified subsample
    rng = np.random.default_rng(42)
    keep_indices: list[int] = []
    for ct in cell_types:
        mask = adata.obs["cell_type"] == ct
        indices = np.where(mask)[0]
        if len(indices) <= max_cells_per_type:
            keep_indices.extend(indices.tolist())
        else:
            chosen = rng.choice(indices, size=max_cells_per_type, replace=False)
            keep_indices.extend(chosen.tolist())

    keep_indices.sort()
    adata = adata[keep_indices].copy()

    # Re-categorize after subsample
    adata.obs["cell_type"] = adata.obs["cell_type"].cat.remove_unused_categories()
    final_types = sorted(adata.obs["cell_type"].cat.categories.tolist())
    immune_only = _immune_only_guess(final_types)

    # Atomic write
    h5ad.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False, dir=h5ad.parent, suffix=".tmp.h5ad"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        adata.write_h5ad(tmp_path)
        tmp_path.replace(h5ad)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    # Write metadata sidecar
    metadata = {
        "h5ad_path": str(h5ad),
        "tissue": normalized,
        "organism": organism,
        "census_version": census_version,
        "n_obs": int(adata.n_obs),
        "n_cell_types": len(final_types),
        "cell_type_column": "cell_type",
        "immune_only": immune_only,
        "cell_type_preview": final_types[:30],
        "max_cells_per_type": max_cells_per_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_metadata(meta, metadata)

    return AtlasReference(
        h5ad_path=h5ad,
        metadata_path=meta,
        tissue=normalized,
        organism=organism,
        census_version=census_version,
        n_obs=int(adata.n_obs),
        n_cell_types=len(final_types),
        cell_type_column="cell_type",
        immune_only=immune_only,
    )


def resolve_reference(
    tissue_type: str | None,
    scrna_reference_path: Path | None,
    scrna_celltype_column: str = "cell_type",
    cache_dir: Path | None = None,
) -> tuple[Path | None, str]:
    """Resolve a scRNA reference from either ``--tissue-type`` or ``--scrna-reference-path``.

    Parameters
    ----------
    tissue_type
        Tissue name for Census auto-fetch.
    scrna_reference_path
        Explicit path to a local ``.h5ad`` file.
    scrna_celltype_column
        Cell type column name (used only with ``scrna_reference_path``).
    cache_dir
        Cache directory override.

    Returns
    -------
    tuple[Path | None, str]
        ``(h5ad_path, cell_type_column)`` — both ``None``/default when
        neither argument is provided.

    Raises
    ------
    ValueError
        If both ``tissue_type`` and ``scrna_reference_path`` are provided.
    """
    if tissue_type and scrna_reference_path:
        raise ValueError(
            "Cannot specify both --tissue-type and --scrna-reference-path. "
            "Use one or the other."
        )
    if tissue_type:
        ref = fetch_reference(tissue_type, cache_dir=cache_dir)
        return ref.h5ad_path, ref.cell_type_column
    if scrna_reference_path:
        return Path(scrna_reference_path), scrna_celltype_column
    return None, scrna_celltype_column


def list_cached_references(cache_dir: Path | None = None) -> list[AtlasReference]:
    """List all cached scRNA-seq references.

    Returns
    -------
    list[AtlasReference]
        One entry per cached tissue reference, sorted by tissue name.
    """
    resolved = _get_cache_dir(cache_dir)
    if not resolved.exists():
        return []

    refs: list[AtlasReference] = []
    for meta_path in sorted(resolved.rglob("*.json")):
        try:
            refs.append(_build_reference_from_metadata(meta_path))
        except (KeyError, json.JSONDecodeError):
            continue
    return refs


def clear_cache(
    tissue: str | None = None,
    cache_dir: Path | None = None,
) -> int:
    """Remove cached references.

    Parameters
    ----------
    tissue
        If provided, remove only this tissue's cache.  If ``None``,
        remove all cached references.
    cache_dir
        Cache directory override.

    Returns
    -------
    int
        Number of reference directories removed.
    """
    resolved = _get_cache_dir(cache_dir)
    if not resolved.exists():
        return 0

    if tissue is not None:
        # Use lenient normalization for cache ops (don't error on unknown names)
        normalized = _normalize_tissue_lenient(tissue)
        removed = 0
        # Search across all organism dirs
        for org_dir in resolved.iterdir():
            if not org_dir.is_dir():
                continue
            safe_tissue = normalized.replace(" ", "_")
            tissue_dir = org_dir / safe_tissue
            if tissue_dir.exists():
                shutil.rmtree(tissue_dir)
                removed += 1
        return removed

    # Clear everything
    count = 0
    for org_dir in resolved.iterdir():
        if not org_dir.is_dir():
            continue
        for tissue_dir in org_dir.iterdir():
            if tissue_dir.is_dir():
                shutil.rmtree(tissue_dir)
                count += 1
    return count
