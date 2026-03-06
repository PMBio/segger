"""Tests for segger.data.atlas — no network calls required.

Uses importlib to load atlas.py directly, bypassing segger.data.__init__
which requires GPU packages (cudf, cupy, cuspatial).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest


def _load_atlas_module() -> ModuleType:
    """Load atlas.py directly via importlib, avoiding heavy __init__ imports."""
    # Ensure segger.utils.optional_deps is available (no GPU deps)
    src_root = Path(__file__).resolve().parents[1] / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    # Pre-import optional_deps so the relative import works
    if "segger" not in sys.modules:
        # Create stub packages so relative imports resolve
        segger_pkg = importlib.import_module("segger")
    if "segger.utils" not in sys.modules:
        importlib.import_module("segger.utils")
    if "segger.utils.optional_deps" not in sys.modules:
        importlib.import_module("segger.utils.optional_deps")

    module_path = src_root / "segger" / "data" / "atlas.py"
    spec = importlib.util.spec_from_file_location(
        "segger.data.atlas",
        module_path,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["segger.data.atlas"] = module
    spec.loader.exec_module(module)
    return module


atlas = _load_atlas_module()


# ---------------------------------------------------------------------------
# Tissue normalization
# ---------------------------------------------------------------------------

class TestNormalizeTissue:
    def test_known_alias(self):
        assert atlas._normalize_tissue("crc") == "large intestine"
        assert atlas._normalize_tissue("colon") == "large intestine"
        assert atlas._normalize_tissue("Colon") == "large intestine"
        assert atlas._normalize_tissue("  CRC  ") == "large intestine"

    def test_unknown_tissue_raises(self):
        with pytest.raises(ValueError, match="Unknown tissue type"):
            atlas._normalize_tissue("some_novel_tissue")

    def test_typo_suggests_match(self):
        with pytest.raises(ValueError, match="Did you mean") as exc_info:
            atlas._normalize_tissue("brest")
        assert "breast" in str(exc_info.value)

    def test_error_shows_valid_list(self):
        with pytest.raises(ValueError, match="Valid tissue types"):
            atlas._normalize_tissue("xyzzy")

    def test_common_tissues(self):
        assert atlas._normalize_tissue("breast") == "breast"
        assert atlas._normalize_tissue("mammary") == "breast"
        assert atlas._normalize_tissue("brain") == "brain"
        assert atlas._normalize_tissue("lung") == "lung"
        assert atlas._normalize_tissue("liver") == "liver"
        assert atlas._normalize_tissue("kidney") == "kidney"
        assert atlas._normalize_tissue("pancreas") == "pancreas"


# ---------------------------------------------------------------------------
# Immune-only detection
# ---------------------------------------------------------------------------

class TestImmuneOnlyGuess:
    def test_immune_only(self):
        immune_types = ["T cell", "B cell", "NK cell", "Monocyte", "Macrophage"]
        assert atlas._immune_only_guess(immune_types) is True

    def test_mixed(self):
        mixed = ["T cell", "B cell", "Epithelial cell", "Fibroblast"]
        assert atlas._immune_only_guess(mixed) is False

    def test_empty(self):
        assert atlas._immune_only_guess([]) is False

    def test_no_immune_keywords(self):
        assert atlas._immune_only_guess(["Epithelial", "Fibroblast"]) is False

    def test_non_immune_only(self):
        # Only non-immune keywords → not immune-only (no immune evidence)
        assert atlas._immune_only_guess(["Neuron", "Astrocyte"]) is False


# ---------------------------------------------------------------------------
# resolve_reference
# ---------------------------------------------------------------------------

class TestResolveReference:
    def test_mutual_exclusion(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            atlas.resolve_reference(
                tissue_type="colon",
                scrna_reference_path=Path("/some/ref.h5ad"),
            )

    def test_neither_provided(self):
        path, col = atlas.resolve_reference(
            tissue_type=None,
            scrna_reference_path=None,
        )
        assert path is None
        assert col == "cell_type"

    def test_explicit_path(self):
        p = Path("/my/reference.h5ad")
        path, col = atlas.resolve_reference(
            tissue_type=None,
            scrna_reference_path=p,
            scrna_celltype_column="my_col",
        )
        assert path == p
        assert col == "my_col"

    def test_tissue_type_calls_fetch(self):
        fake_ref = atlas.AtlasReference(
            h5ad_path=Path("/cache/colon.h5ad"),
            metadata_path=Path("/cache/colon.json"),
            tissue="large intestine",
            organism="homo_sapiens",
            census_version="stable",
            n_obs=1000,
            n_cell_types=10,
            cell_type_column="cell_type",
            immune_only=False,
        )
        with mock.patch.object(atlas, "fetch_reference", return_value=fake_ref):
            path, col = atlas.resolve_reference(
                tissue_type="colon", scrna_reference_path=None
            )
        assert path == Path("/cache/colon.h5ad")
        assert col == "cell_type"


# ---------------------------------------------------------------------------
# Cached reference detection (no Census)
# ---------------------------------------------------------------------------

class TestFetchReferenceCached:
    def test_uses_cache_when_present(self, tmp_path):
        """If h5ad + json already exist, fetch_reference should not call Census."""
        tissue = "breast"
        normalized = atlas._normalize_tissue(tissue)
        safe = normalized.replace(" ", "_")
        tissue_dir = tmp_path / "homo_sapiens" / safe
        tissue_dir.mkdir(parents=True)

        h5ad = tissue_dir / f"{safe}.h5ad"
        h5ad.write_bytes(b"fake h5ad content")

        meta = tissue_dir / f"{safe}.json"
        meta.write_text(json.dumps({
            "h5ad_path": str(h5ad),
            "tissue": normalized,
            "organism": "homo_sapiens",
            "census_version": "stable",
            "n_obs": 500,
            "n_cell_types": 8,
            "cell_type_column": "cell_type",
            "immune_only": False,
        }), encoding="utf-8")

        # Patch _require_census so we know it isn't called
        with mock.patch.object(atlas, "_require_census") as mock_census:
            ref = atlas.fetch_reference(tissue, cache_dir=tmp_path)

        mock_census.assert_not_called()
        assert ref.h5ad_path == h5ad
        assert ref.tissue == normalized
        assert ref.n_obs == 500


# ---------------------------------------------------------------------------
# Missing Census raises ImportError
# ---------------------------------------------------------------------------

class TestFetchReferenceNoCensus:
    def test_import_error_when_not_installed(self, tmp_path):
        """fetch_reference should raise ImportError when Census is missing."""
        with mock.patch(
            "segger.utils.optional_deps.CELLXGENE_CENSUS_AVAILABLE", False
        ):
            with pytest.raises(ImportError, match="cellxgene-census"):
                atlas.fetch_reference("colon", cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# Census opener compatibility
# ---------------------------------------------------------------------------

class TestOpenCensusCompatibility:
    def test_prefers_open_soma_when_available(self):
        events: list[str] = []

        class DummyCtx:
            def __enter__(self):
                events.append("enter")
                return "ctx"

            def __exit__(self, exc_type, exc, tb):
                events.append("exit")
                return False

        class CensusModule:
            def open_soma(self, *, census_version):
                events.append(f"open_soma:{census_version}")
                return DummyCtx()

            def open(self, *, census_version):  # pragma: no cover - should not be used
                events.append(f"open:{census_version}")
                return DummyCtx()

        with atlas._open_census(CensusModule(), census_version="stable") as handle:
            assert handle == "ctx"

        assert events == ["open_soma:stable", "enter", "exit"]

    def test_falls_back_to_open_when_open_soma_missing(self):
        events: list[str] = []

        class DummyCtx:
            def __enter__(self):
                events.append("enter")
                return "ctx"

            def __exit__(self, exc_type, exc, tb):
                events.append("exit")
                return False

        class CensusModule:
            def open(self, *, census_version):
                events.append(f"open:{census_version}")
                return DummyCtx()

        with atlas._open_census(CensusModule(), census_version="latest") as handle:
            assert handle == "ctx"

        assert events == ["open:latest", "enter", "exit"]

    def test_raises_when_no_supported_open_function(self):
        class CensusModule:
            pass

        with pytest.raises(AttributeError, match="open_soma or open"):
            atlas._open_census(CensusModule(), census_version="stable")


# ---------------------------------------------------------------------------
# list_cached_references
# ---------------------------------------------------------------------------

class TestListCachedReferences:
    def test_empty_cache(self, tmp_path):
        refs = atlas.list_cached_references(cache_dir=tmp_path)
        assert refs == []

    def test_finds_references(self, tmp_path):
        for tissue in ("breast", "brain"):
            d = tmp_path / "homo_sapiens" / tissue
            d.mkdir(parents=True)
            (d / f"{tissue}.h5ad").write_bytes(b"fake")
            (d / f"{tissue}.json").write_text(json.dumps({
                "h5ad_path": str(d / f"{tissue}.h5ad"),
                "tissue": tissue,
                "organism": "homo_sapiens",
                "census_version": "stable",
                "n_obs": 100,
                "n_cell_types": 5,
                "cell_type_column": "cell_type",
                "immune_only": False,
            }), encoding="utf-8")

        refs = atlas.list_cached_references(cache_dir=tmp_path)
        assert len(refs) == 2
        tissues = {r.tissue for r in refs}
        assert tissues == {"brain", "breast"}


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------

class TestClearCache:
    def test_clear_specific_tissue(self, tmp_path):
        d = tmp_path / "homo_sapiens" / "breast"
        d.mkdir(parents=True)
        (d / "breast.h5ad").write_bytes(b"fake")

        removed = atlas.clear_cache(tissue="breast", cache_dir=tmp_path)
        assert removed == 1
        assert not d.exists()

    def test_clear_all(self, tmp_path):
        for tissue in ("breast", "brain"):
            d = tmp_path / "homo_sapiens" / tissue
            d.mkdir(parents=True)
            (d / f"{tissue}.h5ad").write_bytes(b"fake")

        removed = atlas.clear_cache(tissue=None, cache_dir=tmp_path)
        assert removed == 2

    def test_clear_nonexistent(self, tmp_path):
        removed = atlas.clear_cache(tissue="nonexistent", cache_dir=tmp_path)
        assert removed == 0


# ---------------------------------------------------------------------------
# AtlasReference dataclass
# ---------------------------------------------------------------------------

class TestAtlasReference:
    def test_frozen(self):
        ref = atlas.AtlasReference(
            h5ad_path=Path("/x.h5ad"),
            metadata_path=Path("/x.json"),
            tissue="breast",
            organism="homo_sapiens",
            census_version="stable",
            n_obs=100,
            n_cell_types=5,
            cell_type_column="cell_type",
            immune_only=False,
        )
        with pytest.raises(AttributeError):
            ref.tissue = "brain"  # type: ignore[misc]
