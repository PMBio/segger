"""Regression tests for Xenium zip output behavior."""

from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

zarr = pytest.importorskip("zarr")
from zarr.storage import ZipStore

from segger.export.xenium import seg2explorer


def _create_minimal_xenium_source(source_dir):
    source_dir.mkdir(parents=True, exist_ok=True)
    source_cells_zip = source_dir / "cells.zarr.zip"
    source_store = ZipStore(source_cells_zip, mode="w")
    try:
        source_group = zarr.open_group(source_store, mode="w", zarr_format=2)
    except TypeError:
        source_group = zarr.open_group(source_store, mode="w")
    source_group.attrs.update(
        {
            "major_version": 4,
            "minor_version": 0,
            "spatial_units": "microns",
        }
    )
    source_group.store.close()

    experiment = {
        "xenium_explorer_files": {
            "cells_zarr_filepath": "cells.zarr.zip",
            "analysis_zarr_filepath": "analysis.zarr.zip",
            "cell_features_zarr_filepath": "cell_features.zarr.zip",
        }
    }
    (source_dir / "experiment.xenium").write_text(json.dumps(experiment))


def test_seg2explorer_writes_zip_files_not_directories(tmp_path):
    source_dir = tmp_path / "source_xenium"
    _create_minimal_xenium_source(source_dir)

    seg_df = pd.DataFrame(
        {
            "seg_cell_id": [1, 1, 1, 1, 1, 1],
            "x": [0.0, 5.0, 5.0, 0.0, 2.5, 3.0],
            "y": [0.0, 0.0, 5.0, 5.0, 2.0, 3.5],
            "z": [0, 0, 0, 0, 0, 0],
            "cell_compartment": [2, 2, 2, 1, 1, 1],
        }
    )

    out_dir = tmp_path / "export_out"
    seg2explorer(
        seg_df=seg_df,
        source_path=source_dir,
        output_dir=out_dir,
        boundary_method="convex_hull",
    )

    cells_zip = out_dir / "seg_cells.zarr.zip"
    analysis_zip = out_dir / "seg_analysis.zarr.zip"

    assert cells_zip.exists() and cells_zip.is_file() and not cells_zip.is_dir()
    assert analysis_zip.exists() and analysis_zip.is_file() and not analysis_zip.is_dir()
    assert zipfile.is_zipfile(cells_zip)
    assert zipfile.is_zipfile(analysis_zip)

    manifest = json.loads((out_dir / "seg_experiment.xenium").read_text())
    files = manifest["xenium_explorer_files"]
    assert files["cells_zarr_filepath"] == "seg_cells.zarr.zip"
    assert files["analysis_zarr_filepath"] == "seg_analysis.zarr.zip"

