"""Tests for fragment-aware export helpers."""

from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")
ad = pytest.importorskip("anndata")

from segger.export.anndata_writer import AnnDataWriter
from segger.utils.fragment_outputs import (
    FRAGMENT_FLAG_COLUMN,
    OBJECT_GROUP_COLUMN,
    OBJECT_TYPE_CELL,
    OBJECT_TYPE_COLUMN,
    OBJECT_TYPE_FRAGMENT,
    OBJECT_TYPE_UNASSIGNED,
    with_fragment_annotations,
)


def test_with_fragment_annotations_marks_cells_fragments_and_unassigned():
    frame = pl.DataFrame(
        {
            "segger_cell_id": ["cell-1", "fragment-4", None],
        }
    )

    result = with_fragment_annotations(frame, cell_id_column="segger_cell_id")

    assert result[FRAGMENT_FLAG_COLUMN].to_list() == [False, True, False]
    assert result[OBJECT_TYPE_COLUMN].to_list() == [
        OBJECT_TYPE_CELL,
        OBJECT_TYPE_FRAGMENT,
        OBJECT_TYPE_UNASSIGNED,
    ]
    assert result[OBJECT_GROUP_COLUMN].to_list() == ["cells", "fragments", "unassigned"]


def test_anndata_writer_emits_split_outputs(tmp_path):
    transcripts = pl.DataFrame(
        {
            "row_index": [0, 1, 2, 3, 4],
            "feature_name": ["G1", "G1", "G2", "G2", "G1"],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    predictions = pl.DataFrame(
        {
            "row_index": [0, 1, 2, 3, 4],
            "segger_cell_id": [
                "cell-1",
                "cell-1",
                "fragment-7",
                "fragment-7",
                "cell-2",
            ],
            "segger_similarity": [0.9, 0.8, 0.7, 0.6, 0.95],
        }
    )

    writer = AnnDataWriter(unassigned_marker="-1", compression=None, compression_opts=None)
    output_path = writer.write(
        predictions=predictions,
        output_dir=tmp_path,
        transcripts=transcripts,
    )

    combined = ad.read_h5ad(output_path)
    cells = ad.read_h5ad(tmp_path / "segger_cells.h5ad")
    fragments = ad.read_h5ad(tmp_path / "segger_fragments.h5ad")

    assert combined.n_obs == 3
    assert cells.n_obs == 2
    assert fragments.n_obs == 1

    assert combined.obs.loc["fragment-7", FRAGMENT_FLAG_COLUMN]
    assert not combined.obs.loc["cell-1", FRAGMENT_FLAG_COLUMN]
    assert combined.obs.loc["fragment-7", OBJECT_TYPE_COLUMN] == OBJECT_TYPE_FRAGMENT
    assert combined.obs.loc["cell-1", OBJECT_GROUP_COLUMN] == "cells"

    assert set(cells.obs[OBJECT_TYPE_COLUMN]) == {OBJECT_TYPE_CELL}
    assert set(fragments.obs[OBJECT_TYPE_COLUMN]) == {OBJECT_TYPE_FRAGMENT}
