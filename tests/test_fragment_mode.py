"""Tests for fragment mode connected components."""

import numpy as np
import polars as pl
import pytest

from segger.prediction.fragment import (
    compute_fragment_components,
    apply_fragment_mode,
)


class TestComputeFragmentComponents:
    def test_single_component(self):
        source_ids = np.array([0, 1, 2])
        target_ids = np.array([1, 2, 3])
        similarities = np.array([0.9, 0.8, 0.7])

        components = compute_fragment_components(
            source_ids, target_ids, similarities,
            similarity_threshold=0.5,
            use_gpu=False,
        )

        labels = list(components.values())
        assert len(set(labels)) == 1

    def test_two_components(self):
        source_ids = np.array([0, 1, 10, 11])
        target_ids = np.array([1, 2, 11, 12])
        similarities = np.array([0.9, 0.9, 0.9, 0.9])

        components = compute_fragment_components(
            source_ids, target_ids, similarities,
            similarity_threshold=0.5,
            use_gpu=False,
        )

        labels = list(components.values())
        unique_labels = set(labels)
        assert len(unique_labels) == 2
        assert components[0] == components[1] == components[2]
        assert components[10] == components[11] == components[12]
        assert components[0] != components[10]

    def test_similarity_threshold_filtering(self):
        source_ids = np.array([0, 1, 2])
        target_ids = np.array([1, 2, 3])
        similarities = np.array([0.9, 0.3, 0.9])

        components = compute_fragment_components(
            source_ids, target_ids, similarities,
            similarity_threshold=0.5,
            use_gpu=False,
        )

        labels = list(components.values())
        unique_labels = set(labels)
        assert len(unique_labels) == 2
        assert components[0] == components[1]
        assert components[2] == components[3]
        assert components[0] != components[2]

    def test_empty_edges(self):
        components = compute_fragment_components(
            source_ids=np.array([]),
            target_ids=np.array([]),
            similarities=np.array([]),
            similarity_threshold=0.5,
            use_gpu=False,
        )
        assert components == {}

    def test_all_filtered_edges(self):
        source_ids = np.array([0, 1])
        target_ids = np.array([1, 2])
        similarities = np.array([0.1, 0.2])

        components = compute_fragment_components(
            source_ids, target_ids, similarities,
            similarity_threshold=0.5,
            use_gpu=False,
        )
        assert components == {}

    def test_star_graph(self):
        source_ids = np.array([0, 0, 0, 0])
        target_ids = np.array([1, 2, 3, 4])
        similarities = np.array([0.9, 0.9, 0.9, 0.9])

        components = compute_fragment_components(
            source_ids, target_ids, similarities,
            similarity_threshold=0.5,
            use_gpu=False,
        )

        labels = list(components.values())
        assert len(set(labels)) == 1


class TestApplyFragmentMode:
    @pytest.fixture
    def sample_segmentation(self):
        return pl.DataFrame({
            "row_index": list(range(20)),
            "segger_cell_id": [
                "cell_1", "cell_1", "cell_1",
                "cell_2", "cell_2",
                None, None, None, None, None,
                "cell_3", "cell_3",
                None, None, None, None, None, None,
                "cell_4", "cell_4",
            ],
            "segger_similarity": [0.9] * 20,
        })

    @pytest.fixture
    def sample_edges(self):
        return pl.DataFrame({
            "source": [5, 6, 7, 8, 12, 13, 14, 15, 16],
            "target": [6, 7, 8, 9, 13, 14, 15, 16, 17],
            "similarity": [0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9],
        })

    def test_fragments_assigned(self, sample_segmentation, sample_edges):
        result = apply_fragment_mode(
            segmentation_df=sample_segmentation,
            tx_tx_edges=sample_edges,
            min_transcripts=3,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        fragment_mask = result["segger_cell_id"].str.starts_with("fragment-")
        assert fragment_mask.sum() > 0

    def test_min_transcripts_filter(self, sample_segmentation, sample_edges):
        small_edges = pl.DataFrame({
            "source": [5],
            "target": [6],
            "similarity": [0.9],
        })

        result = apply_fragment_mode(
            segmentation_df=sample_segmentation,
            tx_tx_edges=small_edges,
            min_transcripts=5,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        fragment_mask = result["segger_cell_id"].str.starts_with("fragment-")
        assert fragment_mask.sum() == 0

    def test_assigned_transcripts_unchanged(self, sample_segmentation, sample_edges):
        result = apply_fragment_mode(
            segmentation_df=sample_segmentation,
            tx_tx_edges=sample_edges,
            min_transcripts=3,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        for cell_id in ["cell_1", "cell_2", "cell_3", "cell_4"]:
            original_count = (
                sample_segmentation["segger_cell_id"] == cell_id
            ).sum()
            result_count = (result["segger_cell_id"] == cell_id).sum()
            assert original_count == result_count

    def test_no_unassigned(self):
        segmentation = pl.DataFrame({
            "row_index": [0, 1, 2],
            "segger_cell_id": ["cell_1", "cell_2", "cell_3"],
            "segger_similarity": [0.9, 0.9, 0.9],
        })
        edges = pl.DataFrame({
            "source": [0],
            "target": [1],
            "similarity": [0.9],
        })

        result = apply_fragment_mode(
            segmentation_df=segmentation,
            tx_tx_edges=edges,
            min_transcripts=1,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )
        assert result["segger_cell_id"].to_list() == segmentation["segger_cell_id"].to_list()

    def test_empty_edges(self, sample_segmentation):
        empty_edges = pl.DataFrame({
            "source": [],
            "target": [],
            "similarity": [],
        }).cast({
            "source": pl.Int64,
            "target": pl.Int64,
            "similarity": pl.Float64,
        })

        result = apply_fragment_mode(
            segmentation_df=sample_segmentation,
            tx_tx_edges=empty_edges,
            min_transcripts=3,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )
        assert result.height == sample_segmentation.height

    def test_similarity_threshold_effect(self, sample_segmentation):
        edges = pl.DataFrame({
            "source": [5, 6, 7, 8, 12, 13, 14],
            "target": [6, 7, 8, 9, 13, 14, 15],
            "similarity": [0.3, 0.3, 0.3, 0.3, 0.9, 0.9, 0.9],
        })

        result_low = apply_fragment_mode(
            segmentation_df=sample_segmentation,
            tx_tx_edges=edges,
            min_transcripts=3,
            similarity_threshold=0.2,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )
        result_high = apply_fragment_mode(
            segmentation_df=sample_segmentation,
            tx_tx_edges=edges,
            min_transcripts=3,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        low_fragments = result_low["segger_cell_id"].str.starts_with("fragment-").sum()
        high_fragments = result_high["segger_cell_id"].str.starts_with("fragment-").sum()
        assert low_fragments >= high_fragments


class TestFragmentModeIntegration:
    def test_fragment_ids_are_unique(self):
        segmentation = pl.DataFrame({
            "row_index": list(range(10)),
            "segger_cell_id": [
                "cell_1", "cell_2",
                None, None, None, None, None,
                "cell_3", "cell_4", "cell_5",
            ],
        })
        edges = pl.DataFrame({
            "source": [2, 3, 4, 5],
            "target": [3, 4, 5, 6],
            "similarity": [0.9, 0.9, 0.9, 0.9],
        })

        result = apply_fragment_mode(
            segmentation_df=segmentation,
            tx_tx_edges=edges,
            min_transcripts=3,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        unique_ids = result["segger_cell_id"].unique().drop_nulls().to_list()
        fragment_ids = [id_ for id_ in unique_ids if id_.startswith("fragment-")]
        cell_ids = [id_ for id_ in unique_ids if not id_.startswith("fragment-")]
        assert set(fragment_ids).isdisjoint(set(cell_ids))

    def test_preserves_row_order(self):
        segmentation = pl.DataFrame({
            "row_index": [5, 2, 8, 1, 9],
            "segger_cell_id": [None, "cell_1", None, None, "cell_2"],
        })
        edges = pl.DataFrame({
            "source": [5, 8],
            "target": [8, 1],
            "similarity": [0.9, 0.9],
        })

        result = apply_fragment_mode(
            segmentation_df=segmentation,
            tx_tx_edges=edges,
            min_transcripts=2,
            similarity_threshold=0.5,
            use_gpu=False,
            cell_id_column="segger_cell_id",
            transcript_id_column="row_index",
        )

        assert result["row_index"].to_list() == [5, 2, 8, 1, 9]
