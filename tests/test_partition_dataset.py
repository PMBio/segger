import pytest
import torch
from torch_geometric.data import Data

from segger.data.partition import PartitionDataset


def _toy_graph() -> Data:
    # Include an edge between two nodes that will both be unassigned (-1),
    # which previously triggered torch.bincount() on negative labels.
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )
    return Data(x=torch.randn(4, 2), edge_index=edge_index)


def test_partition_dataset_sanitizes_negative_labels_for_edges():
    data = _toy_graph()
    labels = torch.tensor([0, -1, -1, 1], dtype=torch.long)

    with pytest.warns(RuntimeWarning, match="unassigned nodes"):
        dataset = PartitionDataset(data, labels, clone=True)

    assert dataset.partition.edge_sizes.numel() == 2
    assert torch.all(dataset.partition.edge_sizes >= 0)
    assert dataset.partition.edge_indptr[-1].item() == dataset.data.edge_index.size(1)


def test_partition_dataset_rejects_non_integer_float_labels():
    data = _toy_graph()
    labels = torch.tensor([0.0, 1.0, 1.5, 0.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="integer-valued"):
        PartitionDataset(data, labels, clone=True)
