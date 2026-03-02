from __future__ import annotations

import importlib.util
import numpy as np
import pandas as pd
from anndata import AnnData
from pathlib import Path


def _load_me_genes_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "segger"
        / "validation"
        / "me_genes.py"
    )
    spec = importlib.util.spec_from_file_location("test_me_genes_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_find_mutually_exclusive_genes_drops_markers_shared_across_cell_types() -> None:
    me_genes = _load_me_genes_module()
    cell_types = (["A"] * 10) + (["B"] * 10) + (["C"] * 380)
    obs = pd.DataFrame({"cell_type": cell_types})

    shared = ([1.0] * 10) + ([1.0] * 10) + ([0.0] * 380)
    unique_a = ([1.0] * 10) + ([0.0] * 10) + ([0.0] * 380)
    unique_b = ([0.0] * 10) + ([1.0] * 10) + ([0.0] * 380)

    adata = AnnData(
        X=np.asarray([shared, unique_a, unique_b], dtype=np.float32).T,
        obs=obs,
        var=pd.DataFrame(index=["shared", "unique_a", "unique_b"]),
    )

    markers = {
        "A": {"positive": ["shared", "unique_a"], "negative": []},
        "B": {"positive": ["shared", "unique_b"], "negative": []},
        "C": {"positive": [], "negative": []},
    }

    pairs = me_genes.find_mutually_exclusive_genes(
        adata,
        markers,
        cell_type_column="cell_type",
    )

    assert pairs == [("unique_a", "unique_b")]
