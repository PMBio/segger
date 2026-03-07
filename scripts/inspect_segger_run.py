#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one Segger benchmark job output (parquet + export h5ad)."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to runs/<job> directory.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional path to exports/<job>. Default: sibling inferred from run dir.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON summary.",
    )
    return parser.parse_args()


def summarize_segmentation(seg_path: Path) -> dict[str, object]:
    if not seg_path.exists():
        return {"exists": False}

    df = pl.read_parquet(seg_path)
    object_type_col = "segger_object_type" if "segger_object_type" in df.columns else None
    fragment_flag_col = "segger_is_fragment" if "segger_is_fragment" in df.columns else None

    assigned = df.filter(pl.col("segger_cell_id").is_not_null()) if "segger_cell_id" in df.columns else pl.DataFrame()
    summary: dict[str, object] = {
        "exists": True,
        "rows": int(df.height),
        "columns": list(df.columns),
    }

    if "segger_cell_id" in df.columns:
        summary["assigned_rows"] = int(assigned.height)
        summary["unassigned_rows"] = int(df.height - assigned.height)
        summary["unique_object_ids"] = int(
            assigned.select(pl.col("segger_cell_id").n_unique()).item() if assigned.height > 0 else 0
        )
        counts = (
            assigned.group_by("segger_cell_id")
            .len()
            .rename({"len": "n"})
            .sort("n", descending=True)
        )
        summary["object_size_stats"] = {
            "min": int(counts["n"].min()) if counts.height else 0,
            "median": float(counts["n"].median()) if counts.height else 0.0,
            "max": int(counts["n"].max()) if counts.height else 0,
        }
        summary["top_objects"] = counts.head(10).to_dicts()

    if object_type_col:
        summary["object_type_counts"] = (
            df.group_by(object_type_col).len().rename({"len": "n"}).to_dicts()
        )
    if fragment_flag_col:
        summary["fragment_flag_counts"] = (
            df.group_by(fragment_flag_col).len().rename({"len": "n"}).to_dicts()
        )
    return summary


def summarize_h5ad(h5ad_path: Path) -> dict[str, object]:
    if not h5ad_path.exists():
        return {"exists": False}

    adata = ad.read_h5ad(h5ad_path)
    summary: dict[str, object] = {
        "exists": True,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns": list(adata.obs.columns),
        "var_preview": [str(v) for v in adata.var_names[:10]],
    }
    for col in ("segger_object_type", "segger_object_group", "segger_is_fragment"):
        if col in adata.obs:
            values = adata.obs[col].astype(str).value_counts(dropna=False)
            summary[f"{col}_counts"] = {str(k): int(v) for k, v in values.to_dict().items()}
    return summary


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    export_dir = args.export_dir.resolve() if args.export_dir else None

    if export_dir is None:
        # Infer exports/<job> from .../runs/<job>
        if run_dir.parent.name != "runs":
            raise SystemExit("Could not infer export dir; pass --export-dir explicitly.")
        export_dir = run_dir.parent.parent / "exports" / run_dir.name

    summary = {
        "run_dir": str(run_dir),
        "export_dir": str(export_dir),
        "segmentation": summarize_segmentation(run_dir / "segger_segmentation.parquet"),
        "anndata": summarize_h5ad(export_dir / "anndata" / "segger_segmentation.h5ad"),
        "anndata_cells_only": summarize_h5ad(export_dir / "anndata" / "segger_segmentation.cells.h5ad"),
        "anndata_fragments_only": summarize_h5ad(export_dir / "anndata" / "segger_segmentation.fragments.h5ad"),
    }

    if args.pretty:
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
