from cyclopts import App, Parameter, Group, validators
from typing import Annotated, Literal
from pathlib import Path

from .registry import ParameterRegistry


# Register defaults and descriptions from files directly
# This is to avoid needing to import all requirements before calling CLI
registry = ParameterRegistry(framework='cyclopts')
base_dir = Path(__file__).parent.parent
to_register = [
    ("data/data_module.py", "ISTDataModule"),
    ("data/writer.py", "ISTSegmentationWriter"),
    ("models/lightning_model.py", "LitISTEncoder"),
]
for file_path, class_name in to_register:
    registry.register_from_file(base_dir / file_path, class_name)


# CLI App
app = App(name="Segger")

# Parameter groups
group_io = Group(
    name="I/O",
    help="Related to file inputs/outputs.",
    sort_key=0,
)
group_nodes = Group(
    name="Node Representation",
    help="Related to transcript and cell node representations.",
    sort_key=2,
)
group_transcripts_graph = Group(
    name="Transcript-Transcript Graph",
    help="Related to transcript-transcript graph parameters.",
    sort_key=3,
)
group_prediction = Group(
    name="Segmentation (Prediction) Graph",
    help="Related to segmentation prediction graph parameters.",
    sort_key=4,
)
group_tiling = Group(
    name="Tiling",
    help="Related to tiling parameters.",
    sort_key=5,
)
group_model = Group(
    name="Model",
    help="Related to model architecture and training parameters.",
    sort_key=6,
)
group_loss = Group(
    name="Loss",
    help="Related to loss function parameters.",
    sort_key=7,
)
group_format = Group(
    name="Input/Output Format",
    help="Related to input/output formats.",
    sort_key=1,
)
group_boundary = Group(
    name="Boundary",
    help="Related to boundary generation and polygon settings.",
    sort_key=8,
)
group_export = Group(
    name="Export",
    help="Related to export parameters.",
    sort_key=9,
)
group_quality = Group(
    name="Quality Filtering",
    help="Related to transcript quality filtering.",
    sort_key=10,
)
group_3d = Group(
    name="3D Support",
    help="Related to 3D coordinate handling.",
    sort_key=11,
)
group_checkpoint = Group(
    name="Checkpoint",
    help="Related to loading checkpoints for prediction-only runs.",
    sort_key=12,
)
group_validation = Group(
    name="Validation",
    help="Related to lightweight validation metrics.",
    sort_key=13,
)


def _resolve_use_3d_flag(use_3d: Literal["auto", "true", "false"]) -> bool | str:
    if use_3d == "auto":
        return "auto"
    return use_3d == "true"

def _load_checkpoint_datamodule_hparams(checkpoint_path: Path) -> dict:
    """Load datamodule kwargs from checkpoint metadata."""
    import torch

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} has unexpected type: "
            f"{type(checkpoint).__name__}"
        )

    datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})
    if not isinstance(datamodule_hparams, dict):
        datamodule_hparams = {}
    return datamodule_hparams

@app.command
def segment(
    # I/O
    input_directory: Annotated[Path, registry.get_parameter(
        "input_directory",
        alias="-i",
        group=group_io,
        validator=validators.Path(exists=True, dir_okay=True),
    )] = registry.get_default("input_directory"),

    output_directory: Annotated[Path, registry.get_parameter(
        "output_directory",
        alias="-o",
        group=group_io,
        validator=validators.Path(dir_okay=True),
    )] = registry.get_default("output_directory"),
    

    # Cell Representation
    node_representation_dim: Annotated[int, Parameter(
        help="Number of dimensions used to represent each node type.",
        validator=validators.Number(gt=0),
        group=group_nodes,
        required=False,
    )] = registry.get_default("cells_embedding_size"),

    cells_representation: Annotated[Literal['pca', 'morphology'], registry.get_parameter(
        "cells_representation_mode",
        group=group_nodes,
    )] = registry.get_default("cells_representation_mode"),

    cells_min_counts: Annotated[int, registry.get_parameter(
        "cells_min_counts",
        validator=validators.Number(gte=0),
        group=group_nodes,
    )] = registry.get_default("cells_min_counts"),

    cells_clusters_n_neighbors: Annotated[int, registry.get_parameter(
        "cells_clusters_n_neighbors",
        validator=validators.Number(gt=0),
        group=group_nodes,
    )] = registry.get_default("cells_clusters_n_neighbors"),

    cells_clusters_resolution: Annotated[float, registry.get_parameter(
        "cells_clusters_resolution",
        validator=validators.Number(gt=0, lte=5),
        group=group_nodes,
    )] = registry.get_default("cells_clusters_resolution"),


    # Gene Representation
    genes_clusters_n_neighbors: Annotated[int, registry.get_parameter(
        "genes_clusters_n_neighbors",
        validator=validators.Number(gt=0),
        group=group_nodes,
    )] = registry.get_default("genes_clusters_n_neighbors"),

    genes_clusters_resolution: Annotated[float, registry.get_parameter(
        "genes_clusters_resolution",
        validator=validators.Number(gt=0, lte=5),
        group=group_nodes,
    )] = registry.get_default("genes_clusters_resolution"),


    # Transcript-Transcript Graph
    transcripts_max_k: Annotated[int, registry.get_parameter(
        "transcripts_graph_max_k",  
        validator=validators.Number(gt=0),
        group=group_transcripts_graph,
    )] = registry.get_default("transcripts_graph_max_k"),

    transcripts_max_dist: Annotated[float, registry.get_parameter(
        "transcripts_graph_max_dist",
        validator=validators.Number(gt=0),
        group=group_transcripts_graph,
    )] = registry.get_default("transcripts_graph_max_dist"),


    # Segmentation (Prediction) Graph
    prediction_mode: Annotated[
        Literal["nucleus", "cell", "uniform"],
        registry.get_parameter(
            "prediction_graph_mode",
            group=group_prediction,
        )
    ] = registry.get_default("prediction_graph_mode"),

    prediction_max_k: Annotated[int | None, registry.get_parameter(
        "prediction_graph_max_k",
        validator=validators.Number(gt=0),
        group=group_prediction,
    )] = registry.get_default("prediction_graph_max_k"),

    prediction_scale_factor: Annotated[float | None, Parameter(
        help="Scale factor for prediction polygons. >1.0 expands, <1.0 shrinks.",
        validator=validators.Number(gt=0),
        group=group_prediction,
    )] = 2.2,

    # Tiling
    tiling_margin_training: Annotated[float, registry.get_parameter(
        "tiling_margin_training",
        validator=validators.Number(gte=0),
        group=group_tiling,
    )] = registry.get_default("tiling_margin_training"),

    tiling_margin_prediction: Annotated[float, registry.get_parameter(
        "tiling_margin_prediction",
        validator=validators.Number(gte=0),
        group=group_tiling,
    )] = registry.get_default("tiling_margin_prediction"),

    max_nodes_per_tile: Annotated[int, registry.get_parameter(
        "tiling_nodes_per_tile",
        validator=validators.Number(gt=0),
        group=group_tiling,
    )] = registry.get_default("tiling_nodes_per_tile"),

    max_edges_per_batch: Annotated[int, registry.get_parameter(
        "edges_per_batch",
        validator=validators.Number(gt=0),
        group=group_tiling,
    )] = registry.get_default("edges_per_batch"),

    # Model
    n_epochs: Annotated[int, Parameter(
        validator=validators.Number(gt=0),
        group=group_model,
        help="Number of training epochs.",
    )] = 20,

    n_mid_layers: Annotated[int, registry.get_parameter(
        "n_mid_layers",
        validator=validators.Number(gte=0),
        group=group_model,
    )] = registry.get_default("n_mid_layers"),

    n_heads: Annotated[int, registry.get_parameter(
        "n_heads",
        validator=validators.Number(gt=0),
        group=group_model,
    )] = registry.get_default("n_heads"),

    hidden_channels: Annotated[int, registry.get_parameter(
        "hidden_channels",
        validator=validators.Number(gt=0),
        group=group_model,
    )] = registry.get_default("hidden_channels"),

    out_channels: Annotated[int, registry.get_parameter(
        "out_channels",
        validator=validators.Number(gt=0),
        group=group_model,
    )] = registry.get_default("out_channels"),

    learning_rate: Annotated[float, registry.get_parameter(
        "learning_rate",
        validator=validators.Number(gt=0),
        group=group_model,
    )] = registry.get_default("learning_rate"),

    use_positional_embeddings: Annotated[bool, registry.get_parameter(
        "use_positional_embeddings",
        group=group_model,
    )] = registry.get_default("use_positional_embeddings"),

    normalize_embeddings: Annotated[bool, registry.get_parameter(
        "normalize_embeddings",
        group=group_model,
    )] = registry.get_default("normalize_embeddings"),

    # Loss
    segmentation_loss: Annotated[
        Literal["triplet", "bce"],
        registry.get_parameter(
            "sg_loss_type",
            group=group_loss,
        )
    ] = registry.get_default("sg_loss_type"),

    transcripts_margin: Annotated[float, registry.get_parameter(
        "tx_margin",
        validator=validators.Number(gt=0),
        group=group_loss,
    )] = registry.get_default("tx_margin"),

    segmentation_margin: Annotated[float, registry.get_parameter(
        "sg_margin",
        validator=validators.Number(gt=0),
        group=group_loss,
    )] = registry.get_default("sg_margin"),

    transcripts_loss_weight_start: Annotated[float, registry.get_parameter(
        "tx_weight_start",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = registry.get_default("tx_weight_start"),

    transcripts_loss_weight_end: Annotated[float, registry.get_parameter(
        "tx_weight_end",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = registry.get_default("tx_weight_end"),

    cells_loss_weight_start: Annotated[float, registry.get_parameter(
        "bd_weight_start",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = registry.get_default("bd_weight_start"),

    cells_loss_weight_end: Annotated[float, registry.get_parameter(
        "bd_weight_end",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = registry.get_default("bd_weight_end"),

    segmentation_loss_weight_start: Annotated[float, registry.get_parameter(
        "sg_weight_start",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = registry.get_default("sg_weight_start"),

    segmentation_loss_weight_end: Annotated[float, registry.get_parameter(
        "sg_weight_end",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = registry.get_default("sg_weight_end"),

    alignment_loss: Annotated[bool, Parameter(
        help="Enable additive alignment loss using ME-gene constraints.",
        group=group_loss,
    )] = False,

    alignment_loss_weight_start: Annotated[float, Parameter(
        help="Starting weight for additive alignment loss.",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = 0.0,

    alignment_loss_weight_end: Annotated[float, Parameter(
        help="Final weight for additive alignment loss.",
        validator=validators.Number(gte=0),
        group=group_loss,
    )] = 0.03,

    alignment_me_gene_pairs_path: Annotated[Path | None, Parameter(
        help="Path to ME-gene pair file (tsv/csv/txt with two columns).",
        group=group_loss,
    )] = None,

    scrna_reference_path: Annotated[Path | None, Parameter(
        help="Path to scRNA reference .h5ad used to derive ME-gene pairs.",
        group=group_loss,
    )] = None,

    scrna_celltype_column: Annotated[str, Parameter(
        help="Cell type column in scRNA reference AnnData (.obs).",
        group=group_loss,
    )] = "cell_type",

    # Quality filtering
    min_qv: Annotated[float | None, Parameter(
        help="Minimum transcript quality threshold. Set to 0 to disable.",
        validator=validators.Number(gte=0),
        group=group_quality,
    )] = 20.0,

    # 3D support
    use_3d: Annotated[
        Literal["auto", "true", "false"],
        Parameter(
            help="Use 3D coordinates for graph construction ('false' default).",
            group=group_3d,
        ),
    ] = "false",
):
    """Run cell segmentation on spatial transcriptomics data."""
    use_3d_value = _resolve_use_3d_flag(use_3d)
    output_directory = Path(output_directory)
    if output_directory.exists() and not output_directory.is_dir():
        raise ValueError(
            f"Output path exists and is not a directory: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    # Remove SLURM environment autodetect
    from lightning.pytorch.plugins.environments import SLURMEnvironment
    SLURMEnvironment.detect = lambda: False

    # Setup Lightning Data Module
    from ..data import ISTDataModule
    datamodule = ISTDataModule(
        input_directory=input_directory,
        cells_representation_mode=cells_representation,
        cells_embedding_size=node_representation_dim,
        cells_min_counts=cells_min_counts,
        cells_clusters_n_neighbors=cells_clusters_n_neighbors,
        cells_clusters_resolution=cells_clusters_resolution,
        genes_clusters_n_neighbors=genes_clusters_n_neighbors,
        genes_clusters_resolution=genes_clusters_resolution,
        transcripts_graph_max_k=transcripts_max_k,
        transcripts_graph_max_dist=transcripts_max_dist,
        prediction_graph_mode=prediction_mode,
        prediction_graph_max_k=prediction_max_k,
        prediction_graph_scale_factor=prediction_scale_factor,
        tiling_margin_training=tiling_margin_training,
        tiling_margin_prediction=tiling_margin_prediction,
        tiling_nodes_per_tile=max_nodes_per_tile,
        edges_per_batch=max_edges_per_batch,
        use_3d=use_3d_value,
        min_qv=min_qv,
        alignment_loss=alignment_loss,
        me_gene_pairs_path=alignment_me_gene_pairs_path,
        scrna_reference_path=scrna_reference_path,
        scrna_celltype_column=scrna_celltype_column,
    )
    
    # Setup Lightning Model
    from ..models import LitISTEncoder
    n_genes = datamodule.ad.shape[1]
    model = LitISTEncoder(
        n_genes=n_genes,
        n_mid_layers=n_mid_layers,
        n_heads=n_heads,
        in_channels=node_representation_dim,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        learning_rate=learning_rate,
        sg_loss_type=segmentation_loss,
        tx_margin=transcripts_margin,
        sg_margin=segmentation_margin,
        tx_weight_start=transcripts_loss_weight_start,
        tx_weight_end=transcripts_loss_weight_end,
        bd_weight_start=cells_loss_weight_start,
        bd_weight_end=cells_loss_weight_end,
        sg_weight_start=segmentation_loss_weight_start,
        sg_weight_end=segmentation_loss_weight_end,
        align_loss=alignment_loss,
        align_weight_start=alignment_loss_weight_start,
        align_weight_end=alignment_loss_weight_end,
        normalize_embeddings=normalize_embeddings,
        use_positional_embeddings=use_positional_embeddings,
    )

    # Setup Lightning Trainer
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from ..data import ISTSegmentationWriter
    from lightning.pytorch import Trainer
    logger = CSVLogger(output_directory)
    writer = ISTSegmentationWriter(output_directory)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_directory / "checkpoints",
        filename="epoch-{epoch:03d}",
        save_top_k=0,
        save_last=True,
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )
    trainer = Trainer(
        logger=logger,
        max_epochs=n_epochs,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[checkpoint_callback, writer],
    )

    # Training
    trainer.fit(model=model, datamodule=datamodule)

    # Prediction: use in-memory model directly to avoid checkpoint reload.
    predictions = trainer.predict(model=model, datamodule=datamodule)

    writer.write_on_epoch_end(
        trainer=trainer,
        pl_module=model,
        predictions=predictions,
        batch_indices=[],
    )


@app.command
def predict(
    checkpoint_path: Annotated[Path, Parameter(
        help="Path to a Segger checkpoint (.ckpt).",
        alias="-c",
        group=group_checkpoint,
        validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
    )],
    input_directory: Annotated[Path, registry.get_parameter(
        "input_directory",
        alias="-i",
        group=group_io,
        validator=validators.Path(exists=True, dir_okay=True),
    )] = registry.get_default("input_directory"),
    output_directory: Annotated[Path, registry.get_parameter(
        "output_directory",
        alias="-o",
        group=group_io,
        validator=validators.Path(dir_okay=True),
    )] = registry.get_default("output_directory"),
    use_3d: Annotated[
        Literal["checkpoint", "auto", "true", "false"],
        Parameter(
            help="Use 3D mode from checkpoint (default) or override it.",
            group=group_3d,
        ),
    ] = "checkpoint",
):
    """Run prediction-only segmentation from a checkpoint."""
    from dataclasses import fields as dataclass_fields
    from lightning.pytorch.plugins.environments import SLURMEnvironment
    from lightning.pytorch import Trainer
    from ..data import ISTDataModule, ISTSegmentationWriter
    from ..models import LitISTEncoder

    output_directory = Path(output_directory)
    if output_directory.exists() and not output_directory.is_dir():
        raise ValueError(
            f"Output path exists and is not a directory: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    SLURMEnvironment.detect = lambda: False

    datamodule_hparams = _load_checkpoint_datamodule_hparams(checkpoint_path)
    datamodule_fields = {field.name for field in dataclass_fields(ISTDataModule)}
    datamodule_kwargs = {
        key: value
        for key, value in datamodule_hparams.items()
        if key in datamodule_fields
    }
    datamodule_kwargs["input_directory"] = input_directory

    if use_3d == "auto":
        datamodule_kwargs["use_3d"] = "auto"
    elif use_3d == "true":
        datamodule_kwargs["use_3d"] = True
    elif use_3d == "false":
        datamodule_kwargs["use_3d"] = False

    datamodule = ISTDataModule(**datamodule_kwargs)
    model = LitISTEncoder.load_from_checkpoint(checkpoint_path, map_location="cpu")

    expected_n_genes_raw = model.hparams.get("n_genes")
    if expected_n_genes_raw is not None:
        expected_n_genes = int(expected_n_genes_raw)
        observed_n_genes = int(datamodule.ad.shape[1])
        if observed_n_genes != expected_n_genes:
            raise ValueError(
                "Checkpoint/data mismatch: "
                f"checkpoint expects n_genes={expected_n_genes}, "
                f"data produced n_genes={observed_n_genes}."
            )

    writer = ISTSegmentationWriter(output_directory)
    trainer = Trainer(
        logger=False,
        callbacks=[writer],
        log_every_n_steps=1,
    )
    predictions = trainer.predict(model=model, datamodule=datamodule)
    writer.write_on_epoch_end(
        trainer=trainer,
        pl_module=model,
        predictions=predictions,
        batch_indices=[],
    )


@app.command
def export(
    segmentation_path: Annotated[Path, Parameter(
        help="Path to segmentation result (.parquet or .csv) file.",
        alias="-s",
        group=group_io,
        validator=validators.Path(exists=True),
    )],
    source_path: Annotated[Path, Parameter(
        help="Path to input data directory (or SpatialData .zarr).",
        alias="-i",
        group=group_io,
        validator=validators.Path(exists=True, dir_okay=True),
    )],
    output_dir: Annotated[Path, Parameter(
        help="Output directory for exported files.",
        alias="-o",
        group=group_io,
    )],
    format: Annotated[
        Literal["xenium_explorer", "xenium", "merged", "spatialdata", "anndata"],
        Parameter(
            help="Export format.",
            group=group_export,
        ),
    ] = "xenium_explorer",
    input_format: Annotated[
        Literal["auto", "raw", "spatialdata"],
        Parameter(
            help="Input data format for loading transcripts.",
            group=group_format,
        ),
    ] = "auto",
    spatialdata_points_key: Annotated[str | None, Parameter(
        help="Optional points key for SpatialData input (auto-detected if omitted).",
        group=group_format,
    )] = None,
    spatialdata_cell_shapes_key: Annotated[str | None, Parameter(
        help="Optional cell-shapes key for SpatialData input (auto-detected if omitted).",
        group=group_format,
    )] = None,
    spatialdata_nucleus_shapes_key: Annotated[str | None, Parameter(
        help="Optional nucleus-shapes key for SpatialData input (auto-detected if omitted).",
        group=group_format,
    )] = None,
    boundary_method: Annotated[
        Literal["input", "convex_hull", "delaunay", "skip"],
        Parameter(
            help="Boundary generation mode for export.",
            group=group_boundary,
        ),
    ] = "input",
    boundary_voxel_size: Annotated[float, Parameter(
        help="Voxel size for boundary downsampling.",
        validator=validators.Number(gte=0),
        group=group_boundary,
    )] = 0.0,
    cell_id_column: Annotated[str, Parameter(
        help="Cell-ID column in segmentation file.",
        group=group_export,
    )] = "segger_cell_id",
    x_column: Annotated[str, Parameter(
        help="X coordinate column.",
        group=group_export,
    )] = "x",
    y_column: Annotated[str, Parameter(
        help="Y coordinate column.",
        group=group_export,
    )] = "y",
    z_column: Annotated[str, Parameter(
        help="Z coordinate column.",
        group=group_export,
    )] = "z",
    area_low: Annotated[float, Parameter(
        help="Minimum allowed cell area.",
        validator=validators.Number(gt=0),
        group=group_boundary,
    )] = 10.0,
    area_high: Annotated[float, Parameter(
        help="Maximum allowed cell area.",
        validator=validators.Number(gt=0),
        group=group_boundary,
    )] = 1500.0,
    num_workers: Annotated[int, Parameter(
        help="Number of workers for polygon generation.",
        validator=validators.Number(gte=0),
        group=group_boundary,
    )] = 1,
    polygon_max_vertices: Annotated[int, Parameter(
        help="Maximum polygon vertices including closure.",
        validator=validators.Number(gt=3),
        group=group_boundary,
    )] = 25,
):
    """Export segmentation results in Xenium/merged/AnnData/SpatialData formats."""
    import polars as pl
    from ..export import seg2explorer, seg2explorer_pqdm
    from ..export.merged_writer import merge_predictions_with_transcripts

    def _is_spatialdata_path(path: Path | str) -> bool:
        try:
            from ..io.spatialdata_loader import is_spatialdata_path as _impl
            return _impl(path)
        except Exception:
            p = Path(path)
            return (
                p.suffix == ".zarr"
                or (p / ".zgroup").exists()
                or (p / "zarr.json").exists()
                or (p / "points").exists()
                or (p / "shapes").exists()
            )

    if area_high <= area_low:
        raise ValueError("area_high must be greater than area_low.")

    # Load segmentation table
    segmentation_from_spatialdata = False
    segmentation_boundaries = None
    if segmentation_path.exists() and _is_spatialdata_path(segmentation_path):
        try:
            from ..io.spatialdata_loader import load_from_spatialdata
        except Exception as exc:
            raise ImportError(
                "SpatialData input requested, but spatialdata support is unavailable. "
                "Install with: pip install segger[spatialdata]"
            ) from exc
        tx, bd = load_from_spatialdata(
            segmentation_path,
            points_key=spatialdata_points_key,
            cell_shapes_key=spatialdata_cell_shapes_key,
            nucleus_shapes_key=spatialdata_nucleus_shapes_key,
            boundary_type="all",
        )
        seg_df = tx.collect() if isinstance(tx, pl.LazyFrame) else tx
        segmentation_from_spatialdata = True
        segmentation_boundaries = bd
    elif segmentation_path.suffix == ".parquet":
        seg_df = pl.read_parquet(segmentation_path)
    elif segmentation_path.suffix in {".csv", ".tsv"}:
        seg_df = pl.read_csv(
            segmentation_path,
            separator="\t" if segmentation_path.suffix == ".tsv" else ",",
        )
    else:
        raise ValueError(
            f"Unsupported segmentation format: {segmentation_path.suffix}. "
            "Expected .parquet, .csv, .tsv, or SpatialData .zarr."
        )

    def _resolve_cell_id_column() -> str:
        if cell_id_column in seg_df.columns:
            return cell_id_column
        aliases = [
            "segger_cell_id",
            "seg_cell_id",
            "cell_id",
            "segmentation_cell_id",
        ]
        for alias in aliases:
            if alias in seg_df.columns:
                print(
                    f"Warning: '{cell_id_column}' not found. "
                    f"Using '{alias}' instead."
                )
                return alias
        raise ValueError(
            "Segmentation file is missing a valid cell-ID column. "
            "Set --cell-id-column explicitly."
        )

    effective_cell_id_column = _resolve_cell_id_column()
    if (
        format not in {"xenium", "xenium_explorer"}
        and effective_cell_id_column != "segger_cell_id"
    ):
        seg_df = seg_df.rename({effective_cell_id_column: "segger_cell_id"})
        effective_cell_id_column = "segger_cell_id"

    def _resolve_transcripts_and_boundaries():
        resolved = input_format
        if resolved == "auto":
            resolved = "spatialdata" if _is_spatialdata_path(source_path) else "raw"

        if resolved == "spatialdata":
            try:
                from ..io.spatialdata_loader import load_from_spatialdata
            except Exception as exc:
                raise ImportError(
                    "SpatialData input requested, but spatialdata support is unavailable. "
                    "Install with: pip install segger[spatialdata]"
                ) from exc
            tx, bd = load_from_spatialdata(
                source_path,
                points_key=spatialdata_points_key,
                cell_shapes_key=spatialdata_cell_shapes_key,
                nucleus_shapes_key=spatialdata_nucleus_shapes_key,
                boundary_type="all",
            )
            return (tx.collect() if isinstance(tx, pl.LazyFrame) else tx), bd

        from ..io import get_preprocessor
        pp = get_preprocessor(source_path)
        tx = pp.transcripts
        if isinstance(tx, pl.LazyFrame):
            tx = tx.collect()
        try:
            bd = pp.boundaries
        except Exception:
            bd = None
        return tx, bd

    if format == "xenium":
        print("Warning: '--format xenium' is deprecated. Use xenium_explorer.")
        format = "xenium_explorer"

    if format == "xenium_explorer":
        if boundary_method == "skip":
            raise ValueError("boundary_method='skip' is not supported for Xenium export.")

        needs_tx = x_column not in seg_df.columns or y_column not in seg_df.columns
        needs_bd = boundary_method == "input"
        tx = None
        bd = segmentation_boundaries
        if not segmentation_from_spatialdata and (needs_tx or needs_bd):
            tx, bd = _resolve_transcripts_and_boundaries()
        if needs_tx and tx is not None:
            seg_df = merge_predictions_with_transcripts(
                predictions=seg_df,
                transcripts=tx,
                cell_id_column=effective_cell_id_column,
            )

        effective_n_jobs = max(num_workers, 1)
        seg_pd = seg_df.to_pandas() if isinstance(seg_df, pl.DataFrame) else seg_df
        use_serial = effective_n_jobs <= 1 or (boundary_method == "input" and bd is not None)

        print(f"Exporting Xenium Explorer output to: {output_dir}")
        if use_serial:
            seg2explorer(
                seg_df=seg_pd,
                source_path=source_path,
                output_dir=output_dir,
                cell_id_column=effective_cell_id_column,
                x_column=x_column,
                y_column=y_column,
                z_column=z_column,
                area_low=area_low,
                area_high=area_high,
                polygon_max_vertices=polygon_max_vertices,
                boundary_method=boundary_method,
                boundary_voxel_size=boundary_voxel_size,
                boundaries=bd,
            )
        else:
            seg2explorer_pqdm(
                seg_df=seg_pd,
                source_path=source_path,
                output_dir=output_dir,
                cell_id_column=effective_cell_id_column,
                x_column=x_column,
                y_column=y_column,
                z_column=z_column,
                area_low=area_low,
                area_high=area_high,
                n_jobs=effective_n_jobs,
                polygon_max_vertices=polygon_max_vertices,
                boundary_method=boundary_method,
                boundary_voxel_size=boundary_voxel_size,
                boundaries=bd,
            )
        print("Export complete.")
        return

    tx, bd = _resolve_transcripts_and_boundaries()

    if format == "merged":
        from ..export import MergedTranscriptsWriter
        writer = MergedTranscriptsWriter()
        output_path = writer.write(
            predictions=seg_df,
            output_dir=output_dir,
            transcripts=tx,
            output_name="transcripts_segmented.parquet",
        )
        print(f"Written merged output: {output_path}")
        return

    if format == "anndata":
        from ..export import AnnDataWriter
        writer = AnnDataWriter()
        output_path = writer.write(
            predictions=seg_df,
            output_dir=output_dir,
            transcripts=tx,
            output_name="segger_segmentation.h5ad",
        )
        print(f"Written AnnData output: {output_path}")
        return

    if format == "spatialdata":
        try:
            from ..export import SpatialDataWriter
        except ImportError:
            print(
                "Warning: spatialdata not installed. "
                "Install with: pip install segger[spatialdata]"
            )
            return
        writer = SpatialDataWriter(
            include_boundaries=(boundary_method != "skip"),
            boundary_method=boundary_method,
            boundary_n_jobs=max(num_workers, 1),
        )
        output_path = writer.write(
            predictions=seg_df,
            output_dir=output_dir,
            transcripts=tx,
            boundaries=bd,
            output_name="segmentation.zarr",
        )
        print(f"Written SpatialData output: {output_path}")
        return

    raise ValueError(f"Unsupported export format: {format}")


@app.command
def validate(
    segmentation_path: Annotated[Path, Parameter(
        help="Path to segger_segmentation.parquet.",
        alias="-s",
        group=group_validation,
        validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
    )],
    anndata_path: Annotated[Path | None, Parameter(
        help="Optional path to segger_segmentation.h5ad for MECR.",
        alias="-a",
        group=group_validation,
        validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
    )] = None,
    source_path: Annotated[Path | None, Parameter(
        help=(
            "Source data directory (raw Xenium/MERSCOPE/CosMX or SpatialData .zarr). "
            "Needed for fast contamination + geometry + doublet metrics."
        ),
        alias="-i",
        group=group_validation,
        validator=validators.Path(exists=True),
    )] = None,
    output_path: Annotated[Path | None, Parameter(
        help=(
            "Output file (.tsv/.csv/.parquet). "
            "Default: <segmentation_dir>/validation_metrics.tsv."
        ),
        alias="-o",
        group=group_validation,
    )] = None,
    me_gene_pairs_path: Annotated[Path | None, Parameter(
        help="Optional path to ME-gene pairs file (two columns).",
        group=group_validation,
        validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
    )] = None,
    scrna_reference_path: Annotated[Path | None, Parameter(
        help="Optional scRNA .h5ad used to discover ME-gene pairs if no pair file is given.",
        group=group_validation,
        validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
    )] = None,
    scrna_celltype_column: Annotated[str, Parameter(
        help="Cell type column in scRNA reference for ME-gene discovery.",
        group=group_validation,
    )] = "cell_type",
    max_me_gene_pairs: Annotated[int, Parameter(
        help="Maximum number of ME-gene pairs sampled for fast MECR computation.",
        validator=validators.Number(gt=0),
        group=group_validation,
    )] = 500,
    border_erosion_fraction: Annotated[float, Parameter(
        help="Fraction of cell bounding box used to define center vs border.",
        validator=validators.Number(gt=0, lt=0.5),
        group=group_validation,
    )] = 0.3,
    border_min_transcripts_per_cell: Annotated[int, Parameter(
        help="Minimum transcripts per cell for border contamination metric.",
        validator=validators.Number(gt=0),
        group=group_validation,
    )] = 20,
    border_max_cells: Annotated[int, Parameter(
        help="Max number of cells sampled per run for border contamination (speed cap).",
        validator=validators.Number(gt=0),
        group=group_validation,
    )] = 3000,
    border_contaminated_enrichment_threshold: Annotated[float, Parameter(
        help="Per-cell border enrichment threshold counted as contaminated (ratio > threshold).",
        validator=validators.Number(gt=1),
        group=group_validation,
    )] = 1.25,
    tco_min_transcripts_per_cell: Annotated[int, Parameter(
        help="Minimum transcripts per cell for transcript-centroid-offset metric.",
        validator=validators.Number(gt=0),
        group=group_validation,
    )] = 20,
    tco_max_cells: Annotated[int, Parameter(
        help="Max number of cells sampled per run for transcript-centroid-offset (speed cap).",
        validator=validators.Number(gt=0),
        group=group_validation,
    )] = 3000,
    signal_min_transcripts_per_cell: Annotated[int, Parameter(
        help="Minimum transcripts per cell for z-based doublet metric.",
        validator=validators.Number(gt=0),
        group=group_validation,
    )] = 20,
    signal_max_cells: Annotated[int, Parameter(
        help="Max number of cells sampled per run for z-based doublet metric (speed cap).",
        validator=validators.Number(gt=0),
        group=group_validation,
    )] = 3000,
    signal_doublet_threshold: Annotated[float, Parameter(
        help="Integrity threshold below which a cell is counted as doublet-like.",
        validator=validators.Number(gt=0, lte=1),
        group=group_validation,
    )] = 0.6,
    random_seed: Annotated[int, Parameter(
        help="Random seed for pair/cell subsampling in fast metrics.",
        group=group_validation,
    )] = 0,
):
    """Compute lightweight validation metrics for Segger outputs.

    Metrics:
    - cells_assigned / cells_total (higher coverage is better)
    - transcripts_assigned_pct (higher is better)
    - mecr_fast (lower is better)
    - border_excess_pct_fast + border_contaminated_cells_pct_fast (lower is better)
    - resolvi_contamination_pct_fast (lower is better)
    - transcript_centroid_offset_fast (higher is better)
    - signal_doublet_like_fraction_fast (lower is better)
    """
    import time
    import polars as pl
    from ..io import StandardTranscriptFields
    from ..validation.quick_metrics import (
        count_cells_from_anndata,
        compute_assignment_metrics,
        compute_border_contamination_fast,
        compute_mecr_fast,
        compute_resolvi_contamination_fast,
        compute_signal_doublet_fast,
        compute_transcript_centroid_offset_fast,
        load_me_gene_pairs,
        load_segmentation,
        load_source_transcripts,
        merge_assigned_transcripts,
    )

    segmentation_path = Path(segmentation_path)

    job = segmentation_path.parent.name
    if output_path is None:
        output_path = segmentation_path.parent / "validation_metrics.tsv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gene_pairs = []
    if me_gene_pairs_path is not None or scrna_reference_path is not None:
        gene_pairs = load_me_gene_pairs(
            me_gene_pairs_path=Path(me_gene_pairs_path) if me_gene_pairs_path is not None else None,
            scrna_reference_path=Path(scrna_reference_path) if scrna_reference_path is not None else None,
            scrna_celltype_column=scrna_celltype_column,
        )

    source_tx = None
    tx_fields = StandardTranscriptFields()
    if source_path is not None:
        source_tx = load_source_transcripts(Path(source_path))

    t0 = time.time()
    row: dict[str, object] = {
        "job": job,
        "segmentation_path": str(segmentation_path),
        "anndata_path": str(anndata_path) if anndata_path is not None else None,
        "cells_total": None,
        "transcripts_assigned_pct_ci95": float("nan"),
        "mecr_ci95_fast": float("nan"),
        "border_contaminated_cells_pct_ci95_fast": float("nan"),
        "resolvi_contamination_ci95_fast": float("nan"),
        "transcript_centroid_offset_ci95_fast": float("nan"),
        "signal_doublet_like_fraction_ci95_fast": float("nan"),
    }

    try:
        seg_df = load_segmentation(segmentation_path)
        row.update(compute_assignment_metrics(seg_df))
        cells_total = count_cells_from_anndata(anndata_path)
        if cells_total is None:
            cells_total = int(row.get("cells_assigned", 0))
        row["cells_total"] = int(cells_total)

        if source_tx is not None:
            assigned_tx = merge_assigned_transcripts(seg_df, source_tx)
            row.update(
                compute_border_contamination_fast(
                    assigned_tx,
                    erosion_fraction=border_erosion_fraction,
                    min_transcripts_per_cell=border_min_transcripts_per_cell,
                    max_cells=border_max_cells,
                    contaminated_enrichment_threshold=border_contaminated_enrichment_threshold,
                    seed=random_seed,
                )
            )
            row.update(
                compute_transcript_centroid_offset_fast(
                    assigned_tx,
                    min_transcripts_per_cell=tco_min_transcripts_per_cell,
                    max_cells=tco_max_cells,
                    seed=random_seed,
                )
            )
            row.update(
                compute_signal_doublet_fast(
                    assigned_tx,
                    z_column=tx_fields.z,
                    min_transcripts_per_cell=signal_min_transcripts_per_cell,
                    max_cells=signal_max_cells,
                    seed=random_seed,
                    doublet_threshold=signal_doublet_threshold,
                )
            )
            row.update(
                compute_resolvi_contamination_fast(
                    assigned_tx,
                    scrna_reference_path=Path(scrna_reference_path) if scrna_reference_path is not None else None,
                    scrna_celltype_column=scrna_celltype_column,
                    feature_column=tx_fields.feature,
                    min_transcripts_per_cell=border_min_transcripts_per_cell,
                    max_cells=border_max_cells,
                    seed=random_seed,
                )
            )
        else:
            row.update(
                {
                    "border_contamination_fast": float("nan"),
                    "border_enrichment_fast": float("nan"),
                    "border_excess_pct_fast": float("nan"),
                    "border_contaminated_cells_pct_fast": float("nan"),
                    "border_contaminated_cells_pct_ci95_fast": float("nan"),
                    "border_metric_cells_used": 0,
                    "resolvi_contamination_pct_fast": float("nan"),
                    "resolvi_contamination_ci95_fast": float("nan"),
                    "resolvi_contaminated_cells_pct_fast": float("nan"),
                    "resolvi_contaminated_cells_pct_ci95_fast": float("nan"),
                    "resolvi_metric_cells_used": 0,
                    "resolvi_shared_genes_used": 0,
                    "resolvi_cell_types_used": 0,
                    "transcript_centroid_offset_fast": float("nan"),
                    "transcript_centroid_offset_ci95_fast": float("nan"),
                    "tco_metric_cells_used": 0,
                    "signal_doublet_like_fraction_fast": float("nan"),
                    "signal_doublet_like_fraction_ci95_fast": float("nan"),
                    "signal_metric_cells_used": 0,
                }
            )

        if anndata_path is not None and Path(anndata_path).exists() and len(gene_pairs) > 0:
            row.update(
                compute_mecr_fast(
                    Path(anndata_path),
                    gene_pairs=gene_pairs,
                    max_pairs=max_me_gene_pairs,
                    soft=True,
                    seed=random_seed,
                )
            )
        else:
            row.update(
                {
                    "mecr_fast": float("nan"),
                    "mecr_ci95_fast": float("nan"),
                    "mecr_pairs_used": 0,
                }
            )

        row["validate_status"] = "ok"
    except Exception as exc:
        row["validate_status"] = "failed"
        row["validate_error"] = str(exc)

    row["elapsed_s"] = round(time.time() - t0, 3)
    result_df = pl.DataFrame([row])
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        result_df.write_parquet(output_path)
    elif suffix == ".csv":
        result_df.write_csv(output_path)
    else:
        if suffix not in {".tsv", ""}:
            print(f"[validate] Unknown extension '{suffix}', writing TSV.")
        result_df.write_csv(output_path, separator="\t")

    print(f"[validate] {job}: {row['validate_status']}")
    print(f"[validate] Wrote metrics to: {output_path}")


# Plotting parameter group
group_plot = Group(
    name="Plotting",
    help="Related to plotting loss curves from training logs.",
    sort_key=12,
)


def _resolve_metrics_path(
    output_directory: Path,
    log_version: int | None,
) -> Path:
    output_directory = Path(output_directory)
    direct_candidate = output_directory / "metrics.csv"
    if direct_candidate.exists():
        return direct_candidate

    logs_dir = output_directory / "lightning_logs"
    if logs_dir.exists():
        if log_version is not None:
            candidate = logs_dir / f"version_{log_version}" / "metrics.csv"
            if candidate.exists():
                return candidate
            available_versions = sorted(
                [
                    p.name.replace("version_", "")
                    for p in logs_dir.iterdir()
                    if p.is_dir() and p.name.startswith("version_")
                ]
            )
            hint = (
                f" Available versions: {', '.join(available_versions)}"
                if available_versions
                else ""
            )
            raise SystemExit(f"metrics.csv not found for version_{log_version}.{hint}")

        version_dirs = [
            p for p in logs_dir.iterdir() if p.is_dir() and p.name.startswith("version_")
        ]
        parsed_versions = []
        for vdir in version_dirs:
            suffix = vdir.name.replace("version_", "")
            try:
                parsed_versions.append((int(suffix), vdir))
            except ValueError:
                continue
        if parsed_versions:
            _, latest_dir = max(parsed_versions, key=lambda item: item[0])
            candidate = latest_dir / "metrics.csv"
            if candidate.exists():
                return candidate

        candidates = sorted(logs_dir.rglob("metrics.csv"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]

    candidates = sorted(output_directory.rglob("metrics.csv"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]

    raise SystemExit(f"No metrics.csv found under: {output_directory}")


@app.command
def plot(
    output_directory: Annotated[Path, Parameter(
        help="Segger output directory containing lightning_logs/.../metrics.csv.",
        alias="-o",
        group=group_io,
        validator=validators.Path(exists=True, dir_okay=True),
    )],
    log_version: Annotated[int | None, Parameter(
        alias="-v",
        help=(
            "Lightning log version to use (e.g. 3 for lightning_logs/version_3). "
            "Defaults to the latest version. Use --log-version (not --version, "
            "which is reserved for the Segger app version)."
        ),
        group=group_plot,
    )] = None,
    quick: Annotated[bool, Parameter(
        help="Plot directly in the terminal using uniplot (no image saved).",
        group=group_plot,
    )] = False,
):
    """Plot loss curves from training metrics.csv."""
    output_directory = Path(output_directory)
    if output_directory.is_file():
        raise SystemExit(
            "--output-directory should point to the segmentation output directory, not metrics.csv."
        )

    metrics_csv = _resolve_metrics_path(output_directory, log_version)

    import pandas as pd

    df = pd.read_csv(metrics_csv)
    x_axis = "step"
    if x_axis not in df.columns:
        raise SystemExit(
            "metrics.csv is missing the 'step' column required for plotting."
        )

    numeric_cols = [col for col in df.select_dtypes(include="number").columns]
    metric_columns = [col for col in numeric_cols if col not in ("epoch", "step")]
    if not metric_columns:
        raise SystemExit("No numeric metric columns found in metrics.csv.")

    def _smooth_values(values):
        count = len(values)
        if count < 3:
            return values
        window = max(5, min(25, count // 20))
        return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()

    def _series_for_column(column: str):
        series = df[[x_axis, column]].dropna()
        if series.empty:
            return None, None
        series = series.sort_values(x_axis)
        x_vals = series[x_axis].to_numpy()
        y_vals = series[column].to_numpy()
        y_vals = _smooth_values(y_vals)
        return x_vals, y_vals

    grouped_metrics: dict[str, list[tuple[str, str]]] = {}
    for column in metric_columns:
        if ":" in column:
            split, base = column.split(":", 1)
        else:
            split, base = "", column
        grouped_metrics.setdefault(base, []).append((split, column))

    metrics_data: list[tuple[str, list[tuple[str, str, list[float], list[float]]]]] = []
    for base in sorted(grouped_metrics.keys()):
        series_entries = []
        for split, column in grouped_metrics[base]:
            x_vals, y_vals = _series_for_column(column)
            if x_vals is None:
                continue
            label = split if split else column
            series_entries.append((label, column, x_vals, y_vals))
        if series_entries:
            metrics_data.append((base, series_entries))

    if not metrics_data:
        raise SystemExit("No non-empty loss curves found in metrics.csv.")

    if quick:
        try:
            from uniplot import plot as uniplot_plot
        except ImportError as exc:
            raise SystemExit(
                "uniplot is not installed. Install with: pip install segger[plot]"
            ) from exc

        plots_per_page = 4
        total_pages = (len(metrics_data) + plots_per_page - 1) // plots_per_page
        for page_idx in range(total_pages):
            start = page_idx * plots_per_page
            end = start + plots_per_page
            page_metrics = metrics_data[start:end]
            print(f"[segger] Loss curves (page {page_idx + 1}/{total_pages})")
            for base, series_entries in page_metrics:
                xs = [entry[2] for entry in series_entries]
                ys = [entry[3] for entry in series_entries]
                labels = [entry[0] for entry in series_entries]
                uniplot_plot(
                    xs=xs,
                    ys=ys,
                    legend_labels=labels if len(labels) > 1 else None,
                    color=len(labels) > 1,
                    lines=True,
                    title=base,
                )
                print("")
        print(f"Using metrics: {metrics_csv}")
        print("Quick plot only (no image saved).")
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is not installed. Install with: pip install segger[plot]"
        ) from exc

    colors = plt.cm.tab10.colors
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

    plots_per_page = 4
    total_pages = (len(metrics_data) + plots_per_page - 1) // plots_per_page

    saved_paths = []
    for page_idx in range(total_pages):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()
        start = page_idx * plots_per_page
        end = start + plots_per_page
        page_metrics = metrics_data[start:end]

        for ax_idx, ax in enumerate(axes):
            if ax_idx >= len(page_metrics):
                ax.axis("off")
                continue
            base, series_entries = page_metrics[ax_idx]
            for label, column, x_vals, y_vals in series_entries:
                split = column.split(":", 1)[0] if ":" in column else ""
                linestyle = "--" if split == "val" else "-"
                ax.plot(
                    x_vals,
                    y_vals,
                    label=label,
                    linestyle=linestyle,
                    linewidth=1.6,
                )
            ax.set_title(base)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        for ax in axes[-2:]:
            ax.set_xlabel(x_axis)
        axes[0].set_ylabel("loss")
        axes[2].set_ylabel("loss")

        fig.suptitle("Loss curves")
        fig.tight_layout()

        if page_idx == 0:
            output_path = output_directory / "loss_curves.png"
        else:
            output_path = output_directory / f"loss_curves_{page_idx + 1}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        saved_paths.append(output_path)

    print(f"Using metrics: {metrics_csv}")
    for path in saved_paths:
        print(f"Saved plot to: {path}")
