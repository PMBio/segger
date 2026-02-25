from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import negative_sampling
from lightning.pytorch import LightningDataModule
from torchvision.transforms import Compose
from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path
import polars as pl
import torch
import gc
import os
import numpy as np

from .tile_dataset import (
    TileFitDataset,
    TilePredictDataset, 
    DynamicBatchSamplerPatch
)
from ..io import (
    StandardTranscriptFields,
    StandardBoundaryFields, 
    get_preprocessor
)
from .utils import setup_anndata, setup_heterodata
from .tiling import QuadTreeTiling, SquareTiling
from .partition import PartitionSampler


def _load_me_gene_pairs_file(path: Path) -> list[tuple[str, str]]:
    """Load mutually-exclusive gene pairs from a text/csv/tsv file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ME-gene pair file not found: {path}")

    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                parts = [p.strip() for p in line.split("\t")]
            elif "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = [p.strip() for p in line.split()]
            if len(parts) < 2:
                continue
            if parts[0].lower() in {"gene1", "gene_a"} and parts[1].lower() in {"gene2", "gene_b"}:
                continue
            pairs.append((parts[0], parts[1]))

    if len(pairs) == 0:
        raise ValueError(
            f"No valid ME-gene pairs parsed from: {path}. "
            "Expected two columns per row (gene1,gene2)."
        )
    return pairs



class NegativeSampling(BaseTransform):
    #TODO: Add documentation
    def __init__(
        self,
        edge_type: tuple[str],
        sampling_ratio: float,
        pos_index: str = 'edge_index',
        neg_index: str = 'neg_edge_index',
    ):
        #TODO: Add documentation
        super().__init__()
        self.edge_type = edge_type
        self.pos_index = pos_index
        self.neg_index = neg_index
        self.sampling_ratio = sampling_ratio

    def forward(self, data):
        # Return early if no positive edges
        pos_idx = data[self.edge_type][self.pos_index]
        if pos_idx.size(1) == 0:
            data[self.edge_type][self.neg_index] = pos_idx.clone()
            return data
        # Construct negative index with mapped transcript indices
        val, key = torch.unique(pos_idx[0], return_inverse=True)
        pos_idx[0] = key
        neg_idx = negative_sampling(
            pos_idx,
            pos_idx.max(1).values + 1,
            num_neg_samples=int(pos_idx.shape[1] * self.sampling_ratio),
        )
        # Reset transcript indices
        pos_idx[0] = val
        neg_idx[0] = val[neg_idx[0]]
        data[self.edge_type][self.neg_index] = neg_idx

        return data
    

@dataclass
class ISTDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for preparing and loading spatial 
    transcriptomics data in IST format.

    This class handles preprocessing, graph construction, tiling, and
    DataLoader creation for training, validation, and prediction phases
    of the Segger model. It standardizes transcript, boundary, and
    embedding data into graph-compatible datasets with configurable
    clustering, tiling, and sampling parameters.

    Parameters
    ----------
    input_directory : Path
        Path to the standardized IST dataset directory.
    num_workers : int, default=8
        Number of workers for DataLoader processes.
    cells_representation_mode : {"pca", "morphology"}, default="pca"
        Feature representation used for cell embeddings.
    cells_embedding_size : int or None, default=128
        Dimensionality of the cell embedding space.
    cells_min_counts : int, default=10
        Minimum transcript count threshold per cell.
    cells_clusters_n_neighbors : int, default=10
        Number of neighbors for cell clustering.
    cells_clusters_resolution : float, default=3.0
        Resolution parameter for cell clustering.
    genes_clusters_n_neighbors : int, default=5
        Number of neighbors for gene clustering.
    genes_clusters_resolution : float, default=3.0
        Resolution parameter for gene clustering.
    transcripts_graph_max_k : int, default=5
        Maximum number of edges per transcript in the local graph.
    transcripts_graph_max_dist : float, default=5.0
        Maximum edge distance for transcript graph construction.
    segmentation_graph_mode : {"nucleus", "cell"}, default="nucleus"
        Type of segmentation boundaries used for graph edges.
    segmentation_graph_negative_edge_rate : float, default=1.0
        Ratio of negative to positive edges in the segmentation graph.
    prediction_graph_mode : {"nucleus", "cell", "uniform"}, default="cell"
        Graph structure used during prediction.
    prediction_graph_max_k : int, default=3
        Maximum number of edges per transcript for prediction graphs.
    prediction_graph_scale_factor : float, default=1.2
        Multiplicative scale factor for boundary polygons in prediction graph
        construction. Values > 1 expand, values < 1 shrink.
    tiling_mode : {"adaptive", "square"}, default="adaptive"
        Strategy for spatial graph tiling (adaptive quadtree or grid).
    tiling_margin_training : float, default=20.0
        Margin width (in µm) added to tiles during training.
    tiling_margin_prediction : float, default=20.0
        Margin width (in µm) added to tiles during prediction.
    tiling_nodes_per_tile : int, default=50000
        Maximum number of nodes per tile for adaptive tiling.
    tiling_side_length : float, default=250.0
        Side length of square tiles (benchmarking only).
    training_fraction : float, default=0.75
        Fraction of tiles used for training; the rest for validation.
    edges_per_batch : int, default=1_000_000
        Maximum number of edges per batch in the DataLoader.
    """
    input_directory: Path
    num_workers: int = 8
    cells_representation_mode: Literal["pca", "morphology"] = "pca"
    cells_embedding_size: int | None = 128
    cells_min_counts: int = 10
    cells_clusters_n_neighbors: int = 10
    cells_clusters_resolution: float = 2.
    genes_min_counts: int = 100
    genes_clusters_n_neighbors: int = 5
    genes_clusters_resolution: float = 2.
    transcripts_graph_max_k: int = 5
    transcripts_graph_max_dist: float = 5.
    segmentation_graph_mode: Literal["nucleus", "cell"] = "nucleus"
    segmentation_graph_negative_edge_rate: float = 1.
    prediction_graph_mode: Literal["nucleus", "cell", "uniform"] = "cell"
    prediction_graph_max_k: int = 3
    prediction_graph_scale_factor: float = 1.2
    use_3d: bool | Literal["auto"] = False
    min_qv: Optional[float] = 20.0
    alignment_loss: bool = False
    me_gene_pairs_path: Optional[Path] = None
    scrna_reference_path: Optional[Path] = None
    scrna_celltype_column: str = "cell_type"
    me_gene_pairs: Optional[list[tuple[str, str]]] = None
    tiling_mode: Literal["adaptive", "square"] = "adaptive"  # TODO: Remove (benchmarking only)
    tiling_margin_training: float = 20.
    tiling_margin_prediction: float = 20.
    tiling_nodes_per_tile: int = 50_000
    tiling_side_length: float = 250.  # TODO: Remove (benchmarking only)
    training_fraction: float = 0.75
    edges_per_batch: int = 1_000_000
    
    def __post_init__(self):
        """TODO: Description
        """
        env_num_workers = os.getenv("SEGGER_NUM_WORKERS")
        if env_num_workers is not None and env_num_workers != "":
            try:
                self.num_workers = max(0, int(env_num_workers))
            except ValueError as exc:
                raise ValueError(
                    "Invalid SEGGER_NUM_WORKERS value: "
                    f"{env_num_workers!r}. Expected a non-negative integer."
                ) from exc

        super().__init__()
        self.save_hyperparameters()
        self.load()

    def load(self):
        """TODO: Description
        """
        # Load and prepare shared objects
        tx_fields = StandardTranscriptFields()
        bd_fields = StandardBoundaryFields()

        if self.alignment_loss:
            if self.me_gene_pairs is None:
                if self.me_gene_pairs_path is not None:
                    self.me_gene_pairs = _load_me_gene_pairs_file(self.me_gene_pairs_path)
                elif self.scrna_reference_path is not None:
                    from ..validation.me_genes import load_me_genes_from_scrna

                    self.me_gene_pairs, _ = load_me_genes_from_scrna(
                        scrna_path=Path(self.scrna_reference_path),
                        cell_type_column=self.scrna_celltype_column,
                    )
                else:
                    raise ValueError(
                        "alignment_loss=True requires one of: "
                        "--alignment-me-gene-pairs-path, "
                        "--scrna-reference-path, or precomputed me_gene_pairs."
                    )
            self.me_gene_pairs = [
                (str(g1), str(g2))
                for g1, g2 in self.me_gene_pairs
            ]
            if len(self.me_gene_pairs) == 0:
                raise ValueError("alignment_loss=True but me_gene_pairs is empty.")
        else:
            self.me_gene_pairs = None

        # Ensure checkpoint hparams include resolved pairs for predict-only recovery.
        if hasattr(self, "hparams"):
            self.hparams.me_gene_pairs = self.me_gene_pairs

        # Load standardized IST data (raw platform directory or SpatialData .zarr)
        input_path = Path(self.input_directory)
        tx = None
        bd = None

        try:
            from ..io.spatialdata_loader import (
                is_spatialdata_path,
                load_from_spatialdata,
            )
            has_spatialdata_loader = True
        except Exception:
            has_spatialdata_loader = False

        if has_spatialdata_loader and is_spatialdata_path(input_path):
            tx_lf, bd = load_from_spatialdata(
                input_path,
                boundary_type="all",
                normalize=True,
            )
            tx = tx_lf.collect() if isinstance(tx_lf, pl.LazyFrame) else tx_lf

            # Keep behavior consistent with raw Xenium filtering when quality exists.
            quality_col = getattr(tx_fields, "quality", "qv")
            if (
                self.min_qv is not None
                and self.min_qv > 0
                and quality_col in tx.columns
            ):
                tx = tx.filter(pl.col(quality_col) >= self.min_qv)
        else:
            pp = get_preprocessor(
                self.input_directory,
                min_qv=self.min_qv,
                include_z=(self.use_3d is not False),
            )
            tx = pp.transcripts
            bd = pp.boundaries

        self.tx = tx
        self.bd = bd

        if bd is None or len(bd) == 0:
            raise ValueError(
                "No boundary shapes found in input data. "
                "Segger requires cell/nucleus polygons in raw input or SpatialData shapes."
            )

        # Mask transcripts to reference segmentation
        if self.segmentation_graph_mode == "nucleus":
            compartments = [tx_fields.nucleus_value]
            boundary_type = bd_fields.nucleus_value
        elif self.segmentation_graph_mode == "cell":
            compartments = [
                tx_fields.nucleus_value,
                tx_fields.cytoplasmic_value,
            ]
            boundary_type = bd_fields.cell_value
        else:
            raise ValueError(
                f"Unrecognized segmentation graph mode: "
                f"'{self.segmentation_graph_mode}'."
            )

        if tx_fields.compartment in tx.columns:
            tx_mask = pl.col(tx_fields.compartment).is_in(compartments)
        else:
            tx_mask = pl.col(tx_fields.cell_id).is_not_null()

        if bd_fields.boundary_type in bd.columns:
            bd_mask = bd[bd_fields.boundary_type] == boundary_type
        else:
            bd_mask = np.ones(len(bd), dtype=bool)

        # Generate reference AnnData
        self.ad = setup_anndata(
            transcripts=tx.filter(tx_mask),
            boundaries=bd[bd_mask],
            cell_column=tx_fields.cell_id,
            cells_embedding_size=self.cells_embedding_size,
            cells_min_counts=self.cells_min_counts,
            cells_clusters_n_neighbors=self.cells_clusters_n_neighbors,
            cells_clusters_resolution=self.cells_clusters_resolution,
            genes_min_counts=self.genes_min_counts,
            genes_clusters_n_neighbors=self.genes_clusters_n_neighbors,
            genes_clusters_resolution=self.genes_clusters_resolution,
            compute_morphology=(self.cells_representation_mode == "morphology"),
        )
        self.data = setup_heterodata(
            transcripts=tx,
            boundaries=bd,
            adata=self.ad,
            segmentation_mask=tx_mask, # This is the original mask, which is correct
            cells_embedding_key=(
                'X_pca'
                if self.cells_representation_mode == 'pca'
                else 'X_morphology'
            ),
            transcripts_graph_max_k=self.transcripts_graph_max_k,
            transcripts_graph_max_dist=self.transcripts_graph_max_dist,
            prediction_graph_mode=self.prediction_graph_mode,
            prediction_graph_max_k=self.prediction_graph_max_k,
            prediction_graph_scale_factor=self.prediction_graph_scale_factor,
            use_3d=self.use_3d,
            me_gene_pairs=self.me_gene_pairs,
        )
        # Tile graph dataset
        node_positions = torch.vstack([
            self.data['tx']['pos'],
            self.data['bd']['pos'],
        ])
        if self.tiling_mode == "adaptive":
            self.tiling = QuadTreeTiling(
                positions=node_positions,
                max_tile_size=self.tiling_nodes_per_tile,
            )
        #TODO: Remove (benchmarking only)
        elif self.tiling_mode == "square":
            self.tiling = SquareTiling(
                positions=node_positions,
                side_length=self.tiling_side_length,
            )
        else:
            raise ValueError(
                f"Unrecognized tiling strategy: '{self.tiling_mode}'."
            )
        # Objects needed by lightning model
        self.tx_embedding = (
            pl
            .from_numpy(self.ad.varm['X_corr'])
            .cast(pl.Float32)
            .with_columns(
                pl.Series(self.ad.var.index).alias(tx_fields.feature))
        )
        self.tx_similarity = torch.tensor(
            self.ad.uns['gene_cluster_similarities'])
        self.bd_similarity = torch.tensor(
            self.ad.uns['cell_cluster_similarities'])

    def setup(self, stage: str):
        """TODO: Description
        """
        # Tile dataset (inner margin) for training
        if stage == "fit":
            self.fit_dataset = TileFitDataset(
                data=self.data,
                tiling=self.tiling,
                margin=self.tiling_margin_training,            
                clone=True,  # Keep: Tiling removes edges needed in prediction
            )
            # Setup training-validation split
            n = self.fit_dataset._num_partitions
            indices = torch.randperm(n)
            split = int(self.training_fraction * n)
            self.train_indices = indices[:split]
            self.val_indices = indices[split:]

        # Tile dataset (outer margin) for prediction
        if stage == "predict":
            self.data = self.data.cuda()
            self.predict_dataset = TilePredictDataset(
                data=self.data,
                tiling=self.tiling,
                margin=self.tiling_margin_prediction,
            )
        return super().setup(stage)

    def teardown(self, stage):
        """TODO: Description
        """
        # Clean up data objects no longer needed
        if stage == "fit":
            del self.fit_dataset.data, self.fit_dataset
            gc.collect()

        if stage == "predict":
            # Note: 'self.predict_dataset.data' is not cloned; don't del
            del self.predict_dataset
            self.data = self.data.cpu()

    def train_dataloader(self):
        """TODO: Description
        """
        sampler = PartitionSampler(
            self.fit_dataset,
            max_num=self.edges_per_batch,
            mode="edge",
            subset=self.train_indices.clone(),
            shuffle=True,
        )
        return DataLoader(
            self.fit_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        """TODO: Description
        """
        sampler = PartitionSampler(
            self.fit_dataset,
            max_num=self.edges_per_batch,
            mode="edge",
            subset=self.val_indices.clone(),
            shuffle=False,
        )
        return DataLoader(
            self.fit_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        """TODO: Description
        """
        sampler = DynamicBatchSamplerPatch(
            self.predict_dataset,
            max_num=self.edges_per_batch,
            mode='edge',
            shuffle=False,
            skip_too_big=False,
        )
        return DataLoader(
            self.predict_dataset,
            batch_sampler=sampler,
            shuffle=False,
        )
