# Parameter Steps (Committed Baseline)

Branch baseline: `v2-incremental`  
Commit: `26adcc3` (`Align prediction scale-factor CLI and benchmark defaults`)

This file documents the parameter schedules used by the committed run scripts.

## Canonical Defaults (from `main`)

Source: `main:src/segger/cli/config.yaml`

- `cells_min_counts: 10`
- `transcripts_max_k: 5`
- `transcripts_max_dist: 5`
- `n_mid_layers: 2`
- `n_heads: 2`
- `in_channels (node dim): 128`
- `hidden_channels: 64`
- `out_channels: 64`
- `learning_rate: 1e-3`
- `n_epochs: 10`
- `segmentation_loss: triplet`
- `transcripts_loss_weight_end: 1.0`
- `cells_loss_weight_end: 1.0`
- `segmentation_loss_weight_end: 0.1`

Note:
- `main` does not contain these benchmark/ablation shell runners.
- Runner-specific defaults below are script-level overrides layered on top of
  this canonical model/data baseline.

## 1) `run_param_benchmark_2gpu.sh`

Purpose: one-factor-at-a-time benchmark around baseline.

Script baseline defaults:
- `use_3d=true`
- `prediction_scale_factor=2.2`
- `transcripts_max_k=5`
- `transcripts_max_dist=5`
- `n_mid_layers=2`
- `n_heads=2`
- `cells_min_counts=5`
- `min_qv=0`
- `alignment_loss=true`

Sweep values:
- `USE_3D_VALUES=(false true)`
- `EXPANSION_VALUES=(1 1.5 2.0 2.2 2.5 3.0)`
- `TX_MAX_K_VALUES=(5 10 20)`
- `TX_MAX_DIST_VALUES=(3 5 10 20)`
- `N_MID_LAYER_VALUES=(1 2 3)`
- `N_HEAD_VALUES=(2 4 8)`
- `CELLS_MIN_COUNTS_VALUES=(3 5 10)`
- `ALIGNMENT_VALUES=(false true)`

Dry-run count:
- Total jobs: `19`
- GPU split: `10 / 9`

## 2) `run_ablation_study.sh`

Purpose: component ablation study (loss, graph topology, architecture, prediction mode).

Script anchor defaults:
- `use_3d=true`
- `prediction_scale_factor=2.2`
- `transcripts_max_k=5`
- `transcripts_max_dist=20`
- `n_mid_layers=2`
- `n_heads=4`
- `cells_min_counts=5`
- `min_qv=0`
- `alignment_loss=true`
- `segmentation_loss=triplet`
- `hidden_channels=64`
- `out_channels=64`
- `tx/bd/sg/align weights = 1.0 / 1.0 / 0.5 / 0.03`
- `positional_embeddings=true`
- `normalize_embeddings=true`
- `cells_representation=pca`
- `learning_rate=1e-3`

Blocks (toggleable):
- Loss decomposition: `6` jobs
- Segmentation loss type: `2` jobs
- Alignment weight sweep: `5` jobs
- Architecture ablation: `12` jobs
- Graph topology ablation: `6` jobs
- Prediction mode ablation: `2` jobs
- Learning-rate ablation: `3` jobs (legacy, off by default)

Dry-run count:
- Total jobs: `33` by default (`36` if `RUN_LR_ABLATION=1`)

## 3) `run_robustness_ablation_2gpu.sh`

Purpose: repeatability + interaction + stress robustness.

Script baseline defaults:
- `use_3d=true`
- `prediction_scale_factor=2.2`
- `transcripts_max_k=5`
- `transcripts_max_dist=5`
- `n_mid_layers=2`
- `n_heads=2`
- `cells_min_counts=5`
- `min_qv=0`

Script anchor defaults:
- `use_3d=true`
- `prediction_scale_factor=2.2`
- `transcripts_max_k=5`
- `transcripts_max_dist=20`
- `n_mid_layers=2`
- `n_heads=4`
- `cells_min_counts=5`
- `min_qv=0`
- `alignment_loss=true`

Study controls:
- `STABILITY_REPEATS=3`
- `RUN_INTERACTION_GRID=1`
- `RUN_STRESS_TESTS=1`
- `INTERACTION_EXPANSIONS=(2.2 2.5)`
- `INTERACTION_TX_DISTS=(10 20)`
- `INTERACTION_HEADS=(2 4)`
- `INTERACTION_ALIGN_VALUES=(true false)` (false corners added explicitly)
- `SENS_EXPANSION_RATIO=3.0`

Dry-run count:
- Total jobs: `26`
- GPU split: `13 / 13`

## Runtime defaults shared across runners

- `SEGMENT_TIMEOUT_MIN=90`
- `SEGMENT_NUM_WORKERS=8`
- `SEGMENT_ANC_RETRY_WORKERS=0`
- `PREDICT_FALLBACK_ON_OOM=1`
- `BOUNDARY_METHOD=convex_hull`
- `BOUNDARY_VOXEL_SIZE=5`
- Alignment inputs:
  - `ALIGNMENT_SCRNA_REFERENCE_PATH=data/ref_pancreas.h5ad`
  - `ALIGNMENT_SCRNA_CELLTYPE_COLUMN=cell_type`
