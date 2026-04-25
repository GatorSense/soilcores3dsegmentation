# Evaluation: Computing Prediction Metrics

This document describes how to evaluate segmentation model predictions against ground truth root length measurements using `ComputeMetrics.py`.

## Overview

The script compares per-core root length predictions from multiple models against two ground truth datasets (Manhattan and Colombia/Stillwater splits), computes regression metrics, and saves scatter plots and summary CSVs.

## Required Inputs

### Ground Truth CSVs

| File | Split |
|---|---|
| `gt/CoresGT_MNHT.csv` | Manhattan — used as the test set |
| `gt/CoresGT_COL_STL.csv` | Colombia/Stillwater — used as the validation set |

Both files must contain a core identifier column (`label ID` for MNHT, `corename` for COL/STL) and a `Length(cm)` column.

### Prediction CSVs

For each model to evaluate, place two prediction files in `outputs/`:

| File | Split |
|---|---|
| `outputs/prediction_{model}_col_stl.csv` | Colombia/Stillwater predictions |
| `outputs/prediction_{model}_mnht.csv` | Manhattan predictions |

Each file must contain a `soil_core` column (core identifier) and one or more root length feature columns (e.g., `Root Length Diameter Range 2 (px)`).

## Configuration

Edit the `model_list` at the top of `ComputeMetrics.py` to select which models to evaluate and which predicted feature column to compare against ground truth:

```python
model_list = [
    {"model": "dynunet",   "feat_pred": "Root Length Diameter Range 3 (px)"},
    {"model": "segresnet", "feat_pred": "Root Length Diameter Range 2 (px)"},
    {"model": "unet",      "feat_pred": "Root Length Diameter Range 2 (px)"},
    {"model": "unetr",     "feat_pred": "Root Length Diameter Range 2 (px)"},
]
```

The ground truth feature is set by `feat_gt` further down the script (default: `"Length(cm)"`).

## Running

```bash
python ComputeMetrics.py
```

## Outputs

All outputs are written to `outputs/`. The filename encodes the ground truth feature used.

| File | Description |
|---|---|
| `outputs/model_metrics_mnht_feat_gt_{feat_gt}.csv` | Test metrics (Manhattan split) for all models |
| `outputs/model_metrics_col_stl_feat_gt_{feat_gt}.csv` | Validation metrics (Colombia/Stillwater split) for all models |

Each CSV contains one row per model with the following columns:

| Column | Description |
|---|---|
| `model` | Model name |
| `pearson` | Pearson correlation coefficient (r) |
| `spearman` | Spearman rank correlation (ρ) |
| `r2` | Coefficient of determination (R²) |
| `mse` | Mean squared error |
| `mean_error` | Mean prediction error (bias) |
| `std_error` | Standard deviation of prediction error |
| `n` | Number of cores in the split |

A scatter plot (GT vs. predicted, standardized) is also displayed for each model, annotated with r and ρ for both splits.

## Normalization

Before computing metrics, both GT and predicted values are z-score normalized using the mean and standard deviation of the full matched test split. This puts both datasets on the same scale regardless of the units of the selected feature columns.
