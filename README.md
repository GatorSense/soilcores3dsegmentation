# Visualizing and Quantifying Plant Root Distribution in Soils: A CT Scan and Machine Learning Approach

This repository provides tools to segment plant roots in 3D soil core CT scans using deep learning, extract root topology metrics (length by diameter range), and evaluate model performance against ground truth measurements.

## Procedures

| Script | Description | Guide |
|---|---|---|
| `soilcore_gui.py` | Interactive GUI — load NIfTI scans, run segmentation, apply thresholds, and extract root topology | [READMEgui.md](READMEgui.md) |
| `train.py` | Train a new segmentation model (UNet, UNETR, DynUNet, SegResNet) on labeled soil core data | [READMEtrain.md](READMEtrain.md) |
| `ComputeMetrics.py` | Evaluate model predictions against ground truth root length measurements | [READMEeval.md](READMEeval.md) |

## Environment Setup

```bash
conda create -n soilcores python=3.10
conda activate soilcores
pip install -r requirements.txt
```

Requires CUDA 12.1 (`torch==2.1.0+cu121`). Pre-trained model weights go in `models/`.
