# Training a Segmentation Model

This document describes how to train a new segmentation model on soil core CT data using `train.py`.

## Dataset Preparation

The training script expects data organized in the [Medical Segmentation Decathlon](http://medicalsegmentation.com/decathlon/) format, with a JSON file listing the training and validation splits.

Expected folder structure:
```
datasets/
└── soilcores/
    ├── dataset.json
    └── data/
        ├── training/
        │   ├── images/   ← NIfTI volumes (.nii.gz)
        │   └── labels/   ← binary NIfTI labels (.nii.gz)
        └── validation/
```

The JSON file (`dataset.json`) should follow this structure:
```json
{
  "training": [
    {"image": "data/training/images/C1216.nii.gz", "label": "data/training/labels/C1216.nii.gz"}
  ],
  "valmodel": [
    {"image": "data/training/images/S1919.nii.gz", "label": "data/training/labels/S1919.nii.gz"}
  ]
}
```

## Running Training

```bash
python train.py [options]
```

### Options

| Argument | Default | Description |
|---|---|---|
| `--net` | `unet` | Architecture: `unet`, `unetr`, `dynunet`, `segresnet` |
| `--data-dir` | `data` | Path to dataset folder containing the JSON file |
| `--split-json` | `dataset.json` | Dataset split JSON filename |
| `--output-dir` | `models` | Directory for checkpoints, CSVs, and plots |
| `--max-iter` | `10000` | Total number of training steps |
| `--eval-num` | `100` | Validation frequency (steps) |
| `--lr` | `1e-2` | Learning rate (AdamW) |
| `--weight-decay` | `1e-6` | Weight decay (AdamW) |
| `--num-workers` | `1` | DataLoader worker processes |

### Examples

```bash
# UNet, default settings
python train.py

# UNETR for 100k iterations, validated every 500 steps
python train.py --net unetr --max-iter 100000 --eval-num 500 --lr 1e-4

# SegResNet on a different dataset split
python train.py --net segresnet --split-json dataset_1.json

# Custom data directory and output directory
python train.py --data-dir /path/to/soilcores --output-dir /path/to/checkpoints
```

## Outputs

All outputs are written to `--output-dir` (default: `models/`). The filename suffix is derived from the architecture and split JSON name (e.g., `unet_dataset_2`).

| File | Description |
|---|---|
| `best_metric_model{name}.pth` | Best model checkpoint (highest validation Dice) |
| `results_{name}.csv` | Loss and Dice metric logged at each evaluation step |
| `training_plot_{name}.png` | Plot of training loss and validation Dice over steps |

If a checkpoint already exists for the given model name, training resumes from it automatically.

## Model Architectures

All architectures are binary segmentation models (1 input channel, 1 output channel) with a sigmoid activation applied at the output.

| Name | Class | Notes |
|---|---|---|
| `unet` | `monai.networks.nets.UNet` | 5-level encoder, instance norm, 2 residual units per block |
| `unetr` | `monai.networks.nets.UNETR` | Transformer encoder, patch size 96×96×16, 12 attention heads |
| `dynunet` | `monai.networks.nets.DynUNet` | Dynamic UNet with instance norm |
| `segresnet` | `monai.networks.nets.SegResNet` | Residual encoder-decoder with instance norm |

## Data Transforms

Training transforms applied per sample:
- Reorient to RAS, resample to 1.5×1.5×2.0 mm voxels
- Intensity clipping and normalization to [0, 1] (CT range: −175 to 250 HU)
- Foreground cropping
- Random 96×96×16 patch sampling (positive/negative ratio 1:0.1)
- Random flips (all axes, p=0.10), random 90° rotations (p=0.10)
- Random intensity shift ±0.10 (p=0.50)

Validation uses the same spatial/intensity transforms without augmentation.

## Loss and Optimizer

- **Loss:** `DiceCELoss` (Dice + Cross-Entropy, combined)
- **Optimizer:** AdamW (`lr=1e-2`, `weight_decay=1e-6` by default)
- **Metric:** Mean Dice (background excluded); best checkpoint saved when this improves
