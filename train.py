import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR, UNet, DynUNet, SegResNet
from monai.data import DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch


PATCH_SIZE = (96, 96, 16)
INTENSITY_MIN, INTENSITY_MAX = -175, 250
PIXDIM = (1.5, 1.5, 2.0)


def build_transforms():
    def pad_to_patch(data):
        data["image"] = F.pad(data["image"], (3, 3, 8, 8, 15, 15), "constant", 0)
        data["label"] = F.pad(data["label"], (3, 3, 8, 8, 15, 15), "constant", 0)
        return data

    shared = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=PIXDIM, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], a_min=INTENSITY_MIN, a_max=INTENSITY_MAX, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]

    train_transforms = Compose(
        shared
        + [
            pad_to_patch,
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=1,
                neg=0.1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        ]
    )
    val_transforms = Compose(shared)
    return train_transforms, val_transforms


def build_model(net_type: str, device: torch.device) -> torch.nn.Module:
    net_type = net_type.lower()

    if net_type == "unet":
        class _Model(UNet):
            def forward(self, x):
                return torch.sigmoid(super().forward(x))

        return _Model(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="instance",
        ).to(device)

    if net_type == "unetr":
        class _Model(UNETR):
            def forward(self, x):
                return torch.sigmoid(super().forward(x))

        return _Model(
            in_channels=1,
            out_channels=1,
            img_size=PATCH_SIZE,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.2,
        ).to(device)

    if net_type == "dynunet":
        class _Model(DynUNet):
            def forward(self, x):
                return torch.sigmoid(super().forward(x))

        return _Model(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            kernel_size=[3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2],
            filters=(16, 32, 64, 128, 256),
            norm_name="instance",
        ).to(device)

    if net_type == "segresnet":
        class _Model(SegResNet):
            def forward(self, x):
                return torch.sigmoid(super().forward(x))

        return _Model(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            init_filters=16,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            norm="instance",
        ).to(device)

    raise ValueError(f"Unknown net_type '{net_type}'. Choose: unet, unetr, dynunet, segresnet")


def pad_batch_z(x: torch.Tensor, y: torch.Tensor, target_z: int, device: torch.device):
    pad_z = target_z - x.size(4)
    if pad_z > 0:
        zeros = torch.zeros(*x.shape[:4], pad_z, dtype=torch.float32, device=device)
        x = torch.cat((x, zeros), dim=4)
        y = torch.cat((y, zeros), dim=4)
    return x, y


def validate(model, val_loader, post_label, post_pred, dice_metric, global_step):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validate (step {global_step})", dynamic_ncols=True):
            val_inputs = batch["image"].cuda()
            val_labels = batch["label"].cuda()
            val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 4, model)
            val_labels_convert = [post_label(x) for x in decollate_batch(val_labels)]
            val_output_convert = [post_pred(x) for x in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return mean_dice


def train(args, model, optimizer, train_loader, val_loader, model_path, results_csv):
    loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=False)
    post_label = AsDiscrete(threshold=0.5)
    post_pred = AsDiscrete(threshold=0.5)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    device = next(model.parameters()).device

    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values, metric_values, eval_steps = [], [], []

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Resumed from {model_path}")
    else:
        print("No checkpoint found, starting from scratch.")

    while global_step < args.max_iter:
        model.train()
        epoch_loss = 0.0
        step = 0
        pbar = tqdm(train_loader, desc=f"Training (0 / {args.max_iter}) (loss=?.?????)", dynamic_ncols=True)
        for step, batch in enumerate(pbar, 1):
            x = batch["image"].cuda()
            y = batch["label"].cuda()
            x, y = pad_batch_z(x, y, PATCH_SIZE[2], device)

            logit_map = model(x)
            loss = loss_fn(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(
                f"Training ({global_step} / {args.max_iter}) (loss={loss.item():.5f})"
            )

            if (global_step % args.eval_num == 0 and global_step != 0) or global_step == args.max_iter - 1:
                dice_val = validate(model, val_loader, post_label, post_pred, dice_metric, global_step)
                avg_loss = epoch_loss / step
                eval_steps.append(global_step)
                epoch_loss_values.append(avg_loss)
                metric_values.append(dice_val)

                pd.DataFrame(
                    {"step": eval_steps, "dice_loss": epoch_loss_values, "dice_metric": metric_values}
                ).to_csv(results_csv, index=False)

                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), model_path)
                    print(f"  [saved] step={global_step}  best_dice={dice_val_best:.4f}")
                else:
                    print(
                        f"  [no save] step={global_step}  dice={dice_val:.4f}  best={dice_val_best:.4f}"
                    )

            global_step += 1

    print(f"\nTraining complete. Best Dice: {dice_val_best:.4f} at step {global_step_best}")
    return epoch_loss_values, metric_values, eval_steps


def plot_results(epoch_loss_values, metric_values, eval_steps, save_path):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    plt.xlabel("Step")
    plt.plot(eval_steps, epoch_loss_values)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    plt.xlabel("Step")
    plt.plot(eval_steps, metric_values)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model on soil core CT data.")
    parser.add_argument("--net", default="unet", choices=["unet", "unetr", "dynunet", "segresnet"])
    parser.add_argument("--data-dir", default=os.path.join("..", "datasets", "soilcores"))
    parser.add_argument("--split-json", default="dataset_2.json")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--eval-num", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    split_name = args.split_json.replace(".json", "")
    model_name = f"{args.net}_{split_name}"
    model_path = os.path.join(args.output_dir, f"best_metric_model{model_name}.pth")
    results_csv = os.path.join(args.output_dir, f"results_{model_name}.csv")
    plot_path = os.path.join(args.output_dir, f"training_plot_{model_name}.png")

    train_transforms, val_transforms = build_transforms()

    datasets_json = os.path.join(args.data_dir, args.split_json)
    datalist = load_decathlon_datalist(datasets_json, True, "training")
    val_files = load_decathlon_datalist(datasets_json, True, "validation")

    train_ds = CacheDataset(
        data=datalist, transform=train_transforms, cache_num=24, cache_rate=1.0, num_workers=args.num_workers
    )
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=args.num_workers
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.net, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Model: {args.net}  |  Max iters: {args.max_iter}  |  Eval every: {args.eval_num}")
    print(f"Checkpoint: {model_path}")

    loss_vals, dice_vals, steps = train(args, model, optimizer, train_loader, val_loader, model_path, results_csv)
    plot_results(loss_vals, dice_vals, steps, plot_path)


if __name__ == "__main__":
    main()
