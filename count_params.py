import torch
from train import build_model

device = torch.device("cpu")
models = ["unet", "unetr", "dynunet", "segresnet"]

print(f"{'Model':<12} {'Total params':>15} {'Trainable params':>18}")
print("-" * 47)
for name in models:
    model = build_model(name, device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name:<12} {total:>15,} {trainable:>18,}")
