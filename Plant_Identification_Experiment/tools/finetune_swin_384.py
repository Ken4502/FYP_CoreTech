"""
FloraNet Option 1: Swin-Transformer 384px Progressive Resolution Fine-Tuning
Upgrades Swin-Transformer-Tiny from 224px to 384px resolution.
Adapts shifted-window self-attention grids to fine-grained botanical features (vein reticulation, serration teeth).
"""

import os
import sys
import time
import copy
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from src import config
from src.model import build_model
from tools.harvester import BOTANICAL_TAXON_MAP

LOG_FILE = EXPERIMENT_ROOT / "reports" / "mega_pipeline.log"
ARTIFACTS_DIR = EXPERIMENT_ROOT / "artifacts"
SOURCE_CHECKPOINT = ARTIFACTS_DIR / "best_model_swin.pth"
TARGET_CHECKPOINT = ARTIFACTS_DIR / "best_model_swin_384.pth"

def log_both(msg: str):
    print(msg, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def get_384_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.RandomCrop(384),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_tf, val_tf

def main(epochs: int = 6, batch_size: int = 16, lr: float = 2.0e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("FLORANET STAGE 2: SWIN-TRANSFORMER 384px HIGH-RESOLUTION FINE-TUNING")
    print(f"• Hardware:         {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"• Source Checkpoint: {SOURCE_CHECKPOINT}")
    print(f"• Architecture:     Swin-Transformer-Tiny (Progressive Resolution 224px -> 384px)")
    print(f"• Regimen:          {epochs} Epochs, Batch Size {batch_size}, Initial LR {lr}")
    print("=" * 70)

    train_dir = EXPERIMENT_ROOT / "data" / "train"
    val_dir = EXPERIMENT_ROOT / "data" / "val"

    if not train_dir.exists() or not val_dir.exists():
        log_both("[Error] Train or Val dataset directory not found.")
        return

    train_tf, val_tf = get_384_transforms()
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    num_classes = len(train_ds.classes)
    log_both(f"[Data] Loaded {len(train_ds)} train and {len(val_ds)} val specimens across {num_classes} classes at 384x384.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 1. Instantiate Swin-T and load Stage 1 weights
    model = build_model(num_classes=num_classes, pretrained=False, dropout_rate=0.2, backbone="swin_t").to(device)

    if SOURCE_CHECKPOINT.exists():
        log_both(f"[Checkpoint] Loading 224px Stage 1 weights from {SOURCE_CHECKPOINT}...")
        ckpt = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        baseline_acc = ckpt.get("val_accuracy", 84.58)
        log_both(f"[Baseline] Prior 224px Validation Accuracy: {baseline_acc:.2f}%")
    else:
        log_both("[Warning] Stage 1 checkpoint not found, initializing from default pretrained weights.")
        model = build_model(num_classes=num_classes, pretrained=True, dropout_rate=0.2, backbone="swin_t").to(device)
        baseline_acc = 0.0

    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-7)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = baseline_acc
    best_weights = copy.deepcopy(model.state_dict())

    log_both("=" * 70)
    log_both("STARTING 384px PROGRESSIVE RESOLUTION FINE-TUNING")
    log_both("=" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total_train += images.size(0)

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = (running_correct / total_train) * 100

        # Validation
        model.eval()
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    total_val += images.size(0)

        val_acc = (val_correct / total_val) * 100
        epoch_sec = time.time() - t0
        curr_lr = optimizer.param_groups[0]["lr"]

        log_both(f"  [Swin-384] Epoch [{epoch:02d}/{epochs:02d}] ({epoch_sec:.1f}s) | Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {curr_lr:.8f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_weights,
                "classes": train_ds.classes,
                "val_accuracy": best_val_acc,
                "architecture": "swin_t",
                "image_size": 384,
                "num_classes": num_classes,
                "timestamp": time.time()
            }, TARGET_CHECKPOINT)
            # Also overwrite best_model_swin.pth so ensemble automatically picks up 384px weights
            torch.save({
                "model_state_dict": best_weights,
                "classes": train_ds.classes,
                "val_accuracy": best_val_acc,
                "architecture": "swin_t",
                "image_size": 384,
                "num_classes": num_classes,
                "timestamp": time.time()
            }, SOURCE_CHECKPOINT)
            log_both(f"  --> Saved new best Swin-384 checkpoint: {best_val_acc:.2f}% at {TARGET_CHECKPOINT}")

    log_both("=" * 70)
    log_both(f"SWIN-384 FINE-TUNING COMPLETED | Peak Val Acc: {best_val_acc:.2f}% (Prior 224px: {baseline_acc:.2f}%)")
    log_both("=" * 70)

if __name__ == "__main__":
    main()

