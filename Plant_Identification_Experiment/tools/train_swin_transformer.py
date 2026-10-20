"""
FloraNet Phase 2: Swin-Transformer (Swin-T) Training Engine
Trains a Swin-Transformer (Shifted Window Self-Attention) on the 250-class botanical corpus (~97,500 images).
Provides a vision-transformer counterpart to ConvNeXt-Small for tournament-grade heterogeneous ensembling.
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
CHECKPOINT_PATH = ARTIFACTS_DIR / "best_model_swin.pth"

def log_both(msg: str):
    print(msg, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def get_swin_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_tf, val_tf

def main(epochs: int = 12, batch_size: int = 32, lr: float = 8e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("FLORANET HETEROGENEOUS ENSEMBLE: SWIN-TRANSFORMER-TINY TRAINING ENGINE")
    print(f"• Target Device:    {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"• Architecture:     Swin-Transformer-Tiny (Shifted-Window Self-Attention)")
    print(f"• Dataset:          250 Botanical Classes (~97,500 total specimens)")
    print(f"• Training Regimen: {epochs} Epochs, Batch Size {batch_size}, Initial LR {lr}")
    print("=" * 70)

    train_dir = EXPERIMENT_ROOT / "data" / "train"
    val_dir = EXPERIMENT_ROOT / "data" / "val"

    if not train_dir.exists() or not val_dir.exists():
        log_both("[Error] Train or Val dataset directory not found.")
        return

    train_tf, val_tf = get_swin_transforms(224)
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)

    num_classes = len(train_ds.classes)
    log_both(f"[Data] Found {len(train_ds)} training images, {len(val_ds)} validation images across {num_classes} classes.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Build Swin-T
    log_both("[Model] Instantiating Swin-T pretrained backbone...")
    model = build_model(num_classes=num_classes, pretrained=True, dropout_rate=0.2, backbone="swin_t").to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    log_both("=" * 70)
    log_both("COMMENCING SWIN-TRANSFORMER FINE-TUNING")
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

        log_both(f"  [Swin-T] Epoch [{epoch:02d}/{epochs:02d}] ({epoch_sec:.1f}s) | Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {curr_lr:.7f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_weights,
                "classes": train_ds.classes,
                "val_accuracy": best_val_acc,
                "architecture": "swin_t",
                "image_size": 224,
                "num_classes": num_classes,
                "timestamp": time.time()
            }, CHECKPOINT_PATH)
            log_both(f"  --> Saved new best Swin-T checkpoint: {best_val_acc:.2f}% at {CHECKPOINT_PATH}")

    log_both("=" * 70)
    log_both(f"SWIN-TRANSFORMER TRAINING COMPLETED | Best Val Acc: {best_val_acc:.2f}%")
    log_both("=" * 70)

if __name__ == "__main__":
    main()

