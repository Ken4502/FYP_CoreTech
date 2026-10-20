"""
PhytoCare Flagship Pathology Training Engine (ConvNeXt-Small 384px)
Features:
  1. Botanical Feature Transfer from FloraNet-250 (97,500 curated plant images)
  2. Class-Balanced Focal Loss (gamma=2.0) to eliminate dominant class overpowering
  3. Progressive 2-Stage Resolution Scaling:
       - Stage 1: 256x256 (10 Epochs) for pathology feature adaptation
       - Stage 2: 384x384 (8 Epochs) for micro-lesion sharpening
  4. Softmax Temperature Calibration (T) for well-calibrated clinical probabilities
"""

import os
import sys
import time
import copy
import json
import math
from pathlib import Path
from collections import Counter

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
from torchvision import datasets
from torch.utils.data import DataLoader

from src import config
from src.model import build_disease_model
from src.dataset import get_pathology_transforms, ClassBalancedFocalLoss

LOG_FILE = EXPERIMENT_ROOT / "reports" / "mega_pipeline.log"
ARTIFACTS_DIR = EXPERIMENT_ROOT / "artifacts"
BEST_CHECKPOINT = ARTIFACTS_DIR / "best_model.pth"

def log_both(msg: str):
    print(msg, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def calibrate_temperature(model, val_loader, device):
    """Optimizes Platt Temperature (T) via Negative Log-Likelihood."""
    log_both("[Calibration] Optimizing Softmax Temperature on validation logits...")
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            for images, labels in val_loader:
                images = images.to(device)
                out = model(images)
                logits_list.append(out.cpu())
                labels_list.append(labels)

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    temperature = nn.Parameter(torch.ones(1) * 1.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(all_logits / temperature, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    opt_temp = round(temperature.item(), 2)
    nll = round(criterion(all_logits / opt_temp, all_labels).item(), 4)
    log_both(f"[Calibration] Optimal Calibrated Temperature: T = {opt_temp} (NLL: {nll})")
    return opt_temp

def train_stage(model, train_loader, val_loader, num_epochs, lr, stage_name, image_size, samples_per_class, device):
    criterion = ClassBalancedFocalLoss(samples_per_class=samples_per_class, gamma=2.0, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    log_both("=" * 70)
    log_both(f"COMMENCING {stage_name.upper()} ({image_size}x{image_size})")
    log_both("=" * 70)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
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
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    total_val += images.size(0)

        val_acc = (val_correct / total_val) * 100
        sec = round(time.time() - t0, 1)
        curr_lr = optimizer.param_groups[0]["lr"]

        log_both(f"  [{stage_name}] Epoch [{epoch:02d}/{num_epochs:02d}] ({sec}s) | Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {curr_lr:.7f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model, best_acc

def main():
    device = config.DEVICE
    log_both("=" * 70)
    log_both("PHYTONET ULTRA: CONVNEXT-SMALL BOTANICAL PATHOLOGY ENGINE")
    log_both(f"• Target Device:    {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    log_both(f"• Pretrained Prior: FloraNet-250 Botanical Features ({config.FLORANET_IDENTIFICATION_WEIGHTS})")
    log_both(f"• Classes:          16 Houseplant Pathology Taxa")
    log_both("=" * 70)

    train_dir = config.TRAIN_DIR
    val_dir = config.VAL_DIR

    # Calculate class distribution for Class-Balanced Focal Loss
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    samples_per_class = [len(list((train_dir / c).glob("*.*"))) for c in classes]

    log_both(f"[Data] Loaded {sum(samples_per_class)} train images across {len(classes)} classes.")

    # 1. Instantiate Model with FloraNet Transfer
    model = build_disease_model(num_classes=len(classes), backbone="convnext_small", transfer_floranet_weights=True).to(device)

    # 2. Stage 1: 256x256 Representation Adaptation (10 Epochs, lr=1e-4)
    train_tf_256, val_tf_256 = get_pathology_transforms(256)
    train_ds_256 = datasets.ImageFolder(str(train_dir), transform=train_tf_256)
    val_ds_256 = datasets.ImageFolder(str(val_dir), transform=val_tf_256)
    train_loader_256 = DataLoader(train_ds_256, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader_256 = DataLoader(val_ds_256, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model, stage1_acc = train_stage(model, train_loader_256, val_loader_256, num_epochs=10, lr=1.0e-4, stage_name="Stage-1-256px", image_size=256, samples_per_class=samples_per_class, device=device)

    # 3. Stage 2: 384x384 Micro-Lesion Sharpening (8 Epochs, lr=2e-5)
    train_tf_384, val_tf_384 = get_pathology_transforms(384)
    train_ds_384 = datasets.ImageFolder(str(train_dir), transform=train_tf_384)
    val_ds_384 = datasets.ImageFolder(str(val_dir), transform=val_tf_384)
    train_loader_384 = DataLoader(train_ds_384, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader_384 = DataLoader(val_ds_384, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model, stage2_acc = train_stage(model, train_loader_384, val_loader_384, num_epochs=8, lr=2.0e-5, stage_name="Stage-2-384px", image_size=384, samples_per_class=samples_per_class, device=device)

    # 4. Temperature Calibration
    opt_temp = calibrate_temperature(model, val_loader_384, device)

    # 5. Save Checkpoint
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "val_accuracy": stage2_acc,
        "temperature": opt_temp,
        "architecture": "convnext_small",
        "image_size": 384,
        "num_classes": len(classes),
        "timestamp": time.time()
    }, BEST_CHECKPOINT)

    log_both("=" * 70)
    log_both(f"PHYTONET ULTRA FLAGSHIP TRAINING COMPLETED!")
    log_both(f"  • Stage 1 (256px) Val Acc: {stage1_acc:.2f}%")
    log_both(f"  • Stage 2 (384px) Val Acc: {stage2_acc:.2f}% (Peak Acuity)")
    log_both(f"  • Calibrated Temp:         T = {opt_temp}")
    log_both(f"  • Checkpoint Saved:        {BEST_CHECKPOINT}")
    log_both("=" * 70)

if __name__ == "__main__":
    main()

