"""
Enhanced Training & Fine-Tuning Pipeline for Plant Identification Experimentation.
Implements:
  - Staged Transfer Learning (Head warmup + Full Fine-Tuning)
  - Class-Weighted CrossEntropyLoss with Label Smoothing (solves weak class collapse)
  - Cosine Annealing Learning Rate Scheduler
  - Mixed Precision Training (AMP)
  - Checkpointing and Training History Visualizer
"""

import time
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import get_dataloaders
from src.model import build_model, set_parameter_requires_grad, count_parameters

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            # Top-1
            _, top1_preds = outputs.max(1)
            top1_correct += top1_preds.eq(labels).sum().item()

            # Top-5
            _, top5_preds = outputs.topk(min(5, outputs.size(1)), dim=1)
            top5_correct += top5_preds.eq(labels.view(-1, 1)).sum().item()

    val_loss = running_loss / total
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    return val_loss, top1_acc, top5_acc

def train_pipeline(
    epochs: int = config.TOTAL_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    use_pretrained: bool = config.USE_PRETRAINED,
    use_class_weights: bool = config.USE_CLASS_WEIGHTS,
    lr: float = config.PEAK_LR
):
    print("=" * 70)
    print("🌿 STARTING EXPERIMENTAL ENHANCED TRAINING PIPELINE")
    print(f"Device: {config.DEVICE} | Mixed Precision: {config.USE_AMP}")
    print(f"Pretrained Weights: {use_pretrained} | Class Weights: {use_class_weights}")
    print(f"RandAugment Enabled: {config.USE_RAND_AUGMENT} | Target Epochs: {epochs}")
    print("=" * 70)

    train_loader, val_loader, test_loader, classes, class_weights = get_dataloaders(
        batch_size=batch_size
    )

    # Save class indices
    with open(config.CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes,
            "class_to_idx": {cls: i for i, cls in enumerate(classes)},
            "idx_to_class": {i: cls for i, cls in enumerate(classes)}
        }, f, indent=2)

    # Build Model
    model = build_model(num_classes=len(classes), pretrained=use_pretrained)
    model = model.to(config.DEVICE)
    total_p, trainable_p = count_parameters(model)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Total Parameters: {total_p:,} | Trainable: {trainable_p:,}")

    # Loss Criterion
    weights_tensor = class_weights.to(config.DEVICE) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(
        weight=weights_tensor,
        label_smoothing=config.LABEL_SMOOTHING
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler: Linear Warmup followed by Cosine Annealing
    warmup_epochs = config.WARMUP_EPOCHS
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=config.MIN_LR)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler(device="cuda") if (config.USE_AMP and config.DEVICE.type == "cuda") else None

    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_top1": [], "val_top5": [], "lr": []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, config.DEVICE, config.USE_AMP
        )

        val_loss, val_top1, val_top5 = evaluate(
            model, val_loader, criterion, config.DEVICE
        )

        scheduler.step()
        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_top1"].append(round(val_top1, 4))
        history["val_top5"].append(round(val_top5, 4))
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Top1: {val_top1*100:.2f}% Top5: {val_top5*100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Checkpointing
        is_best = val_top1 > best_val_acc
        if is_best:
            best_val_acc = val_top1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_top1,
                "classes": classes,
                "config": {
                    "pretrained": use_pretrained,
                    "class_weights": use_class_weights,
                    "rand_augment": config.USE_RAND_AUGMENT
                }
            }, config.BEST_MODEL_PATH)
            print(f"  ⭐ Saved new best model checkpoint! (Top-1: {val_top1*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs without improvement.")
                break

    total_time_mins = (time.time() - start_time) / 60
    print("\n" + "=" * 70)
    print(f"🏁 Training finished in {total_time_mins:.2f} mins. Best Validation Top-1: {best_val_acc*100:.2f}%")
    print(f"Best model stored at: {config.BEST_MODEL_PATH}")
    print("=" * 70)

    # Save training history
    with open(config.ARTIFACTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([a * 100 for a in history["train_acc"]], label="Train Acc")
    plt.plot([a * 100 for a in history["val_top1"]], label="Val Top-1 Acc")
    plt.plot([a * 100 for a in history["val_top5"]], label="Val Top-5 Acc")
    plt.title("Accuracy Curves (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(config.TRAINING_CURVES_PATH, dpi=200)
    plt.close()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experimental Enhanced Training")
    parser.add_argument("--epochs", type=int, default=config.TOTAL_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.PEAK_LR, help="Peak learning rate")
    args = parser.parse_args()

    train_pipeline(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
