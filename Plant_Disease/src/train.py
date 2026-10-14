"""
Hierarchical Multi-Head Training Pipeline for FloraNet-Hierarchical.
Trains:
  1. Shared Botanical MobileNetV3 Backbone
  2. Stage 1 Species Router Head
  3. Stage 2 Specialized Pathology Heads per Plant Group
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

from src import config
from src.dataset import get_dataloaders, compute_group_class_weights
from src.model import build_model
from src.disease_knowledge import export_knowledge_base_json

def train_one_epoch(
    model,
    dataloader,
    criterion_species,
    criterion_disease,
    optimizer,
    scaler,
    device,
    use_amp
):
    model.train()
    total_loss = 0.0
    species_correct = 0
    disease_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, species_targets, condition_targets, global_targets in pbar:
        images = images.to(device, non_blocking=True)
        species_targets = species_targets.to(device, non_blocking=True)
        condition_targets = condition_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                species_logits, disease_outputs = model(images)
                loss_sp = criterion_species(species_logits, species_targets)

                loss_dis = 0.0
                batch_size = images.size(0)

                for sp_idx, sp_name in enumerate(config.SPECIES_NAMES):
                    mask = (species_targets == sp_idx)
                    if mask.sum() > 0:
                        cond_logits = disease_outputs[sp_name][mask]
                        cond_labels = condition_targets[mask]
                        loss_dis += criterion_disease[sp_name](cond_logits, cond_labels) * (mask.sum() / batch_size)

                loss = loss_sp + 1.2 * loss_dis

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            species_logits, disease_outputs = model(images)
            loss_sp = criterion_species(species_logits, species_targets)

            loss_dis = 0.0
            batch_size = images.size(0)

            for sp_idx, sp_name in enumerate(config.SPECIES_NAMES):
                mask = (species_targets == sp_idx)
                if mask.sum() > 0:
                    cond_logits = disease_outputs[sp_name][mask]
                    cond_labels = condition_targets[mask]
                    loss_dis += criterion_disease[sp_name](cond_logits, cond_labels) * (mask.sum() / batch_size)

            loss = loss_sp + 1.2 * loss_dis
            loss.backward()
            optimizer.step()

        # Metrics
        total_loss += loss.item() * images.size(0)
        _, sp_preds = torch.max(species_logits, 1)
        species_correct += (sp_preds == species_targets).sum().item()

        for sp_idx, sp_name in enumerate(config.SPECIES_NAMES):
            mask = (species_targets == sp_idx)
            if mask.sum() > 0:
                cond_logits = disease_outputs[sp_name][mask]
                cond_labels = condition_targets[mask]
                _, cond_preds = torch.max(cond_logits, 1)
                disease_correct += (cond_preds == cond_labels).sum().item()

        total_samples += images.size(0)

    epoch_loss = total_loss / total_samples
    epoch_sp_acc = species_correct / total_samples
    epoch_dis_acc = disease_correct / total_samples
    return epoch_loss, epoch_sp_acc, epoch_dis_acc

@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion_species, criterion_disease, device):
    model.eval()
    total_loss = 0.0
    species_correct = 0
    disease_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for images, species_targets, condition_targets, global_targets in pbar:
        images = images.to(device, non_blocking=True)
        species_targets = species_targets.to(device, non_blocking=True)
        condition_targets = condition_targets.to(device, non_blocking=True)

        species_logits, disease_outputs = model(images)
        loss_sp = criterion_species(species_logits, species_targets)

        loss_dis = 0.0
        batch_size = images.size(0)

        for sp_idx, sp_name in enumerate(config.SPECIES_NAMES):
            mask = (species_targets == sp_idx)
            if mask.sum() > 0:
                cond_logits = disease_outputs[sp_name][mask]
                cond_labels = condition_targets[mask]
                loss_dis += criterion_disease[sp_name](cond_logits, cond_labels) * (mask.sum() / batch_size)

        loss = loss_sp + 1.2 * loss_dis
        total_loss += loss.item() * images.size(0)

        _, sp_preds = torch.max(species_logits, 1)
        species_correct += (sp_preds == species_targets).sum().item()

        for sp_idx, sp_name in enumerate(config.SPECIES_NAMES):
            mask = (species_targets == sp_idx)
            if mask.sum() > 0:
                cond_logits = disease_outputs[sp_name][mask]
                cond_labels = condition_targets[mask]
                _, cond_preds = torch.max(cond_logits, 1)
                disease_correct += (cond_preds == cond_labels).sum().item()

        total_samples += images.size(0)

    epoch_loss = total_loss / total_samples
    epoch_sp_acc = species_correct / total_samples
    epoch_dis_acc = disease_correct / total_samples
    return epoch_loss, epoch_sp_acc, epoch_dis_acc

def plot_history(history, save_path):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-s", label="Val Loss")
    ax1.set_title("FloraNet-Hierarchical: Total Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, [a * 100 for a in history["val_dis_acc"]], "g-^", label="Val Disease Diagnosis Acc")
    ax2.plot(epochs, [a * 100 for a in history["val_sp_acc"]], "m-o", label="Val Species Routing Acc")
    ax2.set_title("FloraNet-Hierarchical: Validation Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {save_path}")

def run_training(pretrained: bool = True, epochs: int = config.TOTAL_EPOCHS):
    print("=" * 65)
    print(f"🔬 Training Hierarchical Multi-Head Model: {config.MODEL_NAME}")
    print(f"   Plant Species Groups: {len(config.PLANT_GROUPS)}")
    print(f"   Device: {config.DEVICE}")
    print("=" * 65)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    export_knowledge_base_json(config.DISEASE_KB_PATH)

    train_loader, val_loader, _, _ = get_dataloaders()

    # Save plant groups metadata
    with open(config.PLANT_GROUPS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "species": config.SPECIES_NAMES,
            "groups": config.PLANT_GROUPS
        }, f, indent=2)

    # Initialize Hierarchical Model
    backbone_path = config.PRETRAINED_BACKBONE_PATH if pretrained else None
    model = build_model(
        plant_groups=config.PLANT_GROUPS,
        dropout_rate=config.DROPOUT_RATE,
        pretrained_backbone_path=backbone_path
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # Loss functions
    criterion_species = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    group_weights = compute_group_class_weights(train_loader.dataset)
    criterion_disease = {}
    for sp_name, w in group_weights.items():
        w = w.to(config.DEVICE)
        criterion_disease[sp_name] = nn.CrossEntropyLoss(weight=w, label_smoothing=config.LABEL_SMOOTHING)

    scaler = torch.amp.GradScaler(enabled=config.USE_AMP and config.DEVICE.type == "cuda")
    optimizer = AdamW(model.parameters(), lr=config.PEAK_LR, weight_decay=config.WEIGHT_DECAY)

    warmup_epochs = config.WARMUP_EPOCHS
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=config.MIN_LR
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_sp_acc": [],
        "val_sp_acc": [],
        "train_dis_acc": [],
        "val_dis_acc": []
    }

    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n--- Starting Hierarchical Training ({epochs} Epochs) ---")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        current_lr = scheduler.get_last_lr()[0]

        train_loss, train_sp, train_dis = train_one_epoch(
            model, train_loader, criterion_species, criterion_disease,
            optimizer, scaler, config.DEVICE, config.USE_AMP
        )

        val_loss, val_sp, val_dis = evaluate_epoch(
            model, val_loader, criterion_species, criterion_disease, config.DEVICE
        )

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_sp_acc"].append(train_sp)
        history["val_sp_acc"].append(val_sp)
        history["train_dis_acc"].append(train_dis)
        history["val_dis_acc"].append(val_dis)

        print(f"Epoch [{epoch:>2}/{epochs}] "
              f"LR: {current_lr:.2e} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Species Acc: {val_sp*100:.1f}% | "
              f"Disease Acc: {val_dis*100:.1f}% | Time: {elapsed:.1f}s")

        if val_dis > best_val_acc:
            best_val_acc = val_dis
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_name": config.MODEL_NAME,
                "model_state_dict": model.state_dict(),
                "plant_groups": config.PLANT_GROUPS,
                "species_names": config.SPECIES_NAMES,
                "val_disease_acc": val_dis,
                "val_species_acc": val_sp
            }, config.BEST_MODEL_PATH)
            print(f"  [*] Saved new best model checkpoint ({best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n[!] Early stopping triggered at epoch {epoch}.")
                break

    # Save last checkpoint
    torch.save({
        "epoch": len(history["train_loss"]),
        "model_name": config.MODEL_NAME,
        "model_state_dict": model.state_dict(),
        "plant_groups": config.PLANT_GROUPS,
        "species_names": config.SPECIES_NAMES,
        "val_disease_acc": history["val_dis_acc"][-1],
        "val_species_acc": history["val_sp_acc"][-1]
    }, config.LAST_MODEL_PATH)

    with open(config.ARTIFACTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_history(history, config.TRAINING_CURVES_PATH)
    print(f"\n=== Training Finished! Best Disease Diagnosis Accuracy: {best_val_acc*100:.2f}% ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical FloraNet Training")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch without botanical backbone")
    parser.add_argument("--epochs", type=int, default=config.TOTAL_EPOCHS, help="Number of epochs")
    args = parser.parse_args()

    run_training(pretrained=not args.scratch, epochs=args.epochs)
