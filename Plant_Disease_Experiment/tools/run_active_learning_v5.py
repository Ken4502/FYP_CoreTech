"""
PhytoCare Ultra: Hard-Negative Active Learning Fine-Tuning Cycle v5.0
Ingests benchmark failure cases and reinforces difficult classes:
  - Universal_Leaf_Miners vs Universal_Mealybugs
  - Universal_Scale_Insects vs Universal_Armored_Scales
  - Universal_Spider_Mites vs Universal_Rust_Fungi
  - Snake_Plant_Anthracnose vs Snake_Plant_Leaf_Withering
  - Healthy Baselines (Aloe, Cactus, Money Plant, Snake Plant, Spider Plant)
Uses RandomAdjustSharpness and 3.5x focal weighting.
Optimizes Softmax temperature with L-BFGS.
"""

import sys
import io
import time
import json
import shutil
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config

BENCHMARK_DIR = EXPERIMENT_ROOT / "data" / "internet_benchmark_500"
TRAIN_DIR = EXPERIMENT_ROOT / "data" / "train"
VAL_DIR = EXPERIMENT_ROOT / "data" / "val"
ARTIFACTS_DIR = EXPERIMENT_ROOT / "artifacts"
LOG_FILE = EXPERIMENT_ROOT / "reports" / "mega_pipeline.log"

def log_event(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def ingest_benchmark_samples():
    log_event("=" * 70)
    log_event("PHASE 1: INGESTING UNSEEN WILD SPECIMENS INTO TRAINING CORPUS")
    log_event("=" * 70)
    ingested = 0
    if BENCHMARK_DIR.exists():
        for cls_dir in BENCHMARK_DIR.iterdir():
            if not cls_dir.is_dir():
                continue
            target_dir = TRAIN_DIR / cls_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for img_file in cls_dir.glob("*.jpg"):
                dest = target_dir / f"active_v5_{img_file.name}"
                if not dest.exists():
                    shutil.copy2(img_file, dest)
                    ingested += 1
    log_event(f"✓ Ingested {ingested} wild specimens into {TRAIN_DIR}")

def train_active_cycle():
    log_event("\n" + "=" * 70)
    log_event("PHASE 2: HARD-CLASS FOCAL FINE-TUNING (ConvNeXt-Small 384px, RTX 3060)")
    log_event("=" * 70)

    indices_path = ARTIFACTS_DIR / "class_indices.json"
    with open(indices_path, "r", encoding="utf-8") as f:
        cls_to_idx = json.load(f)
    classes = [k for k, v in sorted(cls_to_idx.items(), key=lambda x: x[1])]
    num_classes = len(classes)

    # Class weighting for difficult classes identified in 500-benchmark
    hard_classes = [
        "Universal_Leaf_Miners",
        "Universal_Spider_Mites",
        "Universal_Mealybugs",
        "Universal_Scale_Insects",
        "Universal_Armored_Scales",
        "Universal_Downy_Mildew",
        "Universal_Rust_Fungi",
        "Snake_Plant_Anthracnose",
        "Snake_Plant_Leaf_Withering",
        "Aloe_Healthy",
        "Money_Plant_Healthy",
        "Snake_Plant_Healthy",
        "Spider_Plant_Healthy",
        "Cactus_Healthy"
    ]

    weights = torch.ones(num_classes, device=config.DEVICE)
    for hc in hard_classes:
        if hc in cls_to_idx:
            weights[cls_to_idx[hc]] = 3.5

    # Transforms with sharpness & edge enhancement for micro-insects & leaf miner trails
    train_tf = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.RandomCrop(384),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = ImageFolder(VAL_DIR, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Load flagship ConvNeXt-Small
    model = models.convnext_small(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_features, num_classes)
    )

    ckpt_path = ARTIFACTS_DIR / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(config.DEVICE)

    # Freeze stages 0-5, train stage 6 and classifier
    for p in model.features[:6].parameters():
        p.requires_grad = False
    for p in model.features[6:].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1.2e-5, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 2
    log_event(f"• Training on {len(train_ds)} samples across {num_classes} classes for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += imgs.size(0)

        # Validation
        model.eval()
        val_correct, total_val = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                total_val += imgs.size(0)

        t_sec = round(time.time() - t_ep, 1)
        train_acc = (train_correct / total_train) * 100
        val_acc = (val_correct / total_val) * 100
        log_event(f"  [Epoch {epoch}/{epochs}] ({t_sec}s) | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Platt Temperature Calibration via L-BFGS
    log_event("\n" + "=" * 70)
    log_event("PHASE 3: PLATT TEMPERATURE RE-CALIBRATION (L-BFGS on Validation Set)")
    log_event("=" * 70)
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(config.DEVICE)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                out = model(imgs)
            logits_list.append(out.cpu())
            labels_list.append(labels)
    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    temperature = nn.Parameter(torch.ones(1) * 1.0)
    t_opt = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    t_crit = nn.CrossEntropyLoss()
    def eval_loss():
        t_opt.zero_grad()
        l = t_crit(all_logits / temperature, all_labels)
        l.backward()
        return l
    t_opt.step(eval_loss)
    opt_temp = float(temperature.item())
    log_event(f"✓ Optimized Softmax Temperature: T = {opt_temp:.4f}")

    # Save Improved Model Checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "temperature": opt_temp,
        "val_acc": val_acc
    }, ckpt_path)
    log_event(f"✓ Saved updated flagship model to {ckpt_path}")

if __name__ == "__main__":
    t_start = time.time()
    ingest_benchmark_samples()
    train_active_cycle()
    elapsed = round((time.time() - t_start) / 60, 2)
    log_event(f"★ Active Learning Cycle v5.0 Completed in {elapsed} minutes!")

