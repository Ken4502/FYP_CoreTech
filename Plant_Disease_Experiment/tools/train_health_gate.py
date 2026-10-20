"""
PhytoCare Stage-1: Binary Foliar Health & Variegation Gate
Trains a lightweight binary classifier [Healthy vs Pathological] to protect against false positives
on healthy variegated foliage, spines, ribs, and normal leaf growth.
Execution time: ~60-90 seconds on NVIDIA RTX 3060.
Checkpoint: artifacts/health_gate.pth
"""

import sys
import os
import time
import random
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config

random.seed(42)
torch.manual_seed(42)

HEALTH_GATE_CKPT = EXPERIMENT_ROOT / "artifacts" / "health_gate.pth"

class BinaryHealthDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (path, label: 0 for healthy, 1 for disease)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fallback to black square on rare IO error
            img = Image.new("RGB", (384, 384), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

def prepare_data_samples():
    train_dir = EXPERIMENT_ROOT / "data" / "train"
    val_dir = EXPERIMENT_ROOT / "data" / "val"
    flora_dir = EXPERIMENT_ROOT.parent / "Plant_Identification_Experiment" / "data" / "train"

    healthy_train = []
    disease_train = []

    for d in train_dir.iterdir():
        if not d.is_dir():
            continue
        imgs = list(d.glob("*.jpg"))
        if "Healthy" in d.name:
            healthy_train.extend(imgs)
        else:
            disease_train.extend(imgs)

    # Add diverse houseplant healthy leaves from FloraNet (Monstera, Ficus, Epipremnum, etc.)
    if flora_dir.exists():
        houseplant_targets = ["Monstera", "Ficus", "Pothos", "Calathea", "Philodendron", "Dracaena", "Syngonium"]
        for d in flora_dir.iterdir():
            if d.is_dir() and any(ht.lower() in d.name.lower() for ht in houseplant_targets):
                f_imgs = list(d.glob("*.jpg"))
                healthy_train.extend(random.sample(f_imgs, min(100, len(f_imgs))))

    # Balanced split
    n_samples = min(len(healthy_train), 3000)
    sampled_healthy = random.sample(healthy_train, n_samples)
    sampled_disease = random.sample(disease_train, n_samples)

    # 85% train, 15% val
    n_train = int(n_samples * 0.85)
    train_samples = [(p, 0) for p in sampled_healthy[:n_train]] + [(p, 1) for p in sampled_disease[:n_train]]
    val_samples = [(p, 0) for p in sampled_healthy[n_train:]] + [(p, 1) for p in sampled_disease[n_train:]]

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"[HealthGate] Prepared {len(train_samples)} training samples ({n_train} Healthy, {n_train} Disease)")
    print(f"[HealthGate] Prepared {len(val_samples)} validation samples ({len(val_samples)//2} Healthy, {len(val_samples)//2} Disease)")
    return train_samples, val_samples

def train_health_gate():
    t0 = time.time()
    print("=" * 60)
    print("TRAINING STAGE-1 BINARY FOLIAR HEALTH & VARIEGATION GATE")
    print("=" * 60)

    train_samples, val_samples = prepare_data_samples()

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(BinaryHealthDataset(train_samples, train_tf), batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(BinaryHealthDataset(val_samples, val_tf), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # Lightweight ConvNeXt-Tiny / MobileNet backbone for speed (<5ms inference)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_feat = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feat, 2)  # [0: Healthy, 1: Pathological]
    model = model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_acc = 0.0

    for epoch in range(1, 5):
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

        scheduler.step()
        train_acc = (train_correct / total_train) * 100

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

        val_acc = (val_correct / total_val) * 100
        ep_sec = round(time.time() - t_ep, 1)
        print(f"Epoch [{epoch}/4] ({ep_sec}s) | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": ["Healthy_Foliage", "Pathological_Foliage"],
                "best_val_acc": best_val_acc
            }, HEALTH_GATE_CKPT)

    duration = round(time.time() - t0, 1)
    print("=" * 60)
    print(f"HEALTH GATE TRAINING COMPLETE in {duration}s!")
    print(f"Peak Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint Saved: {HEALTH_GATE_CKPT}")
    print("=" * 60)

if __name__ == "__main__":
    train_health_gate()

