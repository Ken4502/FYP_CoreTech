"""
Fast GPU Transfer Fine-Tuning for Expanded Botanical Taxonomy (FloraNet-104+)
Trains with PyTorch AMP, Cosine Annealing, and Class Weighting on NVIDIA RTX 3060.
"""
import sys
import json
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config
from src.model import build_model, set_parameter_requires_grad

def update_class_indices():
    """Scans train directory to update class_indices.json."""
    train_dir = config.TRAIN_DIR
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir() and any(d.glob("*.*"))])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    payload = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "total_classes": len(classes)
    }

    config.CLASS_INDICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Taxonomy] Updated class indices: {len(classes)} active botanical classes registered.")
    return classes, class_to_idx

def load_or_expand_checkpoint(num_classes: int, old_checkpoint_path: Path):
    """Loads existing best_model.pth and expands the classifier head for new classes."""
    model = build_model(num_classes=num_classes, pretrained=True)
    
    if old_checkpoint_path.exists():
        print(f"[Model] Loading existing 90.37% baseline weights from: {old_checkpoint_path.name}")
        checkpoint = torch.load(old_checkpoint_path, map_location="cpu")
        old_state = checkpoint["model_state_dict"]
        old_classes = checkpoint.get("classes", [])

        # Check if classifier weight shape matches
        old_weight = old_state.get("classifier.3.weight")
        old_bias = old_state.get("classifier.3.bias")

        if old_weight is not None and old_weight.shape[0] != num_classes:
            print(f"[Model] Expanding classifier head: {old_weight.shape[0]} classes -> {num_classes} classes.")
            old_out_features, in_features = old_weight.shape
            new_weight = model.classifier[3].weight.data.clone()
            new_bias = model.classifier[3].bias.data.clone()

            # Map matching classes by name
            with open(config.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
                new_idx_map = json.load(f)["class_to_idx"]

            for old_idx, old_name in enumerate(old_classes):
                if old_name in new_idx_map:
                    new_idx = new_idx_map[old_name]
                    new_weight[new_idx] = old_weight[old_idx]
                    if old_bias is not None:
                        new_bias[new_idx] = old_bias[old_idx]

            old_state["classifier.3.weight"] = new_weight
            if old_bias is not None:
                old_state["classifier.3.bias"] = new_bias

        model.load_state_dict(old_state, strict=False)
    else:
        print("[Model] Initializing fresh ImageNet pretrained weights.")

    return model

def run_fast_finetune(epochs: int = 8, batch_size: int = 32, peak_lr: float = 1.5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"[Fine-Tuning] Target Device: {device} (NVIDIA RTX 3060)")
    print(f"[Fine-Tuning] Regimen: {epochs} Epochs, Cosine Annealing, AMP Enabled")
    print("=" * 70)

    classes, class_to_idx = update_class_indices()
    num_classes = len(classes)

    # 1. Transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Datasets
    train_ds = datasets.ImageFolder(str(config.TRAIN_DIR), transform=train_tf)
    val_ds = datasets.ImageFolder(str(config.VAL_DIR), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(config.TEST_DIR), transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"[Data] Train: {len(train_ds)} images | Val: {len(val_ds)} images | Test: {len(test_ds)} images")

    # 3. Model setup
    model = load_or_expand_checkpoint(num_classes, config.BEST_MODEL_PATH)
    model = model.to(device)

    # Class-weighted loss
    counts = Counter([target for _, target in train_ds.samples])
    weights = [len(train_ds) / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    weights = np.clip(weights, 0.5, 3.5)
    class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.08)

    optimizer = AdamW(model.parameters(), lr=peak_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(imgs)
                    loss = criterion(out, lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = out.max(1)
            correct += preds.eq(lbls).sum().item()
            total += lbls.size(0)

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for v_imgs, v_lbls in val_loader:
                v_imgs, v_lbls = v_imgs.to(device), v_lbls.to(device)
                v_out = model(v_imgs)
                _, v_preds = v_out.max(1)
                v_correct += v_preds.eq(v_lbls).sum().item()
                v_total += v_lbls.size(0)

        val_acc = v_correct / v_total
        scheduler.step()
        duration = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": classes
            }, config.BEST_MODEL_PATH)
            print(f"   ⭐ New best model saved! (Top-1: {val_acc*100:.2f}%)", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"🎉 Fine-Tuning Complete! Best Validation Accuracy: {best_val_acc*100:.2f}%", flush=True)
    print(f"Saved Checkpoint: {config.BEST_MODEL_PATH}", flush=True)
    print("=" * 70, flush=True)

    print("\n" + "=" * 70, flush=True)
    print("🔍 VERIFYING PREDICTIONS ON USER'S 3 UPLOADED SPECIMENS:", flush=True)
    print("=" * 70, flush=True)
    from PIL import Image
    test_user_samples = [
        ("Specimen 1 (Columnar Cactus)", config.TEST_DIR / "Columnar Cactus (Pachycereus pringlei)" / "user_specimen_1.jpg"),
        ("Specimen 2 (Golden Pothos)", config.TEST_DIR / "Pothos (Ivy arum)" / "user_specimen_2.webp"),
        ("Specimen 3 (Golden Barrel Cactus)", config.TEST_DIR / "Golden Barrel Cactus (Echinocactus grusonii)" / "user_specimen_3.webp")
    ]
    model.eval()
    with torch.no_grad():
        for label, pth in test_user_samples:
            if pth.exists():
                im = Image.open(pth).convert("RGB")
                tensor = eval_tf(im).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                top_p, top_idx = torch.topk(probs, 5)
                print(f"\n{label}:", flush=True)
                for r, (pr, ix) in enumerate(zip(top_p, top_idx), 1):
                    print(f"  #{r}: {classes[ix.item()]} ({pr.item()*100:.2f}%)", flush=True)

if __name__ == "__main__":
    run_fast_finetune(epochs=6, batch_size=32, peak_lr=1.5e-4)
