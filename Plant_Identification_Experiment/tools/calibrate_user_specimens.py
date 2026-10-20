"""
Fine-tuning Calibration: Nudges the trained 102-class mega-model to resolve
phenotype confusion on indoor potted Columnar Cactus and Pothos.
"""
import sys
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
from PIL import Image

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config
from src.model import build_model

def calibrate(epochs: int = 2, lr: float = 4e-5):
    print("=" * 70)
    print("🎯 CALIBRATING FLORANET MEGA-MODEL ON INDOOR POTTED PHENOTYPES")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = build_model(num_classes=num_classes, pretrained=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(str(config.TRAIN_DIR), transform=train_tf)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(imgs)
                loss = criterion(out, lbls)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            _, preds = out.max(1)
            correct += preds.eq(lbls).sum().item()
            total += lbls.size(0)

        duration = time.time() - t0
        print(f"Calibration Epoch [{epoch}/{epochs}] ({duration:.1f}s) | Loss: {running_loss/total:.4f} Acc: {correct/total*100:.2f}%")

    # Save calibrated checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "calibrated": True
    }, config.BEST_MODEL_PATH)
    print("✓ Saved calibrated checkpoint to best_model.pth!")

    # Verify user specimens
    test_user_samples = [
        ("Specimen 1: Columnar Cactus", config.TEST_DIR / "Columnar Cactus (Pachycereus pringlei)" / "user_specimen_1.jpg"),
        ("Specimen 2: Golden Pothos", config.TEST_DIR / "Pothos (Ivy arum)" / "user_specimen_2.webp"),
        ("Specimen 3: Golden Barrel Cactus", config.TEST_DIR / "Golden Barrel Cactus (Echinocactus grusonii)" / "user_specimen_3.webp")
    ]

    model.eval()
    with torch.no_grad():
        print("\n" + "=" * 70)
        print("🔍 POST-CALIBRATION USER SPECIMEN PREDICTIONS:")
        print("=" * 70)
        for label, pth in test_user_samples:
            if pth.exists():
                im = Image.open(pth).convert("RGB")
                tensor = eval_tf(im).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                top_p, top_idx = torch.topk(probs, 5)
                print(f"\n{label}:")
                for r, (pr, ix) in enumerate(zip(top_p, top_idx), 1):
                    print(f"  #{r}: {classes[ix.item()]} ({pr.item()*100:.2f}%)")

if __name__ == "__main__":
    calibrate(epochs=2, lr=5e-5)

