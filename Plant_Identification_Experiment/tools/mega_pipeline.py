"""
FloraNet Mega-Pipeline: 102-Class Maximum iNaturalist Expansion & Model Training
End-to-end execution:
  Phase 1: High-Volume Multithreaded Specimen Harvester (Target: 350 images/class)
  Phase 2: Taxonomy Indexing & Dataset Verification
  Phase 3: MobileNetV3-Large Training on NVIDIA RTX 3060 (AMP + Cosine Annealing)
  Phase 4: Test Benchmark Evaluation & User Specimen Verification
"""
import sys
import os
import json
import time
import shutil
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config
from src.model import build_model
from tools.harvester import BOTANICAL_TAXON_MAP

LOG_FILE = EXPERIMENT_ROOT / "reports" / "mega_pipeline.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.array(resized, dtype=np.int32)
    diff = arr[:, :-1] > arr[:, 1:]
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")

def query_inaturalist_multi_page(taxon_name: str, target_count: int):
    """Fetches candidate observations across multiple pages to reach target count."""
    encoded = urllib.parse.quote(taxon_name)
    records = []
    seen = set()

    for grade in ["research", "any"]:
        if len(records) >= target_count * 2:
            break
        grade_param = "&quality_grade=research" if grade == "research" else ""
        # Fetch up to 3 pages of 100 observations
        for page in range(1, 4):
            if len(records) >= target_count * 2:
                break
            url = (
                f"https://api.inaturalist.org/v1/observations?"
                f"taxon_name={encoded}{grade_param}&photos=true&per_page=100&page={page}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "FloraResearchMegaBot/1.0 (FYP Project)"})
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    results = data.get("results", [])
                    if not results:
                        break
                    for item in results:
                        for p in item.get("photos", []):
                            orig_url = p.get("url", "")
                            if orig_url and orig_url not in seen:
                                seen.add(orig_url)
                                high_res = orig_url.replace("/square.", "/medium.").replace("/small.", "/medium.")
                                records.append({
                                    "obs_id": item.get("id"),
                                    "url": high_res,
                                    "taxon": taxon_name
                                })
                                if len(records) >= target_count * 2:
                                    break
                time.sleep(0.4)  # Polite pause between pages
            except Exception as e:
                print(f"   [Query Notice]: {taxon_name} (p.{page}): {e}", flush=True)
                break
    return records

def download_candidate(cand, min_dim: int = 224):
    url = cand["url"]
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        import io
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        if w < min_dim or h < min_dim:
            return None
        return img, cand["obs_id"]
    except Exception:
        return None

def harvest_mega_dataset(target_per_class: int = 350):
    print("\n" + "=" * 75)
    print(f"📥 [PHASE 1/4] HIGH-VOLUME iNATURALIST HARVEST (TARGET: {target_per_class} PER CLASS)")
    print(f"Tracking: 102 Botanical Classes | Thread Workers: 8")
    print("=" * 75)

    data_dir = config.DATA_DIR
    classes = sorted(list(BOTANICAL_TAXON_MAP.keys()))
    total_added = 0

    for idx, class_name in enumerate(classes, 1):
        scientific_name = BOTANICAL_TAXON_MAP[class_name]
        train_dir = data_dir / "train" / class_name
        val_dir = data_dir / "val" / class_name
        test_dir = data_dir / "test" / class_name

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        existing_train = len(list(train_dir.glob("*.*")))
        existing_val = len(list(val_dir.glob("*.*")))
        existing_test = len(list(test_dir.glob("*.*")))
        current_total = existing_train + existing_val + existing_test

        if current_total >= target_per_class:
            print(f"[{idx:03d}/{len(classes)}] '{class_name}': {current_total} specimens (Quota Satisfied).", flush=True)
            continue

        needed = target_per_class - current_total
        print(f"\n[{idx:03d}/{len(classes)}] '{class_name}' [{scientific_name}]")
        print(f"   Current: {current_total} | Target: {target_per_class} | Need: +{needed} specimens...", flush=True)

        # Existing hashes to prevent duplicates
        existing_hashes = set()
        for d in [train_dir, val_dir, test_dir]:
            for f in list(d.glob("*.*"))[:100]:
                try:
                    with Image.open(f) as im:
                        existing_hashes.add(compute_dhash(im))
                except Exception:
                    pass

        candidates = query_inaturalist_multi_page(scientific_name, target_count=needed)
        print(f"   Retrieved {len(candidates)} observation candidates. Downloading in parallel...", flush=True)

        downloaded = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(download_candidate, cand) for cand in candidates]
            for f in as_completed(futures):
                res = f.result()
                if res is None:
                    continue
                img, obs_id = res
                h = compute_dhash(img)
                if any(hamming_distance(h, eh) <= 3 for eh in existing_hashes):
                    continue
                existing_hashes.add(h)
                downloaded.append((img, obs_id))
                if len(downloaded) >= needed:
                    break

        n_new = len(downloaded)
        n_val = max(1, int(n_new * 0.15))
        n_test = max(1, int(n_new * 0.15))
        n_train = n_new - n_val - n_test

        splits = (
            [(img, obs, train_dir) for img, obs in downloaded[:n_train]] +
            [(img, obs, val_dir) for img, obs in downloaded[n_train:n_train + n_val]] +
            [(img, obs, test_dir) for img, obs in downloaded[n_train + n_val:]]
        )

        for img, obs, out_dir in splits:
            fn = f"mega_{obs}.jpg"
            img.save(out_dir / fn, "JPEG", quality=92)

        total_added += n_new
        print(f"   ✓ Added {n_new} specimens ({n_train} train, {n_val} val, {n_test} test). Total now: {current_total + n_new}", flush=True)
        time.sleep(0.8)  # Polite pause between species queries

    # Ensure user-provided photos remain in test set
    user_images = [
        ("Columnar Cactus (Pachycereus pringlei)", r"C:/Users/User/.gemini/antigravity/brain/d97fd749-7a53-48bf-99ed-b47c211057f2/.user_uploaded/media_1792003307823_ab7e5f6b.jpg", "user_specimen_1.jpg"),
        ("Pothos (Ivy arum)", r"C:/Users/User/.gemini/antigravity/brain/d97fd749-7a53-48bf-99ed-b47c211057f2/.user_uploaded/media_1792003307872_0d306914.webp", "user_specimen_2.webp"),
        ("Golden Barrel Cactus (Echinocactus grusonii)", r"C:/Users/User/.gemini/antigravity/brain/d97fd749-7a53-48bf-99ed-b47c211057f2/.user_uploaded/media_1792003307885_82be0d60.webp", "user_specimen_3.webp")
    ]
    for class_name, src_path, dst_name in user_images:
        test_dir = data_dir / "test" / class_name
        test_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, test_dir / dst_name)

    print(f"\n✅ Phase 1 Harvesting Complete! Total new specimens harvested: {total_added}", flush=True)

def update_taxonomy():
    print("\n" + "=" * 75)
    print("🔄 [PHASE 2/4] TAXONOMY INDEXING & SYNCHRONIZATION")
    print("=" * 75)
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

    print(f"✓ Registered {len(classes)} botanical species in class_indices.json.", flush=True)
    return classes, class_to_idx

def train_mega_model(epochs: int = 25, batch_size: int = 32, lr: float = 3e-4):
    print("\n" + "=" * 75)
    print("🔥 [PHASE 3/4] TRAINING FLORANET MEGA-MODEL ON NVIDIA RTX 3060")
    print(f"Hyperparameters: {epochs} Epochs | Batch Size: {batch_size} | Peak LR: {lr:.1e} | AMP: True")
    print("=" * 75)

    classes, class_to_idx = update_taxonomy()
    num_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image Transforms with RandAugment
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.2), value="random")
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(str(config.TRAIN_DIR), transform=train_tf)
    val_ds = datasets.ImageFolder(str(config.VAL_DIR), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(config.TEST_DIR), transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset Counts -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}", flush=True)

    # Class weights for balanced loss
    counts = Counter([target for _, target in train_ds.samples])
    weights = [len(train_ds) / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    weights = np.clip(weights, 0.5, 3.5)
    class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.08)

    # Build model (MobileNetV3-Large)
    model = build_model(num_classes=num_classes, pretrained=True).to(device)

    # Scheduler: 3 Warmup Epochs + Cosine Annealing
    warmup_epochs = 3
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

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
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for v_imgs, v_lbls in val_loader:
                v_imgs, v_lbls = v_imgs.to(device), v_lbls.to(device)
                v_out = model(v_imgs)
                l = criterion(v_out, v_lbls)
                v_loss += l.item() * v_imgs.size(0)
                _, v_preds = v_out.max(1)
                v_correct += v_preds.eq(v_lbls).sum().item()
                v_total += v_lbls.size(0)

        val_acc = v_correct / v_total
        val_loss = v_loss / v_total
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        duration = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.6f}",
            flush=True
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": classes,
                "history": history
            }, config.BEST_MODEL_PATH)
            print(f"   ⭐ Saved new best mega-model! (Val Top-1: {val_acc*100:.2f}%)", flush=True)

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot([a * 100 for a in history["train_acc"]], label="Train Acc %")
    plt.plot([a * 100 for a in history["val_acc"]], label="Val Acc %")
    plt.title("Accuracy Curves")
    plt.legend()
    curve_path = EXPERIMENT_ROOT / "reports" / "mega_training_curves.png"
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"✓ Saved training curves to: {curve_path}", flush=True)

    return model, classes, eval_tf, test_loader, device

def evaluate_mega_model(model, classes, eval_tf, test_loader, device):
    print("\n" + "=" * 75)
    print("📊 [PHASE 4/4] COMPREHENSIVE TEST BENCHMARK & SPECIMEN EVALUATION")
    print("=" * 75)

    # Load best checkpoint
    ckpt = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            total += lbls.size(0)
            _, top1_preds = out.max(1)
            top1_correct += top1_preds.eq(lbls).sum().item()
            _, top5_preds = out.topk(min(5, out.size(1)), dim=1)
            for i in range(lbls.size(0)):
                if lbls[i] in top5_preds[i]:
                    top5_correct += 1

    top1_acc = (top1_correct / total) * 100
    top5_acc = (top5_correct / total) * 100
    print(f"Mega-Model Test Benchmark ({total} test images across {len(classes)} classes):", flush=True)
    print(f"  • Top-1 Accuracy: {top1_acc:.2f}%", flush=True)
    print(f"  • Top-5 Accuracy: {top5_acc:.2f}%", flush=True)

    print("\n" + "=" * 70)
    print("🔍 VERIFYING TARGET SPECIMEN PREDICTIONS (USER-UPLOADED PLANTS):")
    print("=" * 70)
    test_user_samples = [
        ("Specimen 1: Columnar Cactus", config.TEST_DIR / "Columnar Cactus (Pachycereus pringlei)" / "user_specimen_1.jpg"),
        ("Specimen 2: Golden Pothos", config.TEST_DIR / "Pothos (Ivy arum)" / "user_specimen_2.webp"),
        ("Specimen 3: Golden Barrel Cactus", config.TEST_DIR / "Golden Barrel Cactus (Echinocactus grusonii)" / "user_specimen_3.webp")
    ]

    for label, pth in test_user_samples:
        if pth.exists():
            im = Image.open(pth).convert("RGB")
            tensor = eval_tf(im).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                top_p, top_idx = torch.topk(probs, 5)
                print(f"\n{label}:", flush=True)
                for r, (pr, ix) in enumerate(zip(top_p, top_idx), 1):
                    print(f"  #{r}: {classes[ix.item()]} ({pr.item()*100:.2f}%)", flush=True)

    print("\n" + "=" * 75)
    print("🏆 FLORANET MEGA-PIPELINE FULLY COMPLETE!")
    print(f"Best Model Artifact: {config.BEST_MODEL_PATH}")
    print("=" * 75, flush=True)

def main():
    sys.stdout = TeeLogger(LOG_FILE)
    sys.stderr = sys.stdout
    t_start = time.time()
    print("=" * 75)
    print("🌿 STARTING FLORANET 102-CLASS MEGA-PIPELINE")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: 350 Images per Class across All 102 Botanical Species")
    print(f"Log: {LOG_FILE}")
    print("=" * 75, flush=True)

    # 1. Harvest
    harvest_mega_dataset(target_per_class=350)

    # 2. Train
    model, classes, eval_tf, test_loader, device = train_mega_model(epochs=25, batch_size=32, lr=3e-4)

    # 3. Evaluate
    evaluate_mega_model(model, classes, eval_tf, test_loader, device)

    total_hrs = (time.time() - t_start) / 3600
    print(f"\nAll operations finished successfully in {total_hrs:.2f} hours.")

if __name__ == "__main__":
    main()

