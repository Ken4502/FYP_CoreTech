"""
CoreTech v2 Phase 3: Foundation Botanical Identification Trainer.
Supports training 3 heterogeneous Cosine-Normalized vision architectures on RTX 3060 (BF16 AMP):
  1. --arch bioclip2 : BioCLIP-2 ViT-L/14 (TreeOfLife-200M botanical prior, 224px)
  2. --arch eva02    : EVA-02 Base MIM+IN22k (448px high-res fine-grained geometry)
  3. --arch convnext : ConvNeXt-Base CLIP-LAION2B+IN12k (384px local epidermal texture)
"""
import os
import sys
import copy
import time
import json
import math
import random
import argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

ImageFile.LOAD_TRUNCATED_IMAGES = True

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))
from configs.v2_config import ID_DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR, DEVICE, USE_BF16
from src.cosine_models import BioCLIP2Classifier, TimmCosineClassifier


import psutil
try:
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

_nvml_inited = False
_nvml_handle = None

def get_gpu_temperature():
    global _nvml_inited, _nvml_handle
    try:
        import pynvml
        if not _nvml_inited:
            pynvml.nvmlInit()
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            _nvml_inited = True
        return pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        return None

def get_cpu_temperature():
    try:
        import wmi
        for ns in ['root\\LibreHardwareMonitor', 'root\\OpenHardwareMonitor']:
            try:
                w = wmi.WMI(namespace=ns)
                for s in w.Sensor():
                    if s.SensorType == 'Temperature' and ('CPU' in s.Name or 'Package' in s.Name):
                        val = float(s.Value)
                        if val > 0:
                            return val
            except Exception:
                continue
    except Exception:
        pass
    return None

def thermal_guard_check(step: int, log_fn):
    """Checks GPU and CPU thermals. Pauses if GPU >= 76C or CPU >= 75C until cooled to <= 71C."""
    gpu_t = get_gpu_temperature()
    cpu_t = get_cpu_temperature()
    overheated = False
    reasons = []
    if gpu_t is not None and gpu_t >= 76:
        overheated = True
        reasons.append(f"GPU: {gpu_t}°C (ceiling 76°C)")
    if cpu_t is not None and cpu_t >= 75:
        overheated = True
        reasons.append(f"CPU: {cpu_t:.1f}°C (ceiling 75°C)")
    if overheated:
        log_fn(f"  [⚠️ Thermal Guard] Temperature threshold reached ({', '.join(reasons)}). Resting for 15s...")
        time.sleep(15)
        for _ in range(6):
            g2 = get_gpu_temperature()
            c2 = get_cpu_temperature()
            if (g2 is None or g2 <= 71) and (c2 is None or c2 <= 70):
                break
            time.sleep(5)
        log_fn(f"  [✓ Thermal Guard] Cooled to GPU: {get_gpu_temperature()}°C. Resuming...")


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9993):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad = False

    def update(self, model: nn.Module):
        with torch.no_grad():
            for ep, mp in zip(self.ema.parameters(), model.parameters()):
                ep.data.mul_(self.decay).add_(mp.data, alpha=1.0 - self.decay)
            for eb, mb in zip(self.ema.buffers(), model.buffers()):
                eb.copy_(mb)


class LogitAdjustedCE(nn.Module):
    def __init__(self, class_counts: list, tau: float = 0.85, label_smoothing: float = 0.06):
        super().__init__()
        total = max(1, sum(class_counts))
        freq = [max(c, 1) / total for c in class_counts]
        log_prior = torch.tensor([math.log(f) for f in freq], dtype=torch.float32)
        self.register_buffer("log_prior", log_prior)
        self.tau = tau
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits + self.tau * self.log_prior.to(logits.device), targets)


class AlbumentationsFolderDataset(Dataset):
    def __init__(self, root_dir: Path, transform: A.Compose, class_to_idx: dict = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        if class_to_idx is None:
            classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        self.classes = [k for k, _ in sorted(self.class_to_idx.items(), key=lambda x: x[1])]

        self.samples = []
        for cls_name, idx in self.class_to_idx.items():
            c_dir = self.root_dir / cls_name
            if not c_dir.exists():
                continue
            for f in c_dir.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((str(f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            with Image.open(path) as im:
                rgb = np.array(im.convert("RGB"))
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = self.transform(image=rgb)["image"]
        return out, target


def get_norm_stats(arch: str):
    if arch == "bioclip2":
        return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(img_size: int, mean, std):
    train_tf = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.65, 1.0), ratio=(0.8, 1.25)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Affine(rotate=(-25, 25), scale=(0.9, 1.1), shear=(-8, 8), p=0.55),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04, p=0.6),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.ImageCompression(quality_range=(55, 95)),
        ], p=0.25),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, int(img_size * 0.16)), hole_width_range=(16, int(img_size * 0.16)), p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.LongestMaxSize(max_size=int(img_size * 1.12)),
        A.PadIfNeeded(min_height=int(img_size * 1.12), min_width=int(img_size * 1.12), border_mode=cv2.BORDER_CONSTANT, fill=128),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return train_tf, val_tf


def apply_mixup_cutmix(x, y, num_classes: int, mixup_alpha=0.25, cutmix_alpha=0.8):
    r = random.random()
    b = x.size(0)
    y_oh = F.one_hot(y, num_classes=num_classes).float()
    perm = torch.randperm(b, device=x.device)

    if r < 0.25:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x_mix = lam * x + (1 - lam) * x[perm]
        y_mix = lam * y_oh + (1 - lam) * y_oh[perm]
        return x_mix, y_mix
    elif r < 0.55:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        _, _, h, w = x.shape
        cut_rat = math.sqrt(1.0 - lam)
        cw, ch = int(w * cut_rat), int(h * cut_rat)
        cx, cy = random.randint(0, w), random.randint(0, h)
        x1, x2 = max(0, cx - cw // 2), min(w, cx + cw // 2)
        y1, y2 = max(0, cy - ch // 2), min(h, cy + ch // 2)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam_adj = 1.0 - ((x2 - x1) * (y2 - y1) / float(w * h))
        y_mix = lam_adj * y_oh + (1 - lam_adj) * y_oh[perm]
        return x_mix, y_mix
    return x, y_oh


def evaluate_model(model: nn.Module, loader: DataLoader, amp_dtype, log_prior: torch.Tensor = None):
    model.eval()
    correct1, correct5, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits, _ = model(x)
            total += y.size(0)
            preds1 = logits.argmax(dim=-1)
            correct1 += (preds1 == y).sum().item()
            top5 = logits.topk(5, dim=-1).indices
            correct5 += sum(y[i].item() in top5[i].tolist() for i in range(y.size(0)))
    return (correct1 / max(1, total)) * 100.0, (correct5 / max(1, total)) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["bioclip2", "eva02", "convnext"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--accum_steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=4e-5)
    args = ap.parse_args()

    arch_cfg = {
        "bioclip2": {"size": 224, "batch": 32, "accum": 1, "lr": 3.0e-5, "epochs": 15, "ckpt": "flora_bioclip2.pth"},
        "eva02": {"size": 448, "batch": 10, "accum": 3, "lr": 3.0e-5, "epochs": 14, "ckpt": "flora_eva02.pth"},
        "convnext": {"size": 384, "batch": 16, "accum": 2, "lr": 6.0e-5, "epochs": 18, "ckpt": "flora_convnext.pth"},
    }[args.arch]

    img_size = arch_cfg["size"]
    bs = arch_cfg["batch"]
    accum = arch_cfg["accum"]
    base_lr = args.lr if "--lr" in sys.argv else arch_cfg["lr"]
    epochs = args.epochs if "--epochs" in sys.argv else arch_cfg["epochs"]

    log_path = REPORTS_DIR / f"train_flora_{args.arch}.log"
    def log_both(msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    train_dir = ID_DATA_DIR / "train"
    val_dir = ID_DATA_DIR / "val"
    mean, std = get_norm_stats(args.arch)
    train_tf, val_tf = build_transforms(img_size, mean, std)

    train_ds = AlbumentationsFolderDataset(train_dir, train_tf)
    val_ds = AlbumentationsFolderDataset(val_dir, val_tf, class_to_idx=train_ds.class_to_idx)
    num_classes = len(train_ds.classes)

    counts_map = Counter(y for _, y in train_ds.samples)
    class_counts = [counts_map[i] for i in range(num_classes)]

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    log_both(f"=== Training Stage 2 Flora Model: {args.arch} ({img_size}px) | {num_classes} classes | Train={len(train_ds)} Val={len(val_ds)} ===")

    if args.arch == "bioclip2":
        model = BioCLIP2Classifier(num_classes=num_classes, unfreeze_last_blocks=12).to(DEVICE)
        head_params = list(model.head.parameters())
        backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    elif args.arch == "eva02":
        model = TimmCosineClassifier("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", num_classes=num_classes).to(DEVICE)
        head_params = list(model.head.parameters())
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    else:
        model = TimmCosineClassifier("convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384", num_classes=num_classes).to(DEVICE)
        head_params = list(model.head.parameters())
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]

    amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(not USE_BF16 and torch.cuda.is_available()))

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params, "lr": base_lr * 12.0},
    ], weight_decay=0.02)

    warmup_epochs = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
    ema = ModelEMA(model, decay=0.9993)

    # Soft cross entropy with logit adjustment for mixed targets
    total = max(1, sum(class_counts))
    log_prior = torch.tensor([math.log(max(c, 1) / total) for c in class_counts], dtype=torch.float32, device=DEVICE)

    best_val1 = 0.0
    out_ckpt = ARTIFACTS_DIR / arch_cfg["ckpt"]

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        if ep <= warmup_epochs:
            wu_scale = ep / float(warmup_epochs)
            optimizer.param_groups[0]["lr"] = base_lr * wu_scale
            optimizer.param_groups[1]["lr"] = base_lr * 12.0 * wu_scale

        running_loss = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            x_m, y_soft = apply_mixup_cutmix(x, y, num_classes=num_classes)
            # Smooth labels slightly
            y_soft = y_soft * 0.94 + 0.06 / num_classes

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits, _ = model(x_m)
                adj_logits = logits + 0.80 * log_prior
                loss = -(y_soft * F.log_softmax(adj_logits, dim=-1)).sum(dim=-1).mean()
                loss_scaled = loss / accum

            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if step % accum == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

            running_loss += loss.item()
            steps += 1
            if step % 250 == 0:
                thermal_guard_check(step, log_both)
            if step % 1000 == 0 or step == len(train_loader):
                vram_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
                elapsed = time.time() - t0
                print(f"  [Ep {ep:02d}] Step {step}/{len(train_loader)} | AvgLoss: {running_loss/steps:.4f} | {elapsed:.0f}s | VRAM: {vram_mb:.0f}MB", flush=True)

        if ep > warmup_epochs:
            scheduler.step()

        val1, val5 = evaluate_model(ema.ema, val_loader, amp_dtype)
        sec = round(time.time() - t0, 1)
        avg_loss = running_loss / max(1, steps)
        log_both(f"Epoch [{ep:02d}/{epochs:02d}] ({sec}s) | Loss: {avg_loss:.4f} | EMA Val Top-1: {val1:.2f}% | Val Top-5: {val5:.2f}%")

        if val1 > best_val1:
            best_val1 = val1
            torch.save({
                "arch": args.arch,
                "img_size": img_size,
                "classes": train_ds.classes,
                "val_top1": best_val1,
                "val_top5": val5,
                "model_state_dict": ema.ema.state_dict(),
            }, out_ckpt)
            log_both(f"  [★] Saved new best {args.arch} checkpoint ({best_val1:.2f}%) -> {out_ckpt}")

        # Inter-epoch Thermal Rest (45 seconds) to allow PC fans to cool heatsinks
        if ep < epochs:
            g_cur = get_gpu_temperature()
            log_both(f"  [❄️ Thermal Rest] Epoch {ep} complete. Fans cooling heatsinks for 45s (current GPU: {g_cur}°C)...")
            time.sleep(45)
            log_both(f"  [❄️ Thermal Rest] Cooled to GPU: {get_gpu_temperature()}°C. Resuming Epoch {ep+1}...")

    log_both(f"Training completed for {args.arch}. Best EMA Val Top-1: {best_val1:.2f}%")


if __name__ == "__main__":
    main()
