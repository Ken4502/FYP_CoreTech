"""
CoreTech v2 Phase 4A: Dual-Scale Multi-Tile Attention MIL Surface Pathology Trainer.
Encodes Global Context View + 4 Native-Resolution Surface Tiles per sample through a shared backbone
(ConvNeXt-Base 384px or DINOv2-Base with Registers 392px), composites studio PlantVillage backgrounds
dynamically onto MIT Indoor scenes, and supervises both:
  - 19-Class Surface Pathology & Abiotic Stress (Cosine Head + Logit-Adjusted Focal Loss)
  - 3-Class Surface Super-State [Healthy | Biotic_Infection | Abiotic_Stress]
"""
import os
import sys
import copy
import time
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

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))
from configs.v2_config import (
    PATH_DATA_DIR, NON_PLANT_DIR, ARTIFACTS_DIR, REPORTS_DIR, SURFACE_PATHOLOGY_CLASSES, DEVICE, USE_BF16
)
from src.cosine_models import MultiTileAttentionMIL


def class_to_super_idx(cls_name: str) -> int:
    if cls_name == "Healthy_Foliage":
        return 0
    if cls_name.startswith("Abiotic_"):
        return 2
    return 1


def composite_pv_onto_indoor(rgb: np.ndarray, bg_paths: list) -> np.ndarray:
    """Replaces uniform gray/black PlantVillage studio background with real indoor scenes dynamically."""
    if not bg_paths:
        return rgb
    h, w, _ = rgb.shape
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Detect neutral uniform studio cardboard/shadow typical of PlantVillage
    bg_mask = (hsv[:, :, 1] < 45) & ((hsv[:, :, 2] > 140) | (hsv[:, :, 2] < 38))
    fg_mask = (~bg_mask).astype(np.uint8)
    fg_ratio = float(fg_mask.mean())
    if fg_ratio < 0.08 or fg_ratio > 0.94:
        return rgb
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(fg_mask.astype(np.float32), (5, 5), 0)[:, :, None]

    bg_p = random.choice(bg_paths)
    try:
        bg_bgr = cv2.imread(str(bg_p))
        if bg_bgr is None:
            return rgb
        bg_rgb = cv2.cvtColor(cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        return (rgb.astype(np.float32) * alpha + bg_rgb.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    except Exception:
        return rgb


def extract_surface_bag(rgb: np.ndarray, img_size: int, is_train: bool) -> list:
    """Returns [Global_View, Tile_1, Tile_2, Tile_3, Tile_4] for Multi-Tile MIL."""
    h, w, _ = rgb.shape
    global_view = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # Pad small images cleanly before extracting 4 quadrant/foveal high-res surface tiles
    min_side = min(h, w)
    if min_side < img_size:
        sc = float(img_size * 1.35) / max(1, min_side)
        rgb = cv2.resize(rgb, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_LINEAR)
        h, w, _ = rgb.shape

    tile_sz = min(h, w, max(img_size, int(min(h, w) * 0.52)))
    coords = [
        (max(0, (w - tile_sz) // 2), max(0, (h - tile_sz) // 2)),  # Center fovea
        (0, 0),                                                    # Top-left
        (max(0, w - tile_sz), 0),                                  # Top-right
        (max(0, w // 4), max(0, h - tile_sz)),                     # Lower center
    ]
    tiles = [global_view]
    for x0, y0 in coords:
        if is_train:
            jx = random.randint(-min(18, x0), min(18, max(0, w - tile_sz - x0))) if w > tile_sz else 0
            jy = random.randint(-min(18, y0), min(18, max(0, h - tile_sz - y0))) if h > tile_sz else 0
            x0, y0 = x0 + jx, y0 + jy
        crop = rgb[y0:y0 + tile_sz, x0:x0 + tile_sz]
        tiles.append(cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR))
    return tiles


class SurfaceMILDataset(Dataset):
    def __init__(self, root_dir: Path, img_size: int, classes: list, indoor_bgs: list, is_train: bool = True):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.indoor_bgs = indoor_bgs
        self.is_train = is_train

        self.samples = []
        for c in classes:
            c_dir = self.root_dir / c
            if not c_dir.exists():
                continue
            y_fine = self.class_to_idx[c]
            y_sup = class_to_super_idx(c)
            for f in c_dir.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((str(f), f.name.startswith("pv_"), y_fine, y_sup))

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if is_train:
            # Preserve exact diagnostic colors (no aggressive hue shift that turns rust orange or chlorosis green!)
            self.tile_tf = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.35),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.14, contrast=0.14, saturation=0.10, hue=0.015, p=0.5),
                A.OneOf([A.Sharpen(alpha=(0.1, 0.3)), A.GaussianBlur(blur_limit=3)], p=0.25),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            self.tile_tf = A.Compose([
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, is_pv, y_fine, y_sup = self.samples[idx]
        bgr = cv2.imread(path)
        if bgr is None:
            with Image.open(path) as im:
                rgb = np.array(im.convert("RGB"))
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.is_train and is_pv and random.random() < 0.65:
            rgb = composite_pv_onto_indoor(rgb, self.indoor_bgs)

        bag_imgs = extract_surface_bag(rgb, self.img_size, is_train=self.is_train)
        bag_tensors = torch.stack([self.tile_tf(image=t)["image"] for t in bag_imgs], dim=0)
        return bag_tensors, y_fine, y_sup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["convnext", "dinov2"], default="convnext")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    cfg = {
        "convnext": {"timm": "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384", "size": 384, "batch": 3, "accum": 8, "lr": 4e-5, "ckpt": "pathology_convnext_mil.pth"},
        "dinov2": {"timm": "vit_base_patch14_reg4_dinov2.lvd142m", "size": 392, "batch": 4, "accum": 6, "lr": 2.5e-5, "ckpt": "pathology_dinov2_mil.pth"},
    }[args.arch]

    indoor_dir = NON_PLANT_DIR / "mit_indoor"
    indoor_bgs = list(indoor_dir.glob("*.jpg")) if indoor_dir.exists() else []

    train_ds = SurfaceMILDataset(PATH_DATA_DIR / "train", cfg["size"], SURFACE_PATHOLOGY_CLASSES, indoor_bgs, is_train=True)
    val_ds = SurfaceMILDataset(PATH_DATA_DIR / "val", cfg["size"], SURFACE_PATHOLOGY_CLASSES, indoor_bgs, is_train=False)

    counts_map = Counter(s[2] for s in train_ds.samples)
    total = max(1, len(train_ds))
    log_prior = torch.tensor([math.log(max(counts_map.get(i, 1), 1) / total) for i in range(len(SURFACE_PATHOLOGY_CLASSES))], dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=6, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    log_path = REPORTS_DIR / f"train_pathology_{args.arch}.log"
    def log_both(msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log_both(f"=== Training Stage 3A Multi-Tile Surface MIL ({args.arch}, {cfg['size']}px) | Train={len(train_ds)} Val={len(val_ds)} ===")

    model = MultiTileAttentionMIL(cfg["timm"], num_classes=len(SURFACE_PATHOLOGY_CLASSES), pretrained=True).to(DEVICE)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad = False

    head_params = list(model.attn_V.parameters()) + list(model.attn_U.parameters()) + list(model.attn_w.parameters()) + list(model.fusion.parameters()) + list(model.fine_head.parameters()) + list(model.super_head.parameters())
    backbone_params = list(model.backbone.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg["lr"]},
        {"params": head_params, "lr": cfg["lr"] * 8.0},
    ], weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(not USE_BF16 and torch.cuda.is_available()))

    super_crit = nn.CrossEntropyLoss(label_smoothing=0.04)
    best_acc = 0.0
    out_ckpt = ARTIFACTS_DIR / cfg["ckpt"]

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (bag, y_fine, y_sup) in enumerate(train_loader, 1):
            bag = bag.to(DEVICE, non_blocking=True)
            y_fine = y_fine.to(DEVICE, non_blocking=True)
            y_sup = y_sup.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                fine_logits, super_logits, _, _ = model(bag)
                adj_fine = fine_logits + 0.75 * log_prior
                loss_fine = F.cross_entropy(adj_fine, y_fine, label_smoothing=0.05)
                loss_sup = super_crit(super_logits, y_sup)
                loss = (loss_fine + 0.35 * loss_sup) / cfg["accum"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg["accum"] == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for ep_p, mp in zip(ema.parameters(), model.parameters()):
                        ep_p.data.mul_(0.999).add_(mp.data, alpha=0.001)
                    for eb, mb in zip(ema.buffers(), model.buffers()):
                        eb.copy_(mb)

            running_loss += loss.item() * cfg["accum"]
            steps += 1
            if step % 600 == 0 or step == len(train_loader):
                vram_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
                elapsed = time.time() - t0
                print(f"  [Ep {ep:02d}] Step {step}/{len(train_loader)} | AvgLoss: {running_loss/steps:.4f} | {elapsed:.0f}s | VRAM: {vram_mb:.0f}MB", flush=True)

        scheduler.step()

        # Validation
        ema.eval()
        val_c1, val_c3, sup_c, val_tot = 0, 0, 0, 0
        with torch.no_grad():
            for bag, y_fine, y_sup in val_loader:
                bag = bag.to(DEVICE, non_blocking=True)
                y_fine = y_fine.to(DEVICE, non_blocking=True)
                y_sup = y_sup.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                    fine_logits, super_logits, _, _ = ema(bag)
                val_tot += y_fine.size(0)
                val_c1 += (fine_logits.argmax(dim=-1) == y_fine).sum().item()
                sup_c += (super_logits.argmax(dim=-1) == y_sup).sum().item()
                top3 = fine_logits.topk(3, dim=-1).indices
                val_c3 += sum(y_fine[i].item() in top3[i].tolist() for i in range(y_fine.size(0)))

        acc1 = (val_c1 / max(1, val_tot)) * 100.0
        acc3 = (val_c3 / max(1, val_tot)) * 100.0
        sup_acc = (sup_c / max(1, val_tot)) * 100.0
        sec = round(time.time() - t0, 1)
        log_both(f"Epoch [{ep:02d}/{args.epochs:02d}] ({sec}s) | Loss: {running_loss/max(1,steps):.4f} | Fine Top-1: {acc1:.2f}% | Top-3: {acc3:.2f}% | Healthy/Biotic/Abiotic: {sup_acc:.2f}%")

        if acc1 > best_acc:
            best_acc = acc1
            torch.save({
                "arch": args.arch,
                "timm_name": cfg["timm"],
                "img_size": cfg["size"],
                "classes": SURFACE_PATHOLOGY_CLASSES,
                "val_top1": best_acc,
                "val_super_acc": sup_acc,
                "model_state_dict": ema.state_dict(),
            }, out_ckpt)
            log_both(f"  [★] Saved best pathology MIL ({best_acc:.2f}%) -> {out_ckpt}")


if __name__ == "__main__":
    main()
