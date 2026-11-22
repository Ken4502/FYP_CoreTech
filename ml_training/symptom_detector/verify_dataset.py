"""
UrbanGreen CoreTech: Automated Pre-Flight Gate & Verification for Symptom Dataset.
Audits image counts, class distribution, background negatives, Roboflow twin leakage,
and bounding-box area percentages; outputs visual debug samples.

Usage:
  python verify_dataset.py [--dataset_dir ml_training/symptom_detector/datasets/symptom_dataset]
"""
import sys
import random
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TARGET_CLASSES = {
    0: "yellowing_chlorosis",
    1: "crispy_brown_edges",
    2: "dark_spots_necrosis",
    3: "white_powdery_residue",
    4: "chewed_holes_stippling",
    5: "healthy_leaf",
}

CLASS_COLORS = {
    0: (234, 179, 8),    # Yellow
    1: (217, 119, 6),    # Brown/Amber
    2: (220, 38, 38),    # Crimson Red
    3: (6, 182, 212),    # Cyan
    4: (147, 51, 234),   # Purple
    5: (16, 185, 129),   # Green
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_base_prefix(filename: str) -> str:
    """Extracts base filename prefix before .rf. or _rf_ to check twin leakage."""
    stem = Path(filename).stem
    if ".rf." in stem:
        return stem.split(".rf.")[0]
    match = re.split(r"[\._]rf[\._]", stem, maxsplit=1)
    # Strip any balancing prefix like bal_c0_d1_
    cleaned = re.sub(r"^bal_c\d+_d\d+_", "", match[0])
    return cleaned


def analyze_split(
    img_dir: Path,
    lbl_dir: Path
) -> Tuple[int, int, Dict[int, int], Dict[int, List[float]], Set[str]]:
    """
    Analyzes images, empty labels (negatives), boxes per class, and box areas.
    Returns: (total_images, negative_images, box_counts, box_areas_pct, base_prefixes)
    """
    if not img_dir.exists():
        return 0, 0, {i: 0 for i in TARGET_CLASSES}, {i: [] for i in TARGET_CLASSES}, set()

    img_files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    total_imgs = len(img_files)
    neg_imgs = 0
    box_counts = {i: 0 for i in TARGET_CLASSES}
    box_areas = {i: [] for i in TARGET_CLASSES}
    base_prefixes = set()

    for img_p in img_files:
        base_prefixes.add(get_base_prefix(img_p.name))
        lbl_p = lbl_dir / f"{img_p.stem}.txt"

        if not lbl_p.exists():
            neg_imgs += 1
            continue

        with open(lbl_p, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            neg_imgs += 1
            continue

        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    cid = int(parts[0])
                    bw = float(parts[3])
                    bh = float(parts[4])
                    area_pct = bw * bh * 100.0  # Normalized area %
                    if cid in TARGET_CLASSES:
                        box_counts[cid] += 1
                        box_areas[cid].append(area_pct)
                except ValueError:
                    continue

    return total_imgs, neg_imgs, box_counts, box_areas, base_prefixes


def generate_debug_samples(
    img_dir: Path,
    lbl_dir: Path,
    output_debug_dir: Path,
    num_samples: int = 8
) -> None:
    """Draws color-coded bounding boxes and labels on random images for manual sanity inspection."""
    output_debug_dir.mkdir(parents=True, exist_ok=True)
    img_files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not img_files:
        return

    random.seed(42)
    sample_pool = random.sample(img_files, min(len(img_files), num_samples * 2))
    saved = 0

    for img_p in sample_pool:
        if saved >= num_samples:
            break
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        try:
            im = Image.open(img_p).convert("RGB")
            draw = ImageDraw.Draw(im)
            w, h = im.size

            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]

            if not lines:
                # Background negative preview
                draw.text((15, 15), "[BACKGROUND NEGATIVE - NO SYMPTOMS]", fill=(16, 185, 129))
            else:
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy = float(parts[1]) * w, float(parts[2]) * h
                        bw, bh = float(parts[3]) * w, float(parts[4]) * h
                        x1 = int(cx - bw / 2.0)
                        y1 = int(cy - bh / 2.0)
                        x2 = int(cx + bw / 2.0)
                        y2 = int(cy + bh / 2.0)

                        col = CLASS_COLORS.get(cid, (255, 255, 255))
                        lbl_text = f"{cid}: {TARGET_CLASSES.get(cid, 'unknown')} ({bw/w*100:.1f}%)"
                        draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
                        draw.rectangle([x1, max(0, y1 - 20), x1 + len(lbl_text) * 8, y1], fill=col)
                        draw.text((x1 + 4, max(0, y1 - 18)), lbl_text, fill=(255, 255, 255))

            out_sample_p = output_debug_dir / f"sample_{saved+1}_{img_p.stem}.jpg"
            im.save(out_sample_p, format="JPEG", quality=90)
            saved += 1
        except Exception as e:
            print(f"[!] Could not render debug sample for {img_p.name}: {e}")

    print(f"[✓] Saved {saved} visual verification samples to: {output_debug_dir}")


def verify_dataset(
    dataset_dir: Path,
    allow_no_negatives: bool = False,
    allow_empty_classes: bool = False
) -> bool:
    print("=" * 75)
    print("  UrbanGreen PhytoCare CoreTech: Automated Dataset Verification Gate")
    print(f"  Target Dataset : {dataset_dir}")
    print("=" * 75)

    tr_img = dataset_dir / "images" / "train"
    tr_lbl = dataset_dir / "labels" / "train"
    val_img = dataset_dir / "images" / "val"
    val_lbl = dataset_dir / "labels" / "val"

    if not tr_img.exists() or not val_img.exists():
        print(f"[FATAL] Missing standard images/train or images/val directories in {dataset_dir}")
        return False

    # 1. Analyze train and val
    tr_imgs, tr_negs, tr_counts, tr_areas, tr_prefixes = analyze_split(tr_img, tr_lbl)
    val_imgs, val_negs, val_counts, val_areas, val_prefixes = analyze_split(val_img, val_lbl)

    print(f"\n[1] IMAGE & NEGATIVE INVENTORY:")
    print(f"    Train: {tr_imgs:5d} images total | {tr_negs:4d} background negatives (empty labels)")
    print(f"    Val  : {val_imgs:5d} images total | {val_negs:4d} background negatives (empty labels)")

    # 2. Leakage Audit
    print("\n[2] ANTI-LEAKAGE AUDIT (Roboflow Pre-augmented Twins):")
    overlap = tr_prefixes.intersection(val_prefixes)
    if overlap:
        print(f"    [FAIL] LEAKAGE DETECTED! {len(overlap)} base image stems appear in BOTH train and val:")
        for stem in list(overlap)[:5]:
            print(f"      - {stem}")
        print("    Failing verification gate to protect benchmark integrity.")
        return False
    else:
        print("    [PASS] Zero twin leakage detected between train and val splits.")

    # 3. Class Distribution & Box Size Audit (Edge Case #3)
    print("\n[3] CLASS CONTRACT & BOUNDING-BOX SIZE AUDIT:")
    print("-" * 75)
    print(f"{'Class ID':<9} {'Class Name':<24} {'Train Bboxes':<14} {'Val Bboxes':<12} {'Mean Area %'}")
    print("-" * 75)

    has_empty_class = False
    for cid, cname in TARGET_CLASSES.items():
        tr_c = tr_counts[cid]
        val_c = val_counts[cid]
        areas = tr_areas[cid] + val_areas[cid]
        mean_area = np.mean(areas) if areas else 0.0

        if tr_c == 0 or val_c == 0:
            has_empty_class = True

        note = ""
        if mean_area > 35.0:
            note = "[Whole-leaf sized]"
        elif mean_area > 0.0:
            note = "[Lesion/spot sized]"

        print(f" {cid:<8} {cname:<24} {tr_c:<14} {val_c:<12} {mean_area:5.1f}%  {note}")
    print("-" * 75)

    # 4. Generate Debug Visual Samples
    debug_dir = dataset_dir.parent.parent / "debug_samples"
    generate_debug_samples(tr_img, tr_lbl, debug_dir, num_samples=8)

    # 5. Gate Pass/Fail Decision
    gate_ok = True
    print("\n[4] PRE-FLIGHT GATE VERIFICATION SUMMARY:")
    if tr_negs == 0 and not allow_no_negatives:
        print("    [!] WARNING/FAIL: 0 background negative images found in train set.")
        print("        Foliar models without background negatives will hallucinate symptoms on pots/tables.")
        print("        Drop photos in ml_training/symptom_detector/raw_negatives/ or pass --allow-no-negatives.")
        gate_ok = False
    else:
        print("    [✓] Background negative images requirement satisfied.")

    if has_empty_class and not allow_empty_classes:
        print("    [!] WARNING/FAIL: One or more symptom classes have 0 bounding boxes.")
        gate_ok = False
    else:
        print("    [✓] All 6 symptom classes have annotated representation.")

    if gate_ok:
        print("\n[✓] VERIFICATION GATE PASSED! Dataset is fully verified and ready for YOLO11 training.\n")
    else:
        print("\n[!] VERIFICATION GATE FAILED. Please resolve issues above before launching training.\n")

    return gate_ok


def main():
    parser = argparse.ArgumentParser(description="UrbanGreen CoreTech: Dataset Verification Pre-Flight Gate")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="ml_training/symptom_detector/datasets/symptom_dataset",
        help="Path to prepared YOLO dataset"
    )
    parser.add_argument("--allow-no-negatives", action="store_true", help="Bypass check for background negatives")
    parser.add_argument("--allow-empty-classes", action="store_true", help="Bypass check for zero-box classes")
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir).resolve()
    success = verify_dataset(
        ds_dir,
        allow_no_negatives=args.allow_no_negatives,
        allow_empty_classes=args.allow_empty_classes
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
