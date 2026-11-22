"""
UrbanGreen CoreTech: Custom YOLO11 Symptom Dataset Preparation Pipeline.
Dynamic Remapper, Anti-Leakage Splitter, Class Balancer & Background Negative Integrator.
Supports both local Roboflow/YOLO exports and automated Hugging Face PlantDoc download.

Usage:
  python prepare_dataset.py [--plantdoc_dir "path/to/PlantDoc"] [--pest_dir "path/to/Pest"]
"""
import os
import sys
import shutil
import random
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import yaml

TARGET_CLASSES = {
    0: "yellowing_chlorosis",
    1: "crispy_brown_edges",
    2: "dark_spots_necrosis",
    3: "white_powdery_residue",
    4: "chewed_holes_stippling",
    5: "healthy_leaf",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def map_class_name(name: str) -> Optional[int]:
    """
    Dynamic Name-Based Remapping into 6 target classes:
      *yellow*, *chlorosis*, *mosaic*, *virus* -> 0 (yellowing_chlorosis)
      *scorch*, *burn*, *dry*, *crispy*, *brown*, *early_blight* -> 1 (crispy_brown_edges)
      *spot*, *blight*, *scab*, *rust*, *necrosis*, *rot*, *anthracnose* -> 2 (dark_spots_necrosis)
      *mildew*, *mold*, *powdery*, *white* -> 3 (white_powdery_residue)
      *pest*, *insect*, *chew*, *hole*, *miner*, *damage*, *bite*, *mite*, *stippl* -> 4 (chewed_holes_stippling)
      *healthy* / uninfected reference leaf -> 5 (healthy_leaf)
    """
    n = name.lower().replace("-", " ").replace("_", " ")

    if "healthy" in n:
        return 5
    # Prioritize chewing/pest/mite indicators before generic 'spot' (e.g. 'two spotted spider mite')
    if any(k in n for k in ["pest", "insect", "chew", "hole", "miner", "damage", "bite", "mite", "stippl"]):
        return 4
    if any(k in n for k in ["yellow", "chlorosis", "mosaic", "virus"]):
        return 0
    # Match early blight or scorched/dry/brown before generic blight/spot
    if any(k in n for k in ["early blight", "scorch", "burn", "dry", "crispy", "brown"]):
        return 1
    if any(k in n for k in ["spot", "blight", "scab", "rust", "necrosis", "rot", "anthracnose"]):
        return 2
    if any(k in n for k in ["mildew", "mold", "powdery", "white"]):
        return 3
    if "leaf" in n:
        return 5
    return None


def get_base_prefix(filename: str) -> str:
    """Extracts base filename prefix before .rf. to prevent Roboflow twin leakage."""
    stem = Path(filename).stem
    if ".rf." in stem:
        return stem.split(".rf.")[0]
    match = re.split(r"[\._]rf[\._]", stem, maxsplit=1)
    return match[0]


def parse_source_yaml(dataset_dir: Path) -> Dict[int, Optional[int]]:
    """Loads source data.yaml and constructs mapping from source_id -> target_id (0-5)."""
    yaml_candidates = list(dataset_dir.glob("*.yaml")) + list(dataset_dir.glob("*.yml"))
    if not yaml_candidates:
        print(f"[!] Warning: No data.yaml found in {dataset_dir}. Falling back to identity or standard index.")
        return {}

    yaml_path = yaml_candidates[0]
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    mapping = {}
    if isinstance(names, list):
        for idx, name in enumerate(names):
            tgt = map_class_name(str(name))
            mapping[idx] = tgt
            status = f"-> {TARGET_CLASSES[tgt]} ({tgt})" if tgt is not None else "-> [DISCARDED/UNMAPPED]"
            print(f"   [{idx}] '{name}' {status}")
    elif isinstance(names, dict):
        for idx_str, name in names.items():
            idx = int(idx_str)
            tgt = map_class_name(str(name))
            mapping[idx] = tgt
            status = f"-> {TARGET_CLASSES[tgt]} ({tgt})" if tgt is not None else "-> [DISCARDED/UNMAPPED]"
            print(f"   [{idx}] '{name}' {status}")
    return mapping


def find_split_dirs(dataset_dir: Path) -> Dict[str, Tuple[Optional[Path], Optional[Path]]]:
    """Finds train and val/valid image and label folders in source dataset."""
    splits = {}
    for s_name in ["train", "valid", "val"]:
        key = "val" if s_name in ("valid", "val") else "train"
        if key in splits and splits[key][0] is not None:
            continue

        img_dir1 = dataset_dir / s_name / "images"
        lbl_dir1 = dataset_dir / s_name / "labels"
        if img_dir1.exists():
            splits[key] = (img_dir1, lbl_dir1)
            continue

        img_dir2 = dataset_dir / "images" / s_name
        lbl_dir2 = dataset_dir / "labels" / s_name
        if img_dir2.exists():
            splits[key] = (img_dir2, lbl_dir2)
            continue

        img_dir3 = dataset_dir / s_name
        if img_dir3.exists() and any(p.suffix.lower() in IMG_EXTS for p in img_dir3.iterdir() if p.is_file()):
            lbl_dir3 = dataset_dir / (s_name + "_labels")
            if not lbl_dir3.exists():
                lbl_dir3 = dataset_dir / "labels" / s_name
            splits[key] = (img_dir3, lbl_dir3 if lbl_dir3.exists() else None)
            continue

    return splits


def process_dataset(
    source_dir: Path,
    out_img_train: Path,
    out_lbl_train: Path,
    out_img_val: Path,
    out_lbl_val: Path,
    prefix_tag: str,
) -> None:
    """Remaps and copies a source dataset into destination train/val splits."""
    if not source_dir or not source_dir.exists():
        return

    print(f"\n[*] Processing source dataset: {source_dir.name} ({source_dir})")
    cls_map = parse_source_yaml(source_dir)
    splits = find_split_dirs(source_dir)

    has_train = "train" in splits and splits["train"][0] is not None
    has_val = "val" in splits and splits["val"][0] is not None

    if has_train and has_val:
        print("   Found existing train/ and valid/ splits. Preserving author split boundaries.")
        _copy_and_remap(splits["train"][0], splits["train"][1], out_img_train, out_lbl_train, cls_map, f"{prefix_tag}_tr")
        _copy_and_remap(splits["val"][0], splits["val"][1], out_img_val, out_lbl_val, cls_map, f"{prefix_tag}_val")
    else:
        print("   No valid/ split found. Grouping by base filename to prevent Roboflow twin leakage (85/15)...")
        search_root = splits.get("train", (source_dir, None))[0] or source_dir
        lbl_root = splits.get("train", (None, None))[1]

        all_imgs = [p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        groups = defaultdict(list)
        for img_p in all_imgs:
            base_pfx = get_base_prefix(img_p.name)
            groups[base_pfx].append(img_p)

        group_keys = list(groups.keys())
        random.seed(42)
        random.shuffle(group_keys)
        n_train = int(len(group_keys) * 0.85)
        train_keys = set(group_keys[:n_train])
        val_keys = set(group_keys[n_train:])

        for k in train_keys:
            for img_p in groups[k]:
                _remap_single_file(img_p, lbl_root, out_img_train, out_lbl_train, cls_map, f"{prefix_tag}_tr")
        for k in val_keys:
            for img_p in groups[k]:
                _remap_single_file(img_p, lbl_root, out_img_val, out_lbl_val, cls_map, f"{prefix_tag}_val")


def _find_matching_label(img_p: Path, explicit_lbl_dir: Optional[Path]) -> Optional[Path]:
    """Finds matching .txt label for an image."""
    candidates = []
    if explicit_lbl_dir and explicit_lbl_dir.exists():
        candidates.append(explicit_lbl_dir / f"{img_p.stem}.txt")
    candidates.append(img_p.with_suffix(".txt"))
    parts = list(img_p.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        lbl_parts = parts[:idx] + ("labels",) + parts[idx+1:]
        candidates.append(Path(*lbl_parts).with_suffix(".txt"))

    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _remap_single_file(
    img_p: Path,
    lbl_dir: Optional[Path],
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    cls_map: Dict[int, Optional[int]],
    prefix: str,
) -> None:
    dst_img_name = f"{prefix}_{img_p.name}"
    dst_img_path = dst_img_dir / dst_img_name
    dst_lbl_path = dst_lbl_dir / f"{Path(dst_img_name).stem}.txt"

    lbl_p = _find_matching_label(img_p, lbl_dir)
    remapped_lines = []

    if lbl_p and lbl_p.exists():
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    src_cls = int(parts[0])
                    if cls_map:
                        tgt_cls = cls_map.get(src_cls)
                    else:
                        tgt_cls = src_cls if src_cls in TARGET_CLASSES else None

                    if tgt_cls is not None:
                        remapped_lines.append(f"{tgt_cls} " + " ".join(parts[1:]))
                except ValueError:
                    continue

    shutil.copy2(img_p, dst_img_path)
    with open(dst_lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(remapped_lines) + ("\n" if remapped_lines else ""))


def _copy_and_remap(
    src_img_dir: Path,
    src_lbl_dir: Optional[Path],
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    cls_map: Dict[int, Optional[int]],
    prefix: str,
) -> None:
    img_files = [p for p in src_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    for img_p in img_files:
        _remap_single_file(img_p, src_lbl_dir, dst_img_dir, dst_lbl_dir, cls_map, prefix)


def auto_download_plantdoc(
    out_img_train: Path,
    out_lbl_train: Path,
    out_img_val: Path,
    out_lbl_val: Path,
    stage_dir: Path
) -> None:
    """Automatically downloads agyaatcoder/PlantDoc parquet shards and converts to YOLO11."""
    import io
    import numpy as np
    import polars as pl
    from PIL import Image
    from huggingface_hub import hf_hub_download

    print("\n[*] Automatically fetching PlantDoc from Hugging Face (agyaatcoder/PlantDoc)...")
    stage_dir.mkdir(parents=True, exist_ok=True)
    shards = [
        ("train", "data/train-00000-of-00002.parquet"),
        ("train", "data/train-00001-of-00002.parquet"),
        ("val", "data/test-00000-of-00001.parquet"),
    ]

    total_converted = 0
    for split, sh in shards:
        dst_img_dir = out_img_train if split == "train" else out_img_val
        dst_lbl_dir = out_lbl_train if split == "train" else out_lbl_val

        print(f"   Downloading shard {sh} ({split})...")
        pq_path = hf_hub_download("agyaatcoder/PlantDoc", sh, repo_type="dataset", local_dir=str(stage_dir))
        df = pl.read_parquet(pq_path)

        for row in df.iter_rows(named=True):
            try:
                raw_bytes = row["image"]["bytes"]
                rgb = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                w, h = rgb.size
                if min(w, h) < 64:
                    continue

                objs = row.get("objects") or {}
                bboxes = objs.get("bbox") or []
                cats = objs.get("category") or []

                lines = []
                for bb, cat in zip(bboxes, cats):
                    tgt_cls = map_class_name(str(cat))
                    if tgt_cls is None:
                        continue
                    # COCO [x, y, bw_px, bh_px] format
                    x, y, bw_px, bh_px = [float(v) for v in bb]
                    if bw_px < 2 or bh_px < 2:
                        continue
                    cx = np.clip((x + bw_px / 2.0) / w, 0.0, 1.0)
                    cy = np.clip((y + bh_px / 2.0) / h, 0.0, 1.0)
                    bw = np.clip(bw_px / w, 0.005, 1.0)
                    bh = np.clip(bh_px / h, 0.005, 1.0)
                    lines.append(f"{tgt_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                out_stem = f"pdoc_{split}_{total_converted:05d}"
                img_path = dst_img_dir / f"{out_stem}.jpg"
                lbl_path = dst_lbl_dir / f"{out_stem}.txt"

                rgb.save(img_path, "JPEG", quality=92)
                lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                total_converted += 1
            except Exception:
                continue

    print(f"[✓] Successfully converted {total_converted} PlantDoc images into YOLO format.")


def populate_default_negatives_if_empty(negatives_dir: Path) -> None:
    """Populates raw_negatives with indoor rooms and healthy plant photos from local CoreTech datasets if empty."""
    negatives_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in negatives_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if existing:
        return

    script_dir = Path(__file__).resolve().parent
    coretech_root = script_dir.parent.parent

    copied = 0
    # 1. Non-plant indoor rooms (tables, floors, pots)
    mit_dir = coretech_root / "CoreTech_v2" / "data" / "non_plant_hard_neg" / "mit_indoor"
    if mit_dir.exists():
        mit_imgs = list(mit_dir.glob("*.jpg"))
        random.seed(42)
        random.shuffle(mit_imgs)
        for p in mit_imgs[:60]:
            shutil.copy2(p, negatives_dir / f"indoor_room_{p.name}")
            copied += 1

    # 2. Healthy indoor house plants
    indoor_train = coretech_root / "Datasets" / "indoor" / "train"
    if indoor_train.exists():
        for healthy_folder in indoor_train.glob("*Healthy"):
            if healthy_folder.is_dir():
                h_imgs = [p for p in healthy_folder.iterdir() if p.suffix.lower() in IMG_EXTS]
                random.seed(42)
                random.shuffle(h_imgs)
                for p in h_imgs[:15]:
                    shutil.copy2(p, negatives_dir / f"healthy_{healthy_folder.name}_{p.name}")
                    copied += 1

    if copied > 0:
        print(f"[✓] Auto-seeded {copied} indoor non-plant and healthy plant photos into: {negatives_dir}")


def handle_negatives(negatives_dir: Path, out_img_train: Path, out_lbl_train: Path, out_img_val: Path, out_lbl_val: Path) -> int:
    """Discovers background negative photos, auto-creates README if missing, and splits 85/15 by base prefix to avoid twin leakage."""
    populate_default_negatives_if_empty(negatives_dir)

    neg_imgs = [p for p in negatives_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not neg_imgs:
        print(f"[!] No background negative photos found in {negatives_dir}.")
        return 0

    # Group by base prefix to prevent Roboflow twin leakage
    groups = defaultdict(list)
    for p in neg_imgs:
        pfx = get_base_prefix(p.name)
        groups[pfx].append(p)

    group_keys = list(groups.keys())
    random.seed(42)
    random.shuffle(group_keys)
    n_tr = int(len(group_keys) * 0.85)
    tr_keys = set(group_keys[:n_tr])
    val_keys = set(group_keys[n_tr:])

    tr_imgs = [img for k in tr_keys for img in groups[k]]
    val_imgs = [img for k in val_keys for img in groups[k]]

    for p in tr_imgs:
        dst_img = out_img_train / f"neg_{p.name}"
        dst_lbl = out_lbl_train / f"{dst_img.stem}.txt"
        shutil.copy2(p, dst_img)
        dst_lbl.write_text("", encoding="utf-8")

    for p in val_imgs:
        dst_img = out_img_val / f"neg_{p.name}"
        dst_lbl = out_lbl_val / f"{dst_img.stem}.txt"
        shutil.copy2(p, dst_img)
        dst_lbl.write_text("", encoding="utf-8")

    print(f"[✓] Added {len(neg_imgs)} background negative photos ({len(tr_imgs)} train / {len(val_imgs)} val, grouped across {len(group_keys)} unique prefixes).")
    return len(neg_imgs)


def count_boxes(lbl_dir: Path) -> Tuple[Dict[int, int], Dict[Path, List[int]]]:
    """Counts boxes per class and returns mapping from file -> list of class IDs."""
    counts = {i: 0 for i in range(len(TARGET_CLASSES))}
    file_classes = defaultdict(list)
    for lbl_p in lbl_dir.glob("*.txt"):
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                    if cid in counts:
                        counts[cid] += 1
                        file_classes[lbl_p].append(cid)
                except ValueError:
                    continue
    return counts, file_classes


def balance_training_set(
    img_dir: Path,
    lbl_dir: Path,
    max_class_ratio: float = 3.0,
) -> None:
    """
    Automated Class Balancing on Train Split Only:
      - Oversample rare symptom classes (< 300 boxes) up to 3x or 300 boxes.
      - Subsample pure dominant-only images if dominant class > max_class_ratio * median.
    """
    counts, file_classes = count_boxes(lbl_dir)
    print("\n" + "=" * 65)
    print("  TRAINING SET CLASS DISTRIBUTION (BEFORE BALANCING)")
    print("=" * 65)
    for cid, cname in TARGET_CLASSES.items():
        print(f"  Class {cid:2d} ({cname:24s}): {counts[cid]:6d} boxes")

    non_empty_counts = [cnt for cnt in counts.values() if cnt > 0]
    if not non_empty_counts:
        print("[!] Warning: No annotated bounding boxes found in training set.")
        return

    # 1. Rare Class Oversampling (Classes with < 300 boxes)
    for cid in range(len(TARGET_CLASSES)):
        cnt = counts[cid]
        if 0 < cnt < 300:
            target_need = 300 - cnt
            candidate_files = [f for f, cls_list in file_classes.items() if cid in cls_list]
            if not candidate_files:
                continue

            max_duplicates = min(3, max(1, target_need // max(1, len(candidate_files))))
            print(f"[*] Oversampling rare class {cid} ({TARGET_CLASSES[cid]}): current={cnt}, duplicating {len(candidate_files)} images x{max_duplicates}...")

            for dup_idx in range(1, max_duplicates + 1):
                for lbl_p in candidate_files:
                    stem = lbl_p.stem
                    matching_imgs = list(img_dir.glob(f"{stem}.*"))
                    if not matching_imgs:
                        continue
                    orig_img = matching_imgs[0]

                    new_stem = f"bal_c{cid}_d{dup_idx}_{stem}"
                    new_lbl_p = lbl_dir / f"{new_stem}.txt"
                    new_img_p = img_dir / f"{new_stem}{orig_img.suffix}"

                    shutil.copy2(lbl_p, new_lbl_p)
                    shutil.copy2(orig_img, new_img_p)

    # Re-count after oversampling
    counts, file_classes = count_boxes(lbl_dir)
    non_zero = sorted([cnt for cnt in counts.values() if cnt > 0])
    median_cnt = non_zero[len(non_zero) // 2] if non_zero else 1
    max_allowed = int(median_cnt * max_class_ratio)

    # 2. Dominant Class Subsampling (pure dominant-only images)
    for cid, cnt in counts.items():
        if cnt > max_allowed:
            excess = cnt - max_allowed
            print(f"[*] Dominant class {cid} ({TARGET_CLASSES[cid]}) has {cnt} boxes (> {max_allowed}). Subsampling pure images...")
            pure_files = [
                f for f, cls_list in file_classes.items()
                if len(cls_list) > 0 and all(c == cid for c in cls_list)
            ]
            random.seed(42)
            random.shuffle(pure_files)

            pruned_boxes = 0
            for lbl_p in pure_files:
                if pruned_boxes >= excess:
                    break
                box_count = len(file_classes[lbl_p])
                stem = lbl_p.stem
                matching_imgs = list(img_dir.glob(f"{stem}.*"))

                lbl_p.unlink(missing_ok=True)
                for img_p in matching_imgs:
                    img_p.unlink(missing_ok=True)

                pruned_boxes += box_count

    # Final count display
    final_counts, _ = count_boxes(lbl_dir)
    print("\n" + "=" * 65)
    print("  TRAINING SET CLASS DISTRIBUTION (AFTER BALANCING)")
    print("=" * 65)
    for cid, cname in TARGET_CLASSES.items():
        print(f"  Class {cid:2d} ({cname:24s}): {final_counts[cid]:6d} boxes (delta: {final_counts[cid] - counts.get(cid, 0):+d})")
    print("=" * 65)


def generate_data_yaml(output_dir: Path) -> Path:
    """Generates standard YOLO11 data.yaml for the 6-class symptom detector."""
    yaml_path = output_dir / "data.yaml"
    content = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in TARGET_CLASSES.items()}
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, sort_keys=False)
    print(f"\n[✓] Generated data.yaml config: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="UrbanGreen CoreTech: YOLO11 Symptom Dataset Preparation")
    parser.add_argument("--plantdoc_dir", type=str, default="", help="Path to PlantDoc YOLO dataset (optional)")
    parser.add_argument("--pest_dir", type=str, default="", help="Path to Pest / Leaf Damage dataset (optional)")
    parser.add_argument(
        "--negatives_dir",
        type=str,
        default="ml_training/symptom_detector/raw_negatives",
        help="Path to healthy indoor negative photos"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ml_training/symptom_detector/datasets/symptom_dataset",
        help="Target output dataset directory"
    )
    parser.add_argument("--max_class_ratio", type=float, default=3.0, help="Max ratio of dominant to median class")
    parser.add_argument("--auto_download", action="store_true", help="Auto-download PlantDoc from Hugging Face")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_img_tr = out_dir / "images" / "train"
    out_img_val = out_dir / "images" / "val"
    out_lbl_tr = out_dir / "labels" / "train"
    out_lbl_val = out_dir / "labels" / "val"

    for d in [out_img_tr, out_img_val, out_lbl_tr, out_lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  UrbanGreen PhytoCare CoreTech: Preparing Symptom Dataset")
    print(f"  Output Directory : {out_dir}")
    print(f"  Target Classes   : {len(TARGET_CLASSES)} {list(TARGET_CLASSES.values())}")
    print("=" * 70)

    # 1. Process PlantDoc dataset
    if args.plantdoc_dir and Path(args.plantdoc_dir).exists():
        process_dataset(
            Path(args.plantdoc_dir),
            out_img_tr, out_lbl_tr, out_img_val, out_lbl_val,
            prefix_tag="plantdoc"
        )
    else:
        # Auto-download from Hugging Face
        stage_dir = out_dir.parent / "_hf_plantdoc_stage"
        auto_download_plantdoc(out_img_tr, out_lbl_tr, out_img_val, out_lbl_val, stage_dir)

    # 2. Process Pest dataset if provided
    if args.pest_dir and Path(args.pest_dir).exists():
        process_dataset(
            Path(args.pest_dir),
            out_img_tr, out_lbl_tr, out_img_val, out_lbl_val,
            prefix_tag="pest"
        )

    # 3. Handle Background Negatives
    handle_negatives(
        Path(args.negatives_dir).resolve(),
        out_img_tr, out_lbl_tr, out_img_val, out_lbl_val
    )

    # 4. Automated Class Balancing (Train Split Only)
    balance_training_set(out_img_tr, out_lbl_tr, max_class_ratio=args.max_class_ratio)

    # 5. Write data.yaml
    generate_data_yaml(out_dir)
    print("\n[✓] Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
