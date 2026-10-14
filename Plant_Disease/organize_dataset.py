"""
Dataset Organization & Verification Script for Plant Disease Identification.
Processes indoor dataset into Train, Val, and Test splits.
Validates image files, utilizes instant hardlinks where possible, and exports class_indices.json.
"""

import os
import shutil
import json
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def is_valid_image_file(fpath: Path) -> bool:
    """Quick validation: check non-zero file size and recognized image extension."""
    if fpath.suffix.lower() not in VALID_EXTENSIONS:
        return False
    try:
        return fpath.stat().st_size > 100
    except Exception:
        return False

def link_or_copy(src, dst):
    """Attempt hardlink first (instant, 0 disk space), fallback to copy."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def organize_dataset(
    source_dir="../Datasets/indoor",
    target_dir="data",
    artifacts_dir="artifacts"
):
    source_path = Path(source_dir).resolve()
    target_path = Path(target_dir).resolve()
    artifacts_path = Path(artifacts_dir).resolve()

    artifacts_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset directory not found at: {source_path}")

    # Dataset splits mapping: source split -> target split
    split_mapping = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }

    train_source = source_path / "train"
    if not train_source.exists():
        raise FileNotFoundError(f"Train directory not found in {source_path}")

    class_folders = sorted([f for f in os.listdir(train_source) if (train_source / f).is_dir()])
    print(f"🌿 Found {len(class_folders)} plant disease classes in {source_path}:")
    for idx, c in enumerate(class_folders, 1):
        print(f"   {idx:>2}. {c}")

    class_to_idx = {name: idx for idx, name in enumerate(class_folders)}
    idx_to_class = {idx: name for idx, name in enumerate(class_folders)}

    class_indices_file = artifacts_path / "class_indices.json"
    with open(class_indices_file, "w", encoding="utf-8") as f:
        json.dump({
            "classes": class_folders,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class
        }, f, indent=2)
    print(f"\nSaved class index mapping to {class_indices_file}")

    stats = {"train": 0, "val": 0, "test": 0}
    class_stats = {c: {"train": 0, "val": 0, "test": 0} for c in class_folders}

    for src_split, tgt_split in split_mapping.items():
        src_split_dir = source_path / src_split
        if not src_split_dir.exists():
            print(f"[Warning] Split {src_split} not found in {source_path}. Skipping.")
            continue

        print(f"\nLinking and organizing '{src_split}' -> '{tgt_split}'...")
        for class_name in tqdm(class_folders, desc=f"Processing {tgt_split}"):
            class_src_dir = src_split_dir / class_name
            class_tgt_dir = target_path / tgt_split / class_name

            if not class_src_dir.exists():
                continue

            for fname in os.listdir(class_src_dir):
                fpath = class_src_dir / fname
                if not fpath.is_file():
                    continue

                if is_valid_image_file(fpath):
                    dst_file = class_tgt_dir / fname
                    link_or_copy(fpath, dst_file)
                    stats[tgt_split] += 1
                    class_stats[class_name][tgt_split] += 1

    print("\n" + "=" * 60)
    print("🌿 Dataset Organization Summary:")
    print(f"  Train images:      {stats['train']}")
    print(f"  Validation images: {stats['val']}")
    print(f"  Test images:       {stats['test']}")
    print(f"  Total valid:       {sum(stats.values())}")
    print("=" * 60)

    # Save detailed distribution
    with open(artifacts_path / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": sum(stats.values()),
            "splits": stats,
            "per_class": class_stats
        }, f, indent=2)

    return stats

if __name__ == "__main__":
    organize_dataset()

