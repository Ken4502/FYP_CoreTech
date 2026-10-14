"""
Dataset Organization & Verification Script for Plant Identification
Splits house_plant_species dataset into 80% Train, 10% Val, 10% Test (Stratified)
Validates image files and exports class_indices.json
"""

import os
import shutil
import random
import json
from pathlib import Path
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

def verify_image(image_path):
    """Verify that an image can be opened and converted to RGB."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        # verify() only checks header; test actual decode and conversion
        with Image.open(image_path) as img:
            _ = img.convert("RGB")
        return True
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
    source_dir="../Datasets/house_plant_species",
    target_dir="data",
    artifacts_dir="artifacts",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    random.seed(seed)
    source_path = Path(source_dir).resolve()
    target_path = Path(target_dir).resolve()
    artifacts_path = Path(artifacts_dir).resolve()

    artifacts_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset directory not found at: {source_path}")

    # Gather class folders
    class_folders = sorted([f for f in os.listdir(source_path) if (source_path / f).is_dir()])
    print(f"Found {len(class_folders)} plant species classes in {source_path}")

    class_to_idx = {name: idx for idx, name in enumerate(class_folders)}
    idx_to_class = {idx: name for idx, name in enumerate(class_folders)}

    with open(artifacts_path / "class_indices.json", "w", encoding="utf-8") as f:
        json.dump({
            "classes": class_folders,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class
        }, f, indent=2)
    print(f"Saved class index mapping to {artifacts_path / 'class_indices.json'}")

    splits = ["train", "val", "test"]
    stats = {split: 0 for split in splits}
    corrupt_files = []

    print("\nVerifying and splitting images into 80/10/10...")
    for class_name in tqdm(class_folders, desc="Processing Classes"):
        class_src_dir = source_path / class_name
        valid_files = []

        for fname in os.listdir(class_src_dir):
            fpath = class_src_dir / fname
            if not fpath.is_file():
                continue
            if verify_image(fpath):
                valid_files.append(fname)
            else:
                corrupt_files.append(str(fpath))

        random.shuffle(valid_files)
        total_valid = len(valid_files)

        n_train = int(total_valid * train_ratio)
        n_val = int(total_valid * val_ratio)
        # Ensure at least 1 for val and test if class has enough images
        if n_val == 0 and total_valid >= 5:
            n_val = 1
        n_test = total_valid - n_train - n_val

        train_files = valid_files[:n_train]
        val_files = valid_files[n_train:n_train + n_val]
        test_files = valid_files[n_train + n_val:]

        split_dict = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        for split_name, files in split_dict.items():
            split_class_dir = target_path / split_name / class_name
            for fname in files:
                src_file = class_src_dir / fname
                dst_file = split_class_dir / fname
                link_or_copy(src_file, dst_file)
                stats[split_name] += 1

    print("\nDataset Organization Summary:")
    print(f"  Train images:      {stats['train']} ({stats['train']/(sum(stats.values()) or 1)*100:.1f}%)")
    print(f"  Validation images: {stats['val']} ({stats['val']/(sum(stats.values()) or 1)*100:.1f}%)")
    print(f"  Test images:       {stats['test']} ({stats['test']/(sum(stats.values()) or 1)*100:.1f}%)")
    print(f"  Total valid:       {sum(stats.values())}")
    print(f"  Corrupt images:    {len(corrupt_files)}")

    if corrupt_files:
        with open(artifacts_path / "corrupt_files.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(corrupt_files))
        print(f"Logged {len(corrupt_files)} corrupt files to {artifacts_path / 'corrupt_files.txt'}")

    return stats

if __name__ == "__main__":
    organize_dataset()
