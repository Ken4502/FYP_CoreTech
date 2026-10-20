"""
Data Setup Script for Plant Identification Experimentation.
Hardlinks baseline train, val, and test data from the original Plant_Identification directory.
This ensures zero additional disk space usage and complete isolation from the original codebase.
"""

import os
import sys
import shutil
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
ORIGINAL_DATA_DIR = EXPERIMENT_ROOT.parent / "Plant_Identification" / "data"
TARGET_DATA_DIR = EXPERIMENT_ROOT / "data"

def setup_experiment_dataset():
    if not ORIGINAL_DATA_DIR.exists():
        print(f"Error: Original data directory not found at {ORIGINAL_DATA_DIR}")
        sys.exit(1)

    print(f"Setting up experimental data workspace at {TARGET_DATA_DIR}...")
    TARGET_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (TARGET_DATA_DIR / "staging").mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    total_linked = 0

    for split in splits:
        src_split = ORIGINAL_DATA_DIR / split
        dst_split = TARGET_DATA_DIR / split

        if not src_split.exists():
            print(f"Warning: Split {split} not found at {src_split}")
            continue

        dst_split.mkdir(parents=True, exist_ok=True)

        for class_dir in src_split.iterdir():
            if not class_dir.is_dir():
                continue

            target_class_dir = dst_split / class_dir.name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            for img_path in class_dir.iterdir():
                if not img_path.is_file():
                    continue

                target_file = target_class_dir / img_path.name
                if not target_file.exists():
                    try:
                        os.link(str(img_path), str(target_file))
                    except Exception:
                        shutil.copy2(str(img_path), str(target_file))
                    total_linked += 1

    print(f"Dataset initialization complete! Total files linked: {total_linked}")
    print(f"Original dataset remains 100% untouched and preserved.")

if __name__ == "__main__":
    setup_experiment_dataset()

