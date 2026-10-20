"""
PhytoCare Data Equalization and Pathology Harvester Engine
Balances the 88% Aloe Vera skew and enriches starved minority classes:
  - Cacti (Dactylopius opuntiae scale & healthy pads)
  - Money Plant (Bacterial wilt, Manganese toxicity, healthy)
  - Snake Plant (Anthracnose, Withering, healthy)
  - Spider Plant (Fungal leaf spot, Tip necrosis, healthy)
"""

import os
import sys
import time
import json
import random
import shutil
import urllib.request
import urllib.parse
from pathlib import Path
from PIL import Image, ImageEnhance

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
LOG_FILE = EXPERIMENT_ROOT / "reports" / "data_balance.log"

def log_both(msg: str):
    print(msg, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
            f.write(msg + "\n")
    except Exception:
        pass

# Target bounds
ALOE_MAX_TRAIN = 1000
MINORITY_TARGET_TRAIN = 500

# iNaturalist Taxon Queries for Pathologies & Host Species
INAT_PATHOLOGY_QUERIES = {
    "Cactus_Dactylopius_Opuntia": {
        "taxon_id": 335785, # Dactylopius opuntiae
        "query": None
    },
    "Aloe_Rust": {
        "taxon_id": 1124873, # Uromyces aloes
        "query": None
    },
    "Money_Plant_Healthy": {
        "taxon_id": 82069, # Epipremnum aureum
        "query": None
    },
    "Snake_Plant_Healthy": {
        "taxon_id": 1109015, # Dracaena trifasciata (Sansevieria)
        "query": None
    },
    "Spider_Plant_Healthy": {
        "taxon_id": 160419, # Chlorophytum comosum
        "query": None
    },
    "Cactus_Healthy": {
        "taxon_id": 53330, # Opuntia
        "query": None
    }
}

HEADERS = {"User-Agent": "PhytoCare-Research-Harvester/1.0 (academic FYP)"}

def harvest_inaturalist(taxon_id: int, count: int = 150) -> list:
    """Fetches high-resolution research-grade photos from iNaturalist."""
    images = []
    page = 1
    fetched = 0
    while fetched < count and page <= 5:
        url = f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&photos=true&quality_grade=research&per_page=30&page={page}"
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode())
                obs_list = data.get("results", [])
                if not obs_list:
                    break
                for obs in obs_list:
                    for photo in obs.get("photos", []):
                        raw_url = photo.get("url", "")
                        if raw_url:
                            # Upgrade square.jpg to large.jpg (1024px)
                            large_url = raw_url.replace("square.jpg", "large.jpg")
                            images.append(large_url)
                            fetched += 1
                            if fetched >= count:
                                break
                    if fetched >= count:
                        break
        except Exception as e:
            time.sleep(0.5)
        page += 1
        time.sleep(0.3)
    return images

def prune_aloe_classes():
    """Downsamples excessive Aloe Vera images in train directory to 1,000 distinct images."""
    log_both("=" * 70)
    log_both("STEP 1: PRUNING & EQUALIZING ALOE VERA REDUNDANCY")
    log_both("=" * 70)

    for cls_dir in TRAIN_DIR.glob("Aloe_*"):
        files = list(cls_dir.glob("*.*"))
        total = len(files)
        if total > ALOE_MAX_TRAIN:
            # Deterministic shuffle to keep high variance
            random.seed(42)
            random.shuffle(files)
            to_remove = files[ALOE_MAX_TRAIN:]
            for f in to_remove:
                try:
                    f.unlink()
                except Exception:
                    pass
            log_both(f"  • {cls_dir.name}: Pruned {len(to_remove)} redundant images ({total} -> {ALOE_MAX_TRAIN})")
        else:
            log_both(f"  • {cls_dir.name}: Kept existing {total} images (under cap)")

def augment_and_enrich_minorities():
    """Enriches starved minority classes via targeted online harvesting and lesion-preserving photometric transformations."""
    log_both("=" * 70)
    log_both("STEP 2: HARVESTING & BALANCING STARVED MINORITY PATHOLOGY CLASSES")
    log_both("=" * 70)

    minority_classes = [d for d in TRAIN_DIR.iterdir() if d.is_dir() and not d.name.startswith("Aloe_")]

    for cls_dir in minority_classes:
        cls_name = cls_dir.name
        existing_files = list(cls_dir.glob("*.*"))
        count = len(existing_files)
        log_both(f"\n[Processing {cls_name}] Current count: {count} images (Target: {MINORITY_TARGET_TRAIN})")

        # 1. Harvest from iNaturalist if taxon mapping exists
        if cls_name in INAT_PATHOLOGY_QUERIES:
            t_id = INAT_PATHOLOGY_QUERIES[cls_name]["taxon_id"]
            log_both(f"  --> Fetching verified research-grade photos for taxon {t_id}...")
            urls = harvest_inaturalist(t_id, count=150)
            added_harvest = 0
            for i, u in enumerate(urls):
                try:
                    req = urllib.request.Request(u, headers=HEADERS)
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        content = resp.read()
                    out_path = cls_dir / f"inat_harvest_{int(time.time())}_{i}.jpg"
                    with open(out_path, "wb") as f_out:
                        f_out.write(content)
                    added_harvest += 1
                except Exception:
                    pass
            log_both(f"  --> Successfully saved {added_harvest} research-grade photos from iNaturalist.")

        # Re-check count
        current_files = list(cls_dir.glob("*.*"))
        deficit = MINORITY_TARGET_TRAIN - len(current_files)

        # 2. Lesion-Preserving Photometric Augmentations to fill deficit
        if deficit > 0:
            log_both(f"  --> Generating {deficit} lesion-preserving synthetic augmentations...")
            seed_images = [f for f in current_files if f.stat().st_size > 1000]
            if not seed_images:
                continue

            created = 0
            while created < deficit:
                src_path = random.choice(seed_images)
                try:
                    with Image.open(src_path) as img:
                        img = img.convert("RGB")
                        w, h = img.size

                        # Random Crop (0.85 - 1.0)
                        crop_scale = random.uniform(0.85, 1.0)
                        cw, ch = int(w * crop_scale), int(h * crop_scale)
                        left = random.randint(0, w - cw)
                        top = random.randint(0, h - ch)
                        img = img.crop((left, top, left + cw, top + ch))

                        # Dual Flips
                        if random.random() > 0.5:
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        if random.random() > 0.7:
                            img = img.transpose(Image.FLIP_TOP_BOTTOM)

                        # Subtle Rotation (-15 to 15 deg)
                        rot = random.uniform(-15, 15)
                        img = img.rotate(rot, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

                        # Color Jitter (Diagnostic preservation: slight contrast/brightness)
                        enh_b = ImageEnhance.Brightness(img)
                        img = enh_b.enhance(random.uniform(0.88, 1.12))
                        enh_c = ImageEnhance.Contrast(img)
                        img = enh_c.enhance(random.uniform(0.88, 1.15))

                        out_aug_path = cls_dir / f"aug_lesion_{created}_{int(time.time())}.jpg"
                        img.save(out_aug_path, quality=92)
                        created += 1
                except Exception as e:
                    pass

            log_both(f"  --> Staged {created} augmentations. Final class count: {len(list(cls_dir.glob('*.*')))}")

def summarize_final_distribution():
    log_both("=" * 70)
    log_both("EQUALIZED DATASET SUMMARY MATRIX (TRAIN SET)")
    log_both("=" * 70)
    total_imgs = 0
    for cls_dir in sorted(TRAIN_DIR.iterdir()):
        if cls_dir.is_dir():
            c = len(list(cls_dir.glob("*.*")))
            total_imgs += c
            log_both(f"  • {cls_dir.name:<35} {c:>5} images")
    log_both("-" * 70)
    log_both(f"  TOTAL TRAINING CORPUS: {total_imgs} balanced specimens across 16 classes.")
    log_both("=" * 70)

def main():
    log_both("Starting PhytoCare Dataset Equalization & Harvesting Pipeline...")
    prune_aloe_classes()
    augment_and_enrich_minorities()
    summarize_final_distribution()

if __name__ == "__main__":
    main()

