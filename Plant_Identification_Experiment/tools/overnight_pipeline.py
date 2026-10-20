"""
Automated Overnight Pipeline for FloraNet-100.
Executes the full end-to-end workflow:
  Phase 1: Deepen existing 15 weak classes from iNaturalist
  Phase 2: Widen dataset with 53 newly introduced houseplant species
  Phase 3: Merge staged images & rebalance 100-class dataset (Train/Val/Test)
  Phase 4: Train MobileNetV3 Large (ImageNet transfer learning) for 30 epochs on RTX 3060
  Phase 5: Full 100-class evaluation and benchmark report generation
"""

import os
import sys
import time
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CURRENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = CURRENT_DIR.parent
CORETECH_ROOT = EXPERIMENT_ROOT.parent

if str(CORETECH_ROOT) not in sys.path:
    sys.path.insert(0, str(CORETECH_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config

LOG_FILE = config.REPORTS_DIR / "overnight_pipeline.log"

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run_overnight_pipeline():
    sys.stdout = Logger(LOG_FILE)
    sys.stderr = sys.stdout

    overall_start = time.time()
    print("=" * 75)
    print("🌙 STARTING OVERNIGHT PIPELINE: FLORANET-100 BOTANICAL EXPANSION")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: 100 Botanical Species (47 Deepened + 53 Newly Introduced)")
    print(f"Hardware: NVIDIA GeForce RTX 3060 (12GB VRAM) | CUDA Active: {config.DEVICE.type == 'cuda'}")
    print(f"Logs: {LOG_FILE}")
    print("=" * 75)

    from tools.harvester import harvest_species, merge_staged_dataset, DEEPEN_15_CLASSES, NEW_53_CLASSES

    # Check if dataset is already assembled and balanced with 100 non-empty classes
    dataset_ready = (
        config.CLASS_INDICES_PATH.exists() and
        len([d for d in config.TRAIN_DIR.iterdir() if d.is_dir()]) == 100 and
        all(len(list(d.glob("*.*"))) > 0 for d in config.TRAIN_DIR.iterdir() if d.is_dir())
    )

    if dataset_ready:
        print("\n" + "=" * 70)
        print("🎉 100-CLASS DATASET ALREADY ASSEMBLED AND BALANCED!")
        print(f"Verified: 100 classes populated across Train, Val, and Test.")
        print("Skipping Phases 1, 2, and 3 directly to Phase 4 (Model Training on RTX 3060).")
        print("=" * 70)
    else:
        # -------------------------------------------------------------------------
        # PHASE 1: Deepening the 15 Weakest Original Classes
        # -------------------------------------------------------------------------
        print("\n" + "=" * 70)
        print(f"📥 [PHASE 1/5] DEEPENING {len(DEEPEN_15_CLASSES)} UNDER-REPRESENTED SPECIES")
        print("Target: 25 research-grade images per class with Method 1 foliage validation")
        print("=" * 70)
        p1_start = time.time()
        deepen_count = 0
        for i, cls in enumerate(DEEPEN_15_CLASSES, 1):
            print(f"\n({i}/{len(DEEPEN_15_CLASSES)}) Deepening '{cls}'...")
            deepen_count += harvest_species(cls, target_count=25)
        p1_time = (time.time() - p1_start) / 60
        print(f"\n✅ Phase 1 Complete in {p1_time:.1f} mins. Harvested {deepen_count} images.")

        # -------------------------------------------------------------------------
        # PHASE 2: Widening with 53 New Houseplant Species
        # -------------------------------------------------------------------------
        print("\n" + "=" * 70)
        print(f"🌱 [PHASE 2/5] WIDENING TAXONOMY WITH {len(NEW_53_CLASSES)} NEW SPECIES")
        print("Target: 35 research-grade images per class with Method 1 foliage validation")
        print("=" * 70)
        p2_start = time.time()
        widen_count = 0
        for i, cls in enumerate(NEW_53_CLASSES, 1):
            print(f"\n({i}/{len(NEW_53_CLASSES)}) Harvesting new species '{cls}'...")
            widen_count += harvest_species(cls, target_count=35)
        p2_time = (time.time() - p2_start) / 60
        print(f"\n✅ Phase 2 Complete in {p2_time:.1f} mins. Harvested {widen_count} images.")

        # -------------------------------------------------------------------------
        # PHASE 3: Dataset Rebalancing & 100-Class Index Generation
        # -------------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("🔄 [PHASE 3/5] MERGING & REBALANCING DATASET INTO 100 BOTANICAL CLASSES")
        print("Partitioning: Existing classes (Train 85%, Val 15%), New classes (Train 70%, Val 15%, Test 15%)")
        print("=" * 70)
        merge_staged_dataset()

    # -------------------------------------------------------------------------
    # PHASE 4: Model Training on RTX 3060
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("🔥 [PHASE 4/5] TRAINING FLORANET-100 (TRANSFER LEARNING ON RTX 3060)")
    print(f"Regimen: 30 Epochs, Cosine Annealing, RandAugment, Class-Weighted Loss")
    print("=" * 70)
    p4_start = time.time()
    from src.train import train_pipeline
    train_pipeline(epochs=config.TOTAL_EPOCHS, batch_size=config.BATCH_SIZE, lr=config.PEAK_LR)
    p4_time = (time.time() - p4_start) / 60
    print(f"\n✅ Phase 4 Complete in {p4_time:.1f} mins.")

    # -------------------------------------------------------------------------
    # PHASE 5: Comprehensive 100-Class Test Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 [PHASE 5/5] EVALUATING FLORANET-100 ON TEST BENCHMARK")
    print("=" * 70)
    from src.evaluate import evaluate_test_set
    evaluate_test_set()

    total_pipeline_hours = (time.time() - overall_start) / 3600
    print("\n" + "=" * 75)
    print(f"🏆 OVERNIGHT PIPELINE FULLY COMPLETED in {total_pipeline_hours:.2f} hours!")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Artifacts:")
    print(f"  • Best Checkpoint: {config.BEST_MODEL_PATH}")
    print(f"  • Classification Report: {config.CLASSIFICATION_REPORT_PATH}")
    print(f"  • Training Curves: {config.TRAINING_CURVES_PATH}")
    print("=" * 75)

if __name__ == "__main__":
    run_overnight_pipeline()

