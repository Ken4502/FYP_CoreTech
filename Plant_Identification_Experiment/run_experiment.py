"""
End-to-End Orchestrator for Enhanced Plant Identification Experimentation.
Runs data harvesting, dataset merging, transfer learning training, and comparative evaluation.
"""

import sys
import argparse
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="Plant Identification Experiment Runner")
    parser.add_argument("--harvest", action="store_true", help="Harvest verified images for weak classes via iNaturalist API")
    parser.add_argument("--species", type=str, default="weak", help="Target species ('weak', 'all', or name like 'Yucca')")
    parser.add_argument("--count", type=int, default=15, help="Number of images per class to harvest")
    parser.add_argument("--merge", action="store_true", help="Merge staged harvested images into experimental training data")
    parser.add_argument("--train", action="store_true", help="Run enhanced training pipeline (Transfer Learning + RandAugment)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model and compare against baseline metrics")
    parser.add_argument("--all", action="store_true", help="Run harvest -> merge -> train -> evaluate end-to-end")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    # 1. Harvest Stage
    if args.harvest or args.all:
        from tools.harvester import harvest_species, WEAK_CLASSES, BOTANICAL_TAXON_MAP
        if args.species == "weak":
            targets = WEAK_CLASSES
        elif args.species == "all":
            targets = list(BOTANICAL_TAXON_MAP.keys())
        else:
            targets = [k for k in BOTANICAL_TAXON_MAP.keys() if args.species.lower() in k.lower()]
        print(f"\n[Stage 1/4] Harvesting {args.count} images for: {targets}")
        for t in targets:
            harvest_species(t, target_count=args.count)

    # 2. Merge Stage
    if args.merge or args.all:
        print("\n[Stage 2/4] Merging staged images into experimental dataset...")
        from tools.harvester import merge_staged_dataset
        merge_staged_dataset()

    # 3. Train Stage
    if args.train or args.all:
        print(f"\n[Stage 3/4] Launching enhanced training for {args.epochs} epochs...")
        from src.train import train_pipeline
        train_pipeline(epochs=args.epochs)

    # 4. Evaluate Stage
    if args.evaluate or args.all:
        print("\n[Stage 4/4] Evaluating and comparing against baseline...")
        from src.evaluate import evaluate_test_set
        evaluate_test_set()

if __name__ == "__main__":
    main()

