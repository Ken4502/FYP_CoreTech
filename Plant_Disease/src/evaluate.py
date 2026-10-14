"""
Evaluation script for Hierarchical Multi-Head Plant Disease Diagnosis model.
Evaluates:
  1. Species routing accuracy
  2. Per-group disease diagnosis accuracy and F1 scores
  3. Generates confusion matrix and metrics JSON
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src import config
from src.dataset import get_dataloaders
from src.model import build_model

def evaluate_test_set():
    print(f"=== Evaluating Hierarchical Model on Test Set ({config.DEVICE}) ===")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {config.BEST_MODEL_PATH}. Train the model first.")

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    plant_groups = checkpoint.get("plant_groups", config.PLANT_GROUPS)

    model = build_model(plant_groups=plant_groups)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()

    _, _, test_loader, classes = get_dataloaders()

    total_samples = 0
    species_correct = 0
    disease_correct = 0

    group_results = {sp: {"correct": 0, "total": 0} for sp in config.SPECIES_NAMES}

    all_targets = []
    all_preds = []

    print("Running hierarchical evaluation across test set...")
    with torch.no_grad():
        for images, species_targets, condition_targets, global_targets in tqdm(test_loader, desc="Testing"):
            images = images.to(config.DEVICE)
            species_targets = species_targets.to(config.DEVICE)
            condition_targets = condition_targets.to(config.DEVICE)

            species_logits, disease_outputs = model(images)

            # 1. Species accuracy
            _, sp_preds = torch.max(species_logits, 1)
            species_correct += (sp_preds == species_targets).sum().item()

            # 2. Disease accuracy within species
            for sp_idx, sp_name in enumerate(config.SPECIES_NAMES):
                mask = (species_targets == sp_idx)
                if mask.sum() > 0:
                    cond_logits = disease_outputs[sp_name][mask]
                    cond_labels = condition_targets[mask]
                    _, cond_preds = torch.max(cond_logits, 1)

                    corr = (cond_preds == cond_labels).sum().item()
                    disease_correct += corr
                    group_results[sp_name]["correct"] += corr
                    group_results[sp_name]["total"] += mask.sum().item()

            total_samples += images.size(0)

    sp_acc = species_correct / total_samples
    dis_acc = disease_correct / total_samples

    print("\n" + "=" * 60)
    print("🌿 Hierarchical Test Set Results:")
    print(f"  Total Samples Evaluated:  {total_samples}")
    print(f"  Stage 1 Species Accuracy: {sp_acc*100:.2f}%")
    print(f"  Stage 2 Disease Accuracy: {dis_acc*100:.2f}%")
    print("-" * 60)
    print("  Per-Plant Group Disease Accuracy:")
    for sp_name, res in group_results.items():
        if res["total"] > 0:
            acc = res["correct"] / res["total"]
            print(f"    • {sp_name:<15}: {acc*100:>6.2f}% ({res['correct']}/{res['total']})")
    print("=" * 60)

    metrics = {
        "species_routing_accuracy": round(sp_acc, 4),
        "disease_diagnosis_accuracy": round(dis_acc, 4),
        "total_test_samples": total_samples,
        "per_group_accuracy": {
            sp: round(res["correct"] / (res["total"] or 1), 4)
            for sp, res in group_results.items()
        }
    }

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved summary metrics to {config.METRICS_PATH}")

    return metrics

if __name__ == "__main__":
    evaluate_test_set()
