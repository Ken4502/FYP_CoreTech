"""
Evaluation & Comparative Benchmark Script for Plant Identification Experimentation.
Evaluates the trained model on test data, generates detailed metrics, and produces
an ablation comparison against the original baseline model.
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import get_dataloaders
from src.model import build_model

def evaluate_test_set():
    print("=" * 70)
    print("📊 EVALUATING MODEL ON EXPERIMENTAL TEST SET")
    print(f"Model checkpoint: {config.BEST_MODEL_PATH}")
    print("=" * 70)

    if not config.BEST_MODEL_PATH.exists():
        print(f"Error: Model checkpoint not found at {config.BEST_MODEL_PATH}. Run training first.")
        return

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    classes = checkpoint["classes"]
    use_pretrained = checkpoint.get("config", {}).get("pretrained", True)

    model = build_model(num_classes=len(classes), pretrained=use_pretrained)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()

    _, _, test_loader, _, _ = get_dataloaders()

    all_preds = []
    all_targets = []
    top5_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(config.DEVICE, non_blocking=True)
            outputs = model(images)

            _, top1_p = outputs.max(1)
            all_preds.extend(top1_p.cpu().numpy())
            all_targets.extend(targets.numpy())

            _, top5_p = outputs.topk(min(5, outputs.size(1)), dim=1)
            top5_correct += top5_p.eq(targets.to(config.DEVICE).view(-1, 1)).sum().item()
            total_samples += targets.size(0)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    top1_acc = np.mean(all_preds == all_targets)
    top5_acc = top5_correct / total_samples
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted")

    print(f"\n📈 Test Results:")
    print(f"   Top-1 Accuracy:  {top1_acc*100:.2f}%")
    print(f"   Top-5 Accuracy:  {top5_acc*100:.2f}%")
    print(f"   Macro Avg F1:    {macro_f1*100:.2f}%")
    print(f"   Weighted Avg F1: {weighted_f1*100:.2f}%")

    # Generate classification report
    report_str = classification_report(
        all_targets,
        all_preds,
        target_names=classes,
        digits=4
    )
    with open(config.CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_str)

    # Save metrics JSON
    metrics_summary = {
        "top1_accuracy": round(float(top1_acc), 4),
        "top5_accuracy": round(float(top5_acc), 4),
        "macro_avg_f1": round(float(macro_f1), 4),
        "weighted_avg_f1": round(float(weighted_f1), 4),
        "total_test_samples": total_samples,
        "classes_count": len(classes)
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Compare with Baseline if available
    original_metrics_path = config.ORIGINAL_BASELINE_MODEL.parent / "metrics.json"
    if original_metrics_path.exists():
        with open(original_metrics_path, "r", encoding="utf-8") as f:
            base_m = json.load(f)
        print("\n" + "=" * 70)
        print("🔍 COMPARISON AGAINST ORIGINAL BASELINE MODEL")
        print("-" * 70)
        print(f"Metric          Original Baseline       Experimental Model      Delta")
        print("-" * 70)
        d_top1 = (top1_acc - base_m['top1_accuracy']) * 100
        d_top5 = (top5_acc - base_m['top5_accuracy']) * 100
        d_f1 = (macro_f1 - base_m['macro_avg_f1']) * 100
        print(f"Top-1 Accuracy: {base_m['top1_accuracy']*100:>16.2f}% {top1_acc*100:>23.2f}% {d_top1:>+10.2f}%")
        print(f"Top-5 Accuracy: {base_m['top5_accuracy']*100:>16.2f}% {top5_acc*100:>23.2f}% {d_top5:>+10.2f}%")
        print(f"Macro Avg F1:   {base_m['macro_avg_f1']*100:>16.2f}% {macro_f1*100:>23.2f}% {d_f1:>+10.2f}%")
        print("=" * 70)

if __name__ == "__main__":
    evaluate_test_set()
