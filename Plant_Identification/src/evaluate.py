"""
Evaluation script for plant identification model on test set.
Computes Top-1, Top-5 accuracy, per-class F1-scores, and generates confusion matrix.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
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
    print(f"=== Evaluating on Test Set using Device: {config.DEVICE} ===")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {config.BEST_MODEL_PATH}. Train the model first.")

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()

    _, _, test_loader, _ = get_dataloaders()

    all_preds = []
    all_targets = []
    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    print("Running inference across test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(config.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # Top 1
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

            top1_correct += (preds.cpu() == labels).sum().item()

            # Top 5
            maxk = min(5, num_classes)
            _, top5_preds = probs.topk(maxk, dim=1, largest=True, sorted=True)
            labels_expanded = labels.view(-1, 1).expand_as(top5_preds.cpu())
            top5_correct += (top5_preds.cpu() == labels_expanded).sum().item()

            total_samples += labels.size(0)

    top1_acc = top1_correct / total_samples
    top5_acc = top5_correct / total_samples

    print(f"\n================ Test Set Results ================")
    print(f"  Total Test Samples:  {total_samples}")
    print(f"  Top-1 Accuracy:      {top1_acc*100:.2f}%")
    print(f"  Top-5 Accuracy:      {top5_acc*100:.2f}%")
    print(f"==================================================")

    # Classification Report
    clf_report = classification_report(
        all_targets,
        all_preds,
        target_names=classes,
        digits=4,
        output_dict=True
    )
    clf_text = classification_report(
        all_targets,
        all_preds,
        target_names=classes,
        digits=4
    )

    with open(config.CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(clf_text)
    print(f"Saved classification report to {config.CLASSIFICATION_REPORT_PATH}")

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(24, 20))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=[c[:16] for c in classes],
        yticklabels=[c[:16] for c in classes]
    )
    plt.title("Confusion Matrix - Plant Identification (MobileNetV3)", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {config.CONFUSION_MATRIX_PATH}")

    # Save metrics JSON
    metrics = {
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "macro_avg_f1": round(clf_report["macro avg"]["f1-score"], 4),
        "weighted_avg_f1": round(clf_report["weighted avg"]["f1-score"], 4),
        "total_test_samples": total_samples,
        "classes_count": num_classes
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved summary metrics to {config.METRICS_PATH}")

    return metrics

if __name__ == "__main__":
    evaluate_test_set()
