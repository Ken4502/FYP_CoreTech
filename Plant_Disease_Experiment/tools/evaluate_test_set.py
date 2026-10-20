"""
PhytoCare Ultra Comprehensive Test Set Evaluator
Evaluates the trained ConvNeXt-Small (384px) model with 5-Crop TTA across all 2,011 test images.
Generates classification report and confusion matrix for direct comparison with baseline.
"""

import sys
import os
import time
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.predict import diagnose_plant, load_diagnostic_system

TEST_DIR = config.TEST_DIR
REPORTS_DIR = config.REPORTS_DIR
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_TXT = REPORTS_DIR / "classification_report.txt"
CM_PNG = REPORTS_DIR / "confusion_matrix.png"

def main():
    print("=" * 70)
    print("EVALUATING PHYTONET ULTRA ON TEST DATASET (2,011 SPECIMENS)")
    print(f"• Hardware:   {config.DEVICE}")
    print(f"• Checkpoint: {config.BEST_MODEL_ENSEMBLE}")
    print(f"• Resolution: 384x384 with 5-Crop Test-Time Augmentation (TTA)")
    print("=" * 70)

    model, classes, spot_detector, kb = load_diagnostic_system(reload=True)
    temp = 0.81

    # Load test dataset
    _, eval_tf = transforms.Resize((416, 416)), transforms.CenterCrop(384)
    test_ds = datasets.ImageFolder(str(TEST_DIR))
    print(f"[Dataset] Found {len(test_ds)} test images across {len(test_ds.classes)} classes.")

    all_preds = []
    all_targets = []
    t0 = time.time()

    print("[Evaluation] Computing predictions with 5-crop TTA...")

    for i, (img, target) in enumerate(test_ds):
        diag = diagnose_plant(img, top_k=1, use_tta=True, detect_spots=False)
        pred_cls = diag["predictions"][0]["class_name"]
        pred_idx = classes.index(pred_cls) if pred_cls in classes else 0
        all_preds.append(pred_idx)
        all_targets.append(target)

        if (i + 1) % 200 == 0 or (i + 1) == len(test_ds):
            acc_so_far = np.mean(np.array(all_preds) == np.array(all_targets)) * 100
            print(f"  [{i+1:4d}/{len(test_ds)}] Running Accuracy: {acc_so_far:.2f}%")

    elapsed = round(time.time() - t0, 1)
    acc = np.mean(np.array(all_preds) == np.array(all_targets)) * 100

    # 1. Classification Report
    target_names = test_ds.classes
    cls_report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT ON TEST SET:")
    print("=" * 70)
    print(cls_report)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"PhytoCare Ultra Test Set Performance (Evaluation Duration: {elapsed}s)\n")
        f.write(f"Overall Accuracy: {acc:.2f}%\n\n")
        f.write(cls_report)

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=target_names,
        yticklabels=target_names,
        title=f'PhytoCare Ultra Confusion Matrix (Test Accuracy: {acc:.2f}%)',
        ylabel='True Biological Condition',
        xlabel='Predicted Condition'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(CM_PNG, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[Saved] Report saved to: {REPORT_TXT}")
    print(f"[Saved] Confusion Matrix plot saved to: {CM_PNG}")

if __name__ == "__main__":
    main()

