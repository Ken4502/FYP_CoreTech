"""
Configuration module for Enhanced Plant Identification Experimentation.
Supports Transfer Learning (ImageNet-1k pre-trained weights), RandAugment, and Class-Balanced Loss.
"""

from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
CORETECH_ROOT = BASE_DIR.parent

DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
STAGING_DIR = DATA_DIR / "staging"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_INDICES_PATH = ARTIFACTS_DIR / "class_indices.json"
ORIGINAL_BASELINE_MODEL = CORETECH_ROOT / "Plant_Identification" / "artifacts" / "best_model.pth"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.pth"
LAST_MODEL_PATH = ARTIFACTS_DIR / "last_model.pth"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

TRAINING_CURVES_PATH = REPORTS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"

# Model & Architecture
MODEL_NAME = "FloraNet-100 (Deepened & Widened 100-Species Botanical MobileNetV3)"

NUM_CLASSES = 100
if CLASS_INDICES_PATH.exists():
    import json
    try:
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as _f:
            _loaded_classes = json.load(_f).get("classes", [])
            if _loaded_classes:
                NUM_CLASSES = len(_loaded_classes)
    except Exception:
        pass

IMAGE_SIZE = 224
DROPOUT_RATE = 0.3

# Optimization & Regularization Strategy
USE_PRETRAINED = True       # Transfer Learning via MobileNetV3 Large ImageNet weights
USE_RAND_AUGMENT = True     # RandAugment (num_ops=2, magnitude=9)
USE_CLASS_WEIGHTS = True    # Inverse class frequency penalty for under-represented species
LABEL_SMOOTHING = 0.1

BATCH_SIZE = 64
NUM_WORKERS = 2
PIN_MEMORY = True if torch.cuda.is_available() else False

TOTAL_EPOCHS = 30
WARMUP_EPOCHS = 3
PEAK_LR = 1e-4              # Lower learning rate appropriate for transfer learning fine-tuning
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
PATIENCE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True if torch.cuda.is_available() else False

