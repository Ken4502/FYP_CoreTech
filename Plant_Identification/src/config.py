"""
Configuration module for Plant Identification Model Training and Inference.
"""

from pathlib import Path
import torch

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

CLASS_INDICES_PATH = ARTIFACTS_DIR / "class_indices.json"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.pth"
LAST_MODEL_PATH = ARTIFACTS_DIR / "last_model.pth"
ONNX_MODEL_PATH = ARTIFACTS_DIR / "model.onnx"
TORCHSCRIPT_MODEL_PATH = ARTIFACTS_DIR / "model.pt"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

TRAINING_CURVES_PATH = REPORTS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"

# Model Hyperparameters
MODEL_NAME = "FloraNet-V3 (Bespoke MobileNetV3)"
NUM_CLASSES = 47
IMAGE_SIZE = 224
DROPOUT_RATE = 0.3

# Training Hyperparameters (From-Scratch Regimen)
BATCH_SIZE = 64
NUM_WORKERS = 2  # Safe setting for Windows multiprocessing
PIN_MEMORY = True if torch.cuda.is_available() else False

TOTAL_EPOCHS = 28       # Full from-scratch training epochs
WARMUP_EPOCHS = 3      # Linear learning rate warmup
PEAK_LR = 1e-3         # Maximum learning rate after warmup
MIN_LR = 1e-6          # Minimum learning rate for cosine annealing
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
PATIENCE = 7           # Early stopping patience

# Hardware Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True if torch.cuda.is_available() else False

