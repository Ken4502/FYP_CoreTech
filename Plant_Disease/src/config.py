"""
Configuration module for Hierarchical Plant Disease Diagnosis and Spot Detection.
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
PLANT_GROUPS_PATH = ARTIFACTS_DIR / "plant_groups.json"
DISEASE_KB_PATH = ARTIFACTS_DIR / "disease_knowledge_base.json"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.pth"
LAST_MODEL_PATH = ARTIFACTS_DIR / "last_model.pth"
ONNX_MODEL_PATH = ARTIFACTS_DIR / "model.onnx"
TORCHSCRIPT_MODEL_PATH = ARTIFACTS_DIR / "model.pt"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

PRETRAINED_BACKBONE_PATH = BASE_DIR.parent / "Plant_Identification" / "artifacts" / "best_model.pth"
PLANT_IDENTIFICATION_DIR = BASE_DIR.parent / "Plant_Identification"
PLANT_ID_MODEL_PATH = PLANT_IDENTIFICATION_DIR / "artifacts" / "best_model.pth"
PLANT_ID_CLASSES_PATH = PLANT_IDENTIFICATION_DIR / "artifacts" / "class_indices.json"

TRAINING_CURVES_PATH = REPORTS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"

# Mapping from Plant Identification 47 classes to Plant Disease groups
SPECIES_TO_DISEASE_GROUP = {
    "Aloe Vera": "Aloe",
    "Christmas Cactus (Schlumbergera bridgesii)": "Cactus",
    "Chinese Money Plant (Pilea peperomioides)": "Money_Plant",
    "Pothos (Ivy arum)": "Money_Plant",
    "Money Tree (Pachira aquatica)": "Money_Plant",
    "Snake plant (Sanseviera)": "Snake_Plant",
    # Direct alias matches
    "Aloe": "Aloe",
    "Cactus": "Cactus",
    "Money_Plant": "Money_Plant",
    "Snake_Plant": "Snake_Plant",
    "Spider_Plant": "Spider_Plant"
}

SUPPORTED_DISEASE_SPECIES_DISPLAY = [
    "Aloe Vera",
    "Cactus (Opuntia)",
    "Money Plant (Pothos / Pilea)",
    "Snake Plant (Sansevieria)",
    "Spider Plant"
]

# Plant Species Groups and their specific conditions
PLANT_GROUPS = {
    "Aloe": [
        "Healthy",
        "Anthracnose",
        "LeafSpot",
        "Rust",
        "Sunburn"
    ],
    "Cactus": [
        "Healthy",
        "Dactylopius_Opuntia"
    ],
    "Money_Plant": [
        "Healthy",
        "Bacterial_wilt_disease",
        "Manganese_Toxicity"
    ],
    "Snake_Plant": [
        "Healthy",
        "Anthracnose",
        "Leaf_Withering"
    ],
    "Spider_Plant": [
        "Healthy",
        "Fungal_leaf_spot",
        "Leaf_Tip_Necrosis"
    ]
}

SPECIES_NAMES = list(PLANT_GROUPS.keys())
NUM_SPECIES = len(SPECIES_NAMES)

# Model Hyperparameters
MODEL_NAME = "FloraNet-Hierarchical (MobileNetV3 Grouped Pathology & Spot Detector)"
IMAGE_SIZE = 224
DROPOUT_RATE = 0.3

# Training Hyperparameters
BATCH_SIZE = 64
NUM_WORKERS = 2
PIN_MEMORY = True if torch.cuda.is_available() else False

TOTAL_EPOCHS = 8        # Fast convergence with hierarchical multi-head transfer
WARMUP_EPOCHS = 1
PEAK_LR = 5e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
PATIENCE = 4

# Lesion Spot Detection Threshold
SPOT_ACTIVATION_THRESHOLD = 0.45
MIN_SPOT_AREA_PIXELS = 15

# Hardware Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True if torch.cuda.is_available() else False
