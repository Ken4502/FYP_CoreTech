"""
Configuration module for PhytoCare Plant Disease Experimentation Engine.
"""

from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

CLASS_INDICES_PATH = ARTIFACTS_DIR / "class_indices.json"
DISEASE_KB_PATH = ARTIFACTS_DIR / "disease_knowledge_base.json"

BEST_MODEL_CONVNEXT = ARTIFACTS_DIR / "best_model_convnext.pth"
BEST_MODEL_SWIN = ARTIFACTS_DIR / "best_model_swin.pth"
BEST_MODEL_ENSEMBLE = ARTIFACTS_DIR / "best_model.pth"

# Pretrained botanical backbone from our FloraNet-250 Tournament Model
FLORANET_IDENTIFICATION_WEIGHTS = BASE_DIR.parent / "Plant_Identification_Experiment" / "artifacts" / "best_model.pth"
FLORANET_SWIN_WEIGHTS = BASE_DIR.parent / "Plant_Identification_Experiment" / "artifacts" / "best_model_swin_384.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 16
DEFAULT_RESOLUTION = 384

