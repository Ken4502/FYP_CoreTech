"""Central configuration for CoreTech v2 High-Reliability Online Pipeline."""
from pathlib import Path
import json
import torch

V2_ROOT = Path(__file__).resolve().parent.parent
CORETECH_ROOT = V2_ROOT.parent
FYP_ROOT = CORETECH_ROOT.parent

CONFIGS_DIR = V2_ROOT / "configs"
DATA_DIR = V2_ROOT / "data"
ARTIFACTS_DIR = V2_ROOT / "artifacts"
REPORTS_DIR = V2_ROOT / "reports"

for d in [CONFIGS_DIR, DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset subdirectories
ID_DATA_DIR = DATA_DIR / "botanical_id"           # train / val / test
PATH_DATA_DIR = DATA_DIR / "surface_pathology"    # train / val / test
SEG_DATA_DIR = DATA_DIR / "instance_seg"          # YOLO-Seg format
LESION_DATA_DIR = DATA_DIR / "lesion_det"         # YOLO-Det/Seg format
NON_PLANT_DIR = DATA_DIR / "non_plant_hard_neg"   # Indoor/green distractors

TAXONOMY_PATH = CONFIGS_DIR / "botanical_taxonomy_v2.json"

def load_taxonomy():
    if TAXONOMY_PATH.exists():
        return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    return {}

# Unified 19-Class Surface Health, Pathology & Abiotic Stress Taxonomy
SURFACE_PATHOLOGY_CLASSES = [
    "Healthy_Foliage",
    "Fungal_Leaf_Spot_Anthracnose",
    "Bacterial_Wilt_Soft_Rot",
    "Botrytis_Gray_Mold",
    "Powdery_Mildew",
    "Downy_Mildew",
    "Rust_Fungi",
    "Sooty_Mold",
    "Leaf_Miners",
    "Spider_Mites",
    "Mealybugs_Cochineal",
    "Scale_Insects",
    "Aphids",
    "Thrips_Damage",
    "Whiteflies",
    "Abiotic_Dehydration_Underwatering",
    "Abiotic_Overwatering_Edema",
    "Abiotic_Sunburn_Scald",
    "Abiotic_Nutrient_Chlorosis",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# Calibrated Uncertainty & Quality Gate Thresholds (tuned on val during Phase 5)
QUALITY_THRESHOLDS = {
    "min_short_side_px": 224,
    "min_laplacian_var": 32.0,       # Focus sharpness minimum inside foreground
    "min_tenengrad_mean": 8.0,       # Gradient energy minimum
    "max_dark_clipped_pct": 65.0,    # % near-black pixels
    "max_bright_clipped_pct": 55.0,  # % overexposed blown pixels
    "min_plant_coverage_pct": 8.0,   # Dominant plant mask coverage floor
}

SPECIES_CONF_THRESHOLD = 0.55        # Below this -> Genus/Family fallback + angle guidance
SPECIES_MARGIN_THRESHOLD = 0.12      # Top1 - Top2 calibrated probability gap
PATHOLOGY_HEALTH_OVERRIDE = 0.65     # Healthy confidence floor to suppress noisy disease calls
