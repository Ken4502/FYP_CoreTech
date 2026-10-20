"""
CoreTech Botanical Preprocessing Engine
Provides dedicated neural segmentation pipelines:
  - Method 1: Species Identification (Morphology & Margin Preservation)
  - Method 2: Pathology & Disease Diagnosis (Zero-Hole Lamina & Lesion Contrast)
"""

from .methods import PlantPreprocessors
from .normalization import (
    apply_white_balance,
    apply_clahe,
    clean_mask_components,
    composite_background,
    crop_to_plant_bbox,
    check_blur
)
from .evaluator import MaskEvaluator

__all__ = [
    "PlantPreprocessors",
    "apply_white_balance",
    "apply_clahe",
    "clean_mask_components",
    "composite_background",
    "crop_to_plant_bbox",
    "check_blur",
    "MaskEvaluator"
]

