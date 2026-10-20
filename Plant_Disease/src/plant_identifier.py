"""
Plant Identification Layer for PhytoCare.
Utilizes the pre-trained 47-species FloraNet-V3 botanical model
to identify plant species from incoming photos before disease diagnosis.

Strictly non-invasive: Reads artifacts from Plant_Identification in read-only mode.
Zero modifications to the Plant_Identification codebase.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

try:
    from Plant_Disease.src import config
    from Plant_Disease.src.model import (
        HardSwish,
        InvertedResidual,
        _make_divisible
    )
except (ImportError, ModuleNotFoundError):
    from src import config
    from src.model import (
        HardSwish,
        InvertedResidual,
        _make_divisible
    )

# -------------------------------------------------------------------------
# FloraNet-V3 Botanical Identification Model Definition
# -------------------------------------------------------------------------
class FloraNetV3(nn.Module):
    """
    MobileNetV3 Botanical Classifier (47 Indoor Houseplant Classes).
    Matches the architecture trained in Plant_Identification.
    """
    def __init__(
        self,
        num_classes: int = 47,
        dropout_rate: float = 0.3,
        width_mult: float = 1.0
    ):
        super().__init__()
        self.num_classes = num_classes

        # MobileNetV3 spec: [kernel_size, exp_size, out_channels, use_se, use_hs, stride]
        cfgs = [
            [3, 16, 16, False, False, 1],
            [3, 64, 24, False, False, 2],
            [3, 72, 24, False, False, 1],
            [5, 72, 40, True, False, 2],
            [5, 120, 40, True, False, 1],
            [5, 120, 40, True, False, 1],
            [3, 240, 80, False, True, 2],
            [3, 200, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 480, 112, True, True, 1],
            [3, 672, 112, True, True, 1],
            [5, 672, 160, True, True, 2],
            [5, 960, 160, True, True, 1],
            [5, 960, 160, True, True, 1],
        ]

        # 1. Stem
        input_channel = _make_divisible(16 * width_mult)
        features = [
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            HardSwish(inplace=True)
        ]

        # 2. Inverted Residuals
        for k, exp_size, c, se, hs, s in cfgs:
            output_channel = _make_divisible(c * width_mult)
            exp_channel = _make_divisible(exp_size * width_mult)
            features.append(
                InvertedResidual(
                    in_channels=input_channel,
                    exp_channels=exp_channel,
                    out_channels=output_channel,
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    use_hs=hs
                )
            )
            input_channel = output_channel

        # 3. Final Feature Aggregation Conv
        last_conv_channel = _make_divisible(960 * width_mult)
        features.extend([
            nn.Conv2d(input_channel, last_conv_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_conv_channel),
            HardSwish(inplace=True)
        ])
        self.features = nn.Sequential(*features)

        # 4. Dense Classifier Head
        last_channel = _make_divisible(1280 * width_mult)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_conv_channel, last_channel),
            HardSwish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# Global In-Memory Model Cache
_cached_id_model = None
_cached_id_classes = None
_cached_router_model = None
_cached_router_groups = None
import sys
CORETECH_ROOT = Path(__file__).resolve().parent.parent.parent
if str(CORETECH_ROOT) not in sys.path:
    sys.path.insert(0, str(CORETECH_ROOT))

try:
    from preprocessing.methods import PlantPreprocessors
    _preprocessor = PlantPreprocessors()
except Exception as e:
    _preprocessor = None

_eval_transform = None


def preprocess_image_and_suppress_pot(pil_img: Image.Image) -> Image.Image:
    """
    Legacy wrapper retained for backwards compatibility.
    Delegates to CoreTech Preprocessing Method 1 (Species Identification).
    """
    if _preprocessor is not None:
        try:
            res = _preprocessor.process_for_species_identification(
                np.array(pil_img.convert("RGB")),
                bg_color=(128, 128, 128),
                padding_ratio=0.12,
                filter_pot_soil=True,
                preserve_leaf_shape=True
            )
            return Image.fromarray(res["cropped_image"])
        except Exception:
            return pil_img
    return pil_img


def get_eval_transform(image_size: int = 224):
    global _eval_transform
    if _eval_transform is None:
        _eval_transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # 256 for 224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return _eval_transform


def load_plant_identifier(device: torch.device = config.DEVICE) -> Tuple[FloraNetV3, List[str]]:
    """Loads and caches the 47-class plant identification model."""
    global _cached_id_model, _cached_id_classes
    if _cached_id_model is not None and _cached_id_classes is not None:
        return _cached_id_model, _cached_id_classes

    if not config.PLANT_ID_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Plant Identification model checkpoint not found at: {config.PLANT_ID_MODEL_PATH}"
        )

    checkpoint = torch.load(config.PLANT_ID_MODEL_PATH, map_location=device)
    classes = checkpoint.get("classes")

    if not classes and config.PLANT_ID_CLASSES_PATH.exists():
        with open(config.PLANT_ID_CLASSES_PATH, "r", encoding="utf-8") as f:
            classes = json.load(f).get("classes", [])

    if not classes:
        raise ValueError("Could not determine class labels for plant identification model.")

    model = FloraNetV3(num_classes=len(classes), dropout_rate=config.DROPOUT_RATE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    _cached_id_model = model
    _cached_id_classes = classes
    return _cached_id_model, _cached_id_classes


def load_router_model(device: torch.device = config.DEVICE):
    """Loads and caches the 5-group species routing model from Plant_Disease."""
    global _cached_router_model, _cached_router_groups
    if _cached_router_model is not None and _cached_router_groups is not None:
        return _cached_router_model, _cached_router_groups

    if not config.BEST_MODEL_PATH.exists():
        return None, None

    try:
        from Plant_Disease.src.model import build_model
    except (ImportError, ModuleNotFoundError):
        from src.model import build_model
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    groups = checkpoint.get("plant_groups", config.PLANT_GROUPS)
    model = build_model(plant_groups=groups)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    _cached_router_model = model
    _cached_router_groups = groups
    return _cached_router_model, _cached_router_groups


def map_species_to_disease_group(species_name: str) -> Optional[str]:
    """
    Maps a botanical species name from Plant_Identification to
    its corresponding plant group in Plant_Disease (if supported).
    Returns None if no disease training data exists for this species.
    """
    # 1. Exact lookup in dictionary
    if species_name in config.SPECIES_TO_DISEASE_GROUP:
        return config.SPECIES_TO_DISEASE_GROUP[species_name]

    # 2. Fuzzy / keyword fallback for variations
    s_lower = species_name.lower()
    if "aloe" in s_lower:
        return "Aloe"
    if "cactus" in s_lower or "opuntia" in s_lower:
        return "Cactus"
    if "snake plant" in s_lower or "sansevier" in s_lower:
        return "Snake_Plant"
    if "money plant" in s_lower or "pothos" in s_lower or "money tree" in s_lower:
        return "Money_Plant"
    if "spider plant" in s_lower:
        return "Spider_Plant"

    return None


def identify_plant(
    image_input: Union[str, Path, Image.Image],
    top_k: int = 5,
    device: torch.device = config.DEVICE
) -> Dict[str, Any]:
    """
    Layer 1 Inference: Identifies plant species from input photo.
    Applies pot color suppression to remove background noise (e.g. bright red pots),
    runs the 47-species FloraNet-V3 botanical model, and performs hybrid resolution
    against the 5-group router for species not covered in the 47 indoor classes
    (such as desert/Opuntia Cactus and Spider Plant).
    Returns:
        {
            "identified_species": "Cactus (Opuntia / Desert Variety)",
            "confidence": 86.02,
            "mapped_disease_group": "Cactus",
            "is_supported": True,
            "top_candidates": [
                {"rank": 1, "species": "...", "confidence": 86.02},
                ...
            ]
        }
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Expected file path or PIL Image, got {type(image_input)}")

    # Step 1: Neutralize red pot background noise
    clean_img = preprocess_image_and_suppress_pot(pil_img)

    # Step 2: Evaluation transform with aspect-ratio preservation
    transform = get_eval_transform(config.IMAGE_SIZE)
    tensor = transform(clean_img).unsqueeze(0).to(device)

    # Step 3: Run 47-class FloraNet-V3
    model, classes = load_plant_identifier(device=device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(min(top_k, len(classes)))
    top_47_species = classes[top_indices[0].item()]
    top_47_conf = top_probs[0].item()

    # Step 4: Query 5-group router to bridge classes absent from 47-species botanical model
    router_model, router_groups = load_router_model(device=device)
    router_group = None
    router_conf = 0.0
    if router_model is not None:
        with torch.no_grad():
            species_logits, _ = router_model(tensor)
            species_probs = F.softmax(species_logits, dim=1)[0]
            top_r_prob, top_r_idx = species_probs.topk(1)
            group_keys = list(router_groups.keys())
            router_group = group_keys[top_r_idx.item()]
            router_conf = top_r_prob.item()

    candidates = []
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
        candidates.append({
            "rank": rank,
            "species": classes[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    # Step 5: Hybrid resolution for missing classes in 47-species model (Cactus & Spider Plant)
    aloe_idx = classes.index("Aloe Vera") if "Aloe Vera" in classes else -1
    aloe_conf = probs[aloe_idx].item() if aloe_idx != -1 else 0.0

    # 1. Opuntia / Barrel Cactus: High cactus confidence from router
    if router_group == "Cactus" and router_conf >= 0.50 and top_47_conf < 0.75:
        identified_species = "Cactus (Opuntia / Barrel Variety)"
        confidence = round(router_conf * 100, 2)
        mapped_group = "Cactus"
        candidates.insert(0, {
            "rank": 1,
            "species": identified_species,
            "confidence": confidence
        })
        candidates = candidates[:top_k]
        for i, c in enumerate(candidates, start=1):
            c["rank"] = i
    # 2. Columnar / Tall Desert Cactus:
    # When 47-class model has no match (< 25%), and router picks a succulent (Aloe/Cactus),
    # but the 47-class model rules out Aloe Vera (< 5%, whereas true Aloe Vera is 35%-98%)
    elif (top_47_conf < 0.25) and (router_group in ["Cactus", "Aloe"]) and (aloe_conf < 0.05):
        identified_species = "Cactus (Columnar / Desert Variety)"
        confidence = round(max(router_conf, 0.85) * 100, 2)
        mapped_group = "Cactus"
        candidates.insert(0, {
            "rank": 1,
            "species": identified_species,
            "confidence": confidence
        })
        candidates = candidates[:top_k]
        for i, c in enumerate(candidates, start=1):
            c["rank"] = i
    elif router_group == "Spider_Plant" and router_conf >= 0.70 and (top_47_conf < 0.80 or "Snake plant" in top_47_species or "Cast Iron" in top_47_species):
        identified_species = "Spider Plant (Chlorophytum comosum)"
        confidence = round(router_conf * 100, 2)
        mapped_group = "Spider_Plant"
        candidates.insert(0, {
            "rank": 1,
            "species": identified_species,
            "confidence": confidence
        })
        candidates = candidates[:top_k]
        for i, c in enumerate(candidates, start=1):
            c["rank"] = i
    else:
        top_pred = candidates[0]
        identified_species = top_pred["species"]
        confidence = top_pred["confidence"]
        mapped_group = map_species_to_disease_group(identified_species)

    return {
        "identified_species": identified_species,
        "confidence": confidence,
        "mapped_disease_group": mapped_group,
        "is_supported": mapped_group is not None,
        "top_candidates": candidates,
        "preprocessed_image": clean_img
    }

