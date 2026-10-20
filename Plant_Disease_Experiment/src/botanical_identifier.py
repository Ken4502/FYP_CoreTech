"""
PhytoCare Layer 1 Botanical Plant Identifier
Integrates FloraNet-250 tournament botanical priors to identify plant species 
before executing intra-species pathology diagnosis.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config

_cached_flora_model = None
_cached_flora_classes = None

def load_botanical_identifier(device: torch.device = config.DEVICE):
    """Loads and caches the 250-species FloraNet botanical model."""
    global _cached_flora_model, _cached_flora_classes

    if _cached_flora_model is not None and _cached_flora_classes is not None:
        return _cached_flora_model, _cached_flora_classes

    weights_path = config.FLORANET_IDENTIFICATION_WEIGHTS
    if not weights_path.exists():
        print(f"[Warning] FloraNet weights not found at {weights_path}")
        return None, None

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    _cached_flora_classes = ckpt.get("classes", [])
    num_classes = len(_cached_flora_classes)

    model = models.convnext_small(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    _cached_flora_model = model
    return _cached_flora_model, _cached_flora_classes

def map_species_to_disease_group(species_name: str) -> Optional[str]:
    """
    Maps a 250-species botanical taxon name to one of the 5 supported pathology groups:
      - Aloe
      - Cactus
      - Money_Plant
      - Snake_Plant
      - Spider_Plant
    Returns None if the species is outside the supported pathology dataset.
    """
    s = species_name.lower()

    # 1. Aloe Vera & Succulents
    if any(k in s for k in ["aloe", "haworthia", "lace aloe", "tiger aloe", "torch aloe"]):
        return "Aloe"

    # 2. Cacti
    if any(k in s for k in [
        "cactus", "opuntia", "pachycereus", "echinocactus", "mammillaria",
        "saguaro", "prickly pear", "cereus", "cholla", "barrel cactus"
    ]):
        return "Cactus"

    # 3. Money Plant / Pothos
    if any(k in s for k in [
        "money plant", "pothos", "epipremnum", "devil's ivy", "devils ivy",
        "scindapsus", "golden pothos"
    ]):
        return "Money_Plant"

    # 4. Snake Plant
    if any(k in s for k in [
        "snake plant", "sansevieria", "sanseviera", "dracaena trifasciata",
        "mother-in-law", "mother in law", "viper"
    ]):
        return "Snake_Plant"

    # 5. Spider Plant
    if any(k in s for k in [
        "spider plant", "chlorophytum", "airplane plant", "ribbon plant"
    ]):
        return "Spider_Plant"

    return None

def identify_botanical_species(img: Image.Image, top_k: int = 5) -> Dict[str, Any]:
    """
    Runs Layer 1 Botanical Plant Identification.
    """
    model, classes = load_botanical_identifier()
    if model is None:
        return {
            "top_species": "Unknown Botanical Specimen",
            "confidence": 0.0,
            "mapped_disease_group": None,
            "is_supported": False,
            "candidates": []
        }

    eval_tf = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = eval_tf(img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)[0]

    top_probs, top_indices = probs.topk(min(top_k, len(classes)))

    candidates = []
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
        candidates.append({
            "rank": rank,
            "species": classes[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    top_pred = candidates[0]
    top_species = top_pred["species"]
    top_conf = top_pred["confidence"]
    mapped_group = map_species_to_disease_group(top_species)

    return {
        "top_species": top_species,
        "confidence": top_conf,
        "mapped_disease_group": mapped_group,
        "is_supported": mapped_group is not None,
        "candidates": candidates
    }

