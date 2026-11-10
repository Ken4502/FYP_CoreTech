"""
PhytoCare Layer 1 Botanical Plant Identifier
Integrates FloraNet-261 tournament botanical priors to identify plant species 
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
    """Loads and caches the FloraNet botanical model (261 species)."""
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

import numpy as np

_cached_verifier_model = None
_cached_verifier_tf = None
_cached_verifier_cats = None

PLANT_SYNSET_KEYWORDS = {
    'pot', 'flowerpot', 'vase', 'greenhouse', 'daisy', 'rose', 'corn', 'cardoon',
    'artichoke', 'broccoli', 'cabbage', 'cauliflower', 'zucchini', 'squash', 'cucumber',
    'bell pepper', 'mushroom', 'agaric', 'gyromitra', 'stinkhorn', 'earthstar',
    'hen-of-the-woods', 'bolete', 'orange', 'lemon', 'fig', 'pineapple', 'banana',
    'jackfruit', 'custard apple', 'pomegranate', 'hay', 'straw', 'acorn', 'rapeseed',
    'leaf beetle', 'cabbage butterfly', 'leafhopper', 'tree', 'plant', 'grasshopper'
}

UNAMBIGUOUS_NON_PLANT = {
    'laptop', 'notebook', 'cellular telephone', 'mouse', 'keyboard', 'coffee mug',
    'cup', 'dining table', 'desk', 'couch', 'bed', 'wardrobe', 'shoe', 'running shoe',
    'car', 'sports car', 'convertible', 'minivan', 'truck', 'bus', 'train', 'airplane',
    'bicycle', 'motorcycle', 'monitor', 'screen', 'television', 'remote control',
    'clock', 'wall clock', 'watch', 'book', 'paper', 'envelope', 'pen', 'pencil',
    'bottle', 'water bottle', 'beer bottle', 'wine bottle', 'plate', 'fork', 'knife',
    'spoon', 'guitar', 'violin', 'piano', 'drum', 'backpack', 'handbag', 'purse',
    'wallet', 'suit', 'jersey', 'jeans', 't-shirt', 'sandal', 'boot', 'chair', 'stool',
    'wardrobe', 'cabinet', 'refrigerator', 'microwave', 'toaster', 'iron', 'vacuum'
}

def load_open_set_verifier(device: torch.device = config.DEVICE):
    global _cached_verifier_model, _cached_verifier_tf, _cached_verifier_cats
    if _cached_verifier_model is not None:
        return _cached_verifier_model, _cached_verifier_tf, _cached_verifier_cats
    try:
        w = models.MobileNet_V3_Small_Weights.DEFAULT
        m = models.mobilenet_v3_small(weights=w).to(device).eval()
        _cached_verifier_model = m
        _cached_verifier_tf = w.transforms()
        _cached_verifier_cats = w.meta['categories']
        return _cached_verifier_model, _cached_verifier_tf, _cached_verifier_cats
    except Exception as e:
        print(f"[Warning] Could not load open-set verifier: {e}")
        return None, None, None

def check_plant_presence(img: Image.Image) -> tuple[bool, str, dict]:
    """
    Two-Stage Open-Set Botanical Verification Gate:
    1. Semantic Open-Set Verification (ImageNet 1,000 Everyday Classes via MobileNetV3)
    2. Quantitative Chlorophyll Foliage & Spatial Variance Analysis
    """
    # 1. ImageNet Semantic Object Classification
    v_model, v_tf, v_cats = load_open_set_verifier()
    plant_score = 0.0
    top_cat = "unknown"
    top_conf = 0.0
    
    if v_model is not None and v_tf is not None:
        try:
            t = v_tf(img).unsqueeze(0).to(config.DEVICE)
            with torch.no_grad():
                out = v_model(t)
                probs = F.softmax(out, dim=1)[0]
                top_p, top_i = probs.topk(10)
            
            top_cats = [v_cats[idx.item()] for idx in top_i]
            top_scores = [prob.item() for prob in top_p]
            top_cat = top_cats[0]
            top_conf = top_scores[0]
            
            plant_score = sum(s for c, s in zip(top_cats, top_scores) if any(k in c.lower() for k in PLANT_SYNSET_KEYWORDS))
            
            # Explicit Non-Plant Artifact Filter
            # If ImageNet confidently detects a known everyday man-made object (laptop, mug, mouse, etc.)
            if any(k in top_cat.lower() for k in UNAMBIGUOUS_NON_PLANT) and top_conf >= 0.22 and plant_score < 0.08:
                return False, f"Detected non-plant object: {top_cat} ({top_conf*100:.1f}%)", {
                    "top_detected_object": top_cat,
                    "object_confidence": round(top_conf * 100, 1),
                    "plant_score": round(plant_score * 100, 1)
                }
        except Exception as e:
            print(f"[Verifier Error] {e}")

    # 2. Quantitative Foliage Greenness & Surface Variance
    rgb = np.array(img.convert('RGB'), dtype=np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    grad_y = np.abs(np.diff(gray, axis=0))
    grad_x = np.abs(np.diff(gray, axis=1))
    texture_var = float(np.var(gray))
    texture_grad = float(np.mean(grad_x) + np.mean(grad_y))
    
    # Living Chlorophyll Foliage (G strictly greater than R and B, and non-dark)
    foliage_green = (g > r * 1.04) & (g > b * 1.08) & (g > 25)
    green_pct = float(np.mean(foliage_green) * 100.0)
    
    metrics = {
        "green_pct": round(green_pct, 1),
        "plant_score": round(plant_score * 100, 1),
        "top_detected_object": top_cat,
        "object_confidence": round(top_conf * 100, 1),
        "texture_var": round(texture_var, 1)
    }
    
    # Featureless blank surface rejection
    if texture_var < 15.0 or texture_grad < 0.8:
        return False, "Image appears to be a blank, featureless, or solid non-botanical surface", metrics
        
    # Plant Presence Requirement:
    # Must have living green foliar tissue (>= 5.0%) OR be classified as a botanical/floral/potted specimen (>= 8.0%)
    if green_pct < 5.0 and plant_score < 0.08:
        return False, f"No recognizable plant foliage or flowers detected (Subject appears to be: {top_cat})", metrics
        
    return True, "Botanical specimen verified", metrics

def identify_botanical_species(img: Image.Image, top_k: int = 5) -> Dict[str, Any]:
    """
    Runs Layer 1 Botanical Plant Identification.
    """
    is_plant, reason, metrics = check_plant_presence(img)
    if not is_plant:
        return {
            "is_plant": False,
            "rejection_reason": reason,
            "metrics": metrics,
            "top_species": "Non-Plant Object",
            "confidence": 0.0,
            "mapped_disease_group": None,
            "is_supported": False,
            "candidates": []
        }

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
        "is_plant": True,
        "top_species": top_species,
        "confidence": top_conf,
        "mapped_disease_group": mapped_group,
        "is_supported": mapped_group is not None,
        "candidates": candidates
    }

