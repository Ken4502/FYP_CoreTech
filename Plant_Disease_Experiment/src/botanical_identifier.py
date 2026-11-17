"""
PhytoCare Layer 1 Botanical Plant Identifier
Integrates FloraNet-261 tournament botanical priors to identify plant species 
before executing intra-species pathology diagnosis.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
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

def get_botanical_family(species_name: str) -> str:
    """Categorizes botanical species into plant families / morphological groups."""
    s = species_name.lower()
    if any(k in s for k in ['cactus', 'opuntia', 'pachycereus', 'disocactus', 'echinocactus', 'schlumbergera']):
        return 'Cactaceae (Cactus Family)'
    if any(k in s for k in ['aloe', 'haworthiopsis', 'haworthia']):
        return 'Asphodelaceae (Aloe & Haworthia Family)'
    if any(k in s for k in ['sanseviera', 'dracaena angolensis']) or ('snake plant' in s and 'rattlesnake' not in s):
        return 'Dracaenaceae (Snake Plant Group)'
    if any(k in s for k in ['pothos', 'scindapsus']):
        return 'Epipremnum (Pothos / Money Plant Group)'
    if any(k in s for k in ['monstera', 'philodendron', 'alocasia', 'anthurium', 'syngonium', 'dieffenbachia', 'aglaonema', 'zamioculcas', 'zantedeschia', 'caladium']):
        return 'Araceae (Aroid Family)'
    if any(k in s for k in ['crassula', 'echeveria', 'graptopetalum', 'kalanchoe', 'sedum']):
        return 'Crassulaceae (Fleshy Succulents)'
    if any(k in s for k in ['ficus', 'fig ']):
        return 'Moraceae (Fig & Rubber Tree Family)'
    if any(k in s for k in ['palm', 'dypsis', 'chamaedorea', 'howea', 'ravenea', 'caryota']):
        return 'Arecaceae (Palm Family)'
    if any(k in s for k in ['calathea', 'maranta', 'goeppertia', 'stromanthe', 'ctenanthe', 'rattlesnake']):
        return 'Marantaceae (Prayer Plant Family)'
    if any(k in s for k in ['tomato', 'potato', 'eggplant', 'pepper', 'capsicum', 'solanum']):
        return 'Solanaceae (Nightshade Family)'
    if any(k in s for k in ['basil', 'mint', 'mentha', 'salvia', 'rosemary', 'thyme', 'oregano', 'lavender', 'nepeta', 'coleus', 'melissa', 'majorana', 'glechoma', 'lamium']):
        return 'Lamiaceae (Mint & Herb Family)'
    if any(k in s for k in ['fern', 'nephrolepis', 'asplenium', 'platycerium', 'adiantum', 'microsorum']):
        return 'Polypodiopsida (Fern Family)'
    if any(k in s for k in ['begonia']):
        return 'Begoniaceae (Begonia Family)'
    if any(k in s for k in ['citrus', 'murraya', 'lemon', 'lime', 'orange']):
        return 'Rutaceae (Citrus Family)'
    if any(k in s for k in ['cucumber', 'pumpkin', 'zucchini', 'watermelon', 'cantaloupe', 'momordica', 'cucurbita', 'cucumis', 'citrullus']):
        return 'Cucurbitaceae (Gourd & Squash Family)'
    if any(k in s for k in ['cabbage', 'broccoli', 'kale', 'radish', 'capsella', 'brassica', 'raphanus']):
        return 'Brassicaceae (Crucifer Family)'
    if any(k in s for k in ['pea', 'bean', 'clover', 'medicago', 'phaseolus', 'pisum', 'trifolium', 'clitoria']):
        return 'Fabaceae (Legume Family)'
    if any(k in s for k in ['sunflower', 'daisy', 'zinnia', 'cosmos', 'dahlia', 'marigold', 'dandelion', 'coneflower', 'chamomile', 'cirsium', 'tagetes', 'rudbeckia', 'taraxacum', 'echinacea', 'artemisia']):
        return 'Asteraceae (Sunflower & Daisy Family)'
    if any(k in s for k in ['rose', 'strawberry', 'apple', 'blackberry', 'raspberry', 'peach', 'spiraea', 'rosa', 'fragaria', 'malus', 'rubus', 'prunus']):
        return 'Rosaceae (Rose & Fruit Family)'
    if any(k in s for k in ['peperomia', 'piper']):
        return 'Piperaceae (Peperomia Family)'
    return f'Other Taxa ({species_name})'

def aggregate_family_softmax(probs: torch.Tensor, classes: List[str]) -> Tuple[str, str, float, List[Dict[str, Any]]]:
    """
    Experimental Family-Level Softmax Aggregation:
    Sums probabilities belonging to the same botanical family / morphological genus,
    preventing fragmented predictions across fine-grained 261 classes.
    """
    family_sums = {}
    family_exemplars = {}

    for idx, p in enumerate(probs):
        cls_name = classes[idx]
        fam = get_botanical_family(cls_name)
        val = p.item()
        family_sums[fam] = family_sums.get(fam, 0.0) + val
        if fam not in family_exemplars or val > family_exemplars[fam][1]:
            family_exemplars[fam] = (cls_name, val)

    sorted_families = sorted(family_sums.items(), key=lambda x: x[1], reverse=True)
    top_family, top_family_score = sorted_families[0]
    top_exemplar, ex_score = family_exemplars[top_family]

    family_rankings = [
        {
            "rank": r,
            "family": fam,
            "confidence": round(score * 100, 2),
            "top_exemplar": family_exemplars[fam][0]
        }
        for r, (fam, score) in enumerate(sorted_families[:5], start=1)
    ]

    return top_family, top_exemplar, top_family_score, family_rankings

import numpy as np

from src.plant_verifier import verify_is_plant

def check_plant_presence(img: Image.Image, explicit_api_key: Optional[str] = None) -> tuple[bool, str, dict]:
    """
    Two-Stage Open-Set Botanical Verification Gate:
    1. Google Gemini Multimodal Vision API (if GEMINI_API_KEY is configured)
    2. Local Semantic Open-Set Rejector (MobileNetV3 1000-class)
    """
    return verify_is_plant(img, explicit_api_key=explicit_api_key)

def identify_botanical_species(img: Image.Image, top_k: int = 5, gemini_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs Layer 1 Botanical Plant Identification.
    """
    is_plant, reason, metrics = check_plant_presence(img, explicit_api_key=gemini_api_key)
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

    # ---------------------------------------------------------
    # Experimental Family-Level Softmax Aggregation (Toggleable)
    # ---------------------------------------------------------
    use_family_agg = getattr(config, "TEST_FAMILY_LEVEL_AGGREGATION", False)

    if use_family_agg:
        top_fam, top_exemplar, top_fam_score, fam_rankings = aggregate_family_softmax(probs, classes)

        # Map family directly to pathology group
        if "Cactaceae" in top_fam:
            mapped_group = "Cactus"
            top_species = top_exemplar
            top_conf = round(top_fam_score * 100, 2)
            detected_mode = f"Family-Level Softmax Aggregation ({top_fam})"
        elif "Asphodelaceae" in top_fam:
            mapped_group = "Aloe"
            top_species = top_exemplar
            top_conf = round(top_fam_score * 100, 2)
            detected_mode = f"Family-Level Softmax Aggregation ({top_fam})"
        elif "Dracaenaceae" in top_fam:
            mapped_group = "Snake_Plant"
            top_species = top_exemplar
            top_conf = round(top_fam_score * 100, 2)
            detected_mode = f"Family-Level Softmax Aggregation ({top_fam})"
        elif "Epipremnum" in top_fam:
            mapped_group = "Money_Plant"
            top_species = top_exemplar
            top_conf = round(top_fam_score * 100, 2)
            detected_mode = f"Family-Level Softmax Aggregation ({top_fam})"
        else:
            mapped_group = map_species_to_disease_group(top_exemplar)
            top_species = top_exemplar
            top_conf = round(top_fam_score * 100, 2)
            detected_mode = f"Family-Level Softmax Aggregation ({top_fam})"

        return {
            "is_plant": True,
            "top_species": top_species,
            "confidence": top_conf,
            "mapped_disease_group": mapped_group,
            "is_supported": mapped_group is not None,
            "candidates": candidates,
            "family_aggregation": {
                "active": True,
                "top_family": top_fam,
                "family_confidence": round(top_fam_score * 100, 2),
                "family_rankings": fam_rankings
            }
        }
    else:
        # Standard legacy single-species argmax
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
            "candidates": candidates,
            "family_aggregation": {
                "active": False
            }
        }

