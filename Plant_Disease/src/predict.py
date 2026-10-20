"""
Hierarchical Inference and Lesion Spot Detection Engine.
Performs:
  1. Species identification / group routing (Auto-detect or Manual Override)
  2. Specialized disease diagnosis within that species
  3. Spatial lesion spot detection and bounding box visual highlighting
  4. Curated botanical care advice
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import io
import base64
import numpy as np
import torch
import torch.nn.functional as F

CORETECH_ROOT = Path(__file__).resolve().parent.parent.parent
if str(CORETECH_ROOT) not in sys.path:
    sys.path.insert(0, str(CORETECH_ROOT))

try:
    from Plant_Disease.src import config
    from Plant_Disease.src.dataset import get_transforms
    from Plant_Disease.src.model import build_model
    from Plant_Disease.src.spot_detector import LesionSpotDetector
    from Plant_Disease.src.disease_knowledge import get_disease_info
    from Plant_Disease.src.plant_identifier import identify_plant
except (ImportError, ModuleNotFoundError):
    from src import config
    from src.dataset import get_transforms
    from src.model import build_model
    from src.spot_detector import LesionSpotDetector
    from src.disease_knowledge import get_disease_info
    from src.plant_identifier import identify_plant

try:
    from preprocessing.methods import PlantPreprocessors
    _preprocessor = PlantPreprocessors()
except Exception as e:
    _preprocessor = None

_cached_model = None
_cached_spot_detector = None
_cached_groups = None

def load_inference_system():
    """Loads and caches model and spot detector."""
    global _cached_model, _cached_spot_detector, _cached_groups
    if _cached_model is not None:
        return _cached_model, _cached_spot_detector, _cached_groups

    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {config.BEST_MODEL_PATH}")

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    _cached_groups = checkpoint.get("plant_groups", config.PLANT_GROUPS)

    model = build_model(plant_groups=_cached_groups)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()

    spot_detector = LesionSpotDetector(model, device=config.DEVICE)

    _cached_model = model
    _cached_spot_detector = spot_detector
    return _cached_model, _cached_spot_detector, _cached_groups

def _pil_to_base64(pil_img: Image.Image) -> str:
    """Converts a PIL Image to a base64 data URI string."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_b64}"

def diagnose_image(
    image_input,
    target_species: str = "auto",
    top_k: int = 3
) -> dict:
    """
    Executes two-stage hierarchical diagnosis:
      - Layer 1: Determines plant species using the 47-species botanical model.
      - Layer 2:
          * If plant has training data in disease model: diagnoses condition within
            that species group and detects lesion spots with bounding boxes.
          * If plant has no training data in disease model: returns clear
            "disease identification unavailable" status without false spot detections.
    """
    if isinstance(image_input, (str, bytes, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError("Expected filepath or PIL Image instance")

    # -------------------------------------------------------------------------
    # Layer 1: Plant Identification (47 classes)
    # -------------------------------------------------------------------------
    id_result = identify_plant(pil_img, top_k=5)
    identified_species = id_result["identified_species"]
    id_confidence = id_result["confidence"]
    mapped_group = id_result["mapped_disease_group"]
    top_candidates = id_result["top_candidates"]

    model, spot_detector, plant_groups = load_inference_system()
    species_names = list(plant_groups.keys())

    # Determine whether a manual override or auto-routed group applies
    if target_species != "auto" and target_species in plant_groups:
        active_species = target_species
        detected_mode = f"Manual Override ({target_species.replace('_', ' ')})"
        disease_available = True
    elif mapped_group is not None and mapped_group in plant_groups:
        active_species = mapped_group
        detected_mode = "Auto-Detect (Layer 1 Identification)"
        disease_available = True
    else:
        active_species = None
        detected_mode = "Auto-Detect (Layer 1 Identification)"
        disease_available = False

    # -------------------------------------------------------------------------
    # Branch A: Plant has NO disease training data
    # -------------------------------------------------------------------------
    if not disease_available:
        orig_base64 = _pil_to_base64(pil_img)
        prep_img = id_result.get("preprocessed_image", pil_img)
        prep_base64 = _pil_to_base64(prep_img)
        return {
            "disease_available": False,
            "preprocessed_image_base64": prep_base64,
            "status": "disease_unavailable",
            "status_message": (
                f"Disease identification unavailable for {identified_species}. "
                f"Our pathology training dataset currently supports 5 specific indoor plant varieties: "
                f"Aloe Vera, Cactus, Money Plant, Snake Plant, and Spider Plant."
            ),
            "species": {
                "name": identified_species,
                "confidence": id_confidence,
                "detected_mode": detected_mode,
                "is_supported_in_disease_model": False,
                "top_candidates": top_candidates
            },
            "diagnosis": {
                "plant_name": identified_species,
                "condition_name": "Disease identification unavailable",
                "is_healthy": None,
                "severity": "Unavailable",
                "category": "Unsupported Species",
                "confidence": id_confidence,
                "summary": (
                    f"Botanical identification identified this plant as {identified_species} "
                    f"({id_confidence}% confidence). However, pathology training data for this "
                    f"species is not available in the disease model."
                )
            },
            "spots": {
                "has_spots": False,
                "spots_count": 0,
                "spot_boxes": [],
                "infected_area_pct": 0.0,
                "status_message": f"Lesion spot detection inactive: disease model data not available for {identified_species}.",
                "annotated_image_base64": orig_base64
            },
            "care_guide": {
                "plant_name": identified_species,
                "condition_name": "Disease identification unavailable",
                "summary": (
                    f"Botanical classification successfully identified {identified_species}. "
                    f"Disease diagnosis is currently available for: Aloe Vera, Cactus, Money Plant (Pothos/Pilea), Snake Plant, and Spider Plant."
                ),
                "immediate_actions": [
                    f"Identified Species: {identified_species} ({id_confidence}% match).",
                    "Disease diagnosis unavailable: This plant species is not included in the current disease training dataset.",
                    "If this plant is one of our 5 supported varieties, use the 'Target Plant Mode' selector above to manually choose its group."
                ],
                "treatment": {
                    "organic": ["Ensure plant receives species-appropriate light, water, and humidity."],
                    "chemical": ["Inspect visually for common pests (spider mites, mealybugs, scale)."]
                },
                "prevention": [
                    "Isolate new plants before placing them near other houseplants.",
                    "Use well-draining potting mix and pots with drainage holes to avoid root rot."
                ],
                "symptoms": [
                    f"Botanical Type: {identified_species}"
                ],
                "causes": [
                    "Disease training dataset does not yet include pathology models for this specific plant species."
                ]
            },
            "predictions": [],
            "supported_disease_species": config.SUPPORTED_DISEASE_SPECIES_DISPLAY
        }

    # -------------------------------------------------------------------------
    # Branch B: Plant IS supported -> Execute Specialized Disease Diagnosis
    # -------------------------------------------------------------------------
    # Method 2: Zero-hole lamina lesion preservation, selective CLAHE, and symptomatic crop
    disease_mask = None
    clean_img = pil_img
    if _preprocessor is not None:
        try:
            prep_res = _preprocessor.process_for_disease_identification(
                np.array(pil_img),
                bg_color=(128, 128, 128),
                padding_ratio=0.08,
                filter_pot_soil=True,
                preserve_leaf_shape=True
            )
            clean_img = Image.fromarray(prep_res["cropped_image"])
            disease_mask = prep_res["mask"]
        except Exception as e:
            print(f"Warning: Disease Preprocessing Method 2 fallback: {e}")
            clean_img = pil_img

    _, eval_tf = get_transforms(config.IMAGE_SIZE)
    tensor = eval_tf(clean_img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        _, disease_outputs = model(tensor, target_species=active_species)

    condition_list = plant_groups[active_species]
    disease_logits = disease_outputs[active_species][0]
    disease_probs = F.softmax(disease_logits, dim=0)

    top_probs, top_indices = disease_probs.topk(min(top_k, len(condition_list)))

    predictions = []
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
        cond_name = condition_list[idx.item()]
        full_class_id = f"{active_species}_{cond_name}" if cond_name != "Healthy" else f"{active_species}_Healthy"
        info = get_disease_info(full_class_id)

        predictions.append({
            "rank": rank,
            "condition_idx": idx.item(),
            "class_id": full_class_id,
            "plant_name": info["plant_name"],
            "condition_name": cond_name.replace("_", " "),
            "is_healthy": info["is_healthy"],
            "severity": info["severity"],
            "category": info["category"],
            "confidence": round(prob.item() * 100, 2),
            "care_guide": info
        })

    top_pred = predictions[0]

    # Spatial Lesion Spot Detection & Highlighting using Method 2 solid plant mask
    spot_results = spot_detector.detect_and_highlight(
        pil_image=pil_img,
        tensor=tensor,
        species_name=active_species,
        condition_name=top_pred["condition_name"],
        condition_idx=top_pred["condition_idx"],
        is_healthy=top_pred["is_healthy"],
        plant_mask=disease_mask
    )

    # Display name logic
    if target_species == "auto":
        display_species_name = identified_species
        display_confidence = id_confidence
    else:
        display_species_name = active_species.replace("_", " ")
        display_confidence = 100.0

    return {
        "disease_available": True,
        "preprocessed_image_base64": _pil_to_base64(clean_img),
        "status": "success",
        "status_message": "Diagnosis and spot detection completed successfully.",
        "species": {
            "name": display_species_name,
            "active_group": active_species.replace("_", " "),
            "confidence": display_confidence,
            "detected_mode": detected_mode,
            "is_supported_in_disease_model": True,
            "top_candidates": top_candidates
        },
        "diagnosis": {
            "plant_name": top_pred["plant_name"],
            "condition_name": top_pred["condition_name"],
            "is_healthy": top_pred["is_healthy"],
            "severity": top_pred["severity"],
            "category": top_pred["category"],
            "confidence": top_pred["confidence"]
        },
        "spots": {
            "has_spots": spot_results["has_spots"],
            "spots_count": spot_results["spots_count"],
            "spot_boxes": spot_results["spot_boxes"],
            "infected_area_pct": spot_results["infected_area_pct"],
            "status_message": spot_results["status_message"],
            "annotated_image_base64": spot_results["base64_image"]
        },
        "care_guide": top_pred["care_guide"],
        "predictions": predictions,
        "supported_disease_species": config.SUPPORTED_DISEASE_SPECIES_DISPLAY
    }

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Plant Disease Diagnosis & Spot Localizer")
    parser.add_argument("--image", type=str, required=True, help="Path to input plant photo")
    parser.add_argument("--plant", type=str, default="auto", choices=["auto", "Aloe", "Cactus", "Money_Plant", "Snake_Plant", "Spider_Plant"], help="Target plant species")
    args = parser.parse_args()

    print(f"\nAnalyzing: {args.image} (Plant Mode: {args.plant})...")
    result = diagnose_image(args.image, target_species=args.plant)

    sp = result["species"]
    diag = result["diagnosis"]
    spots = result["spots"]

    print("\n" + "=" * 65)
    print(f"🪴 LAYER 1 PLANT IDENTIFICATION: {sp['name']} ({sp['confidence']}% match)")
    print(f"   Mode: {sp['detected_mode']}")
    print("-" * 65)

    if not result["disease_available"]:
        print("⚠️  DISEASE DIAGNOSIS: UNAVAILABLE")
        print(f"   {result['status_message']}")
        print("\n📌 Currently Supported Disease Varieties:")
        for supp in result["supported_disease_species"]:
            print(f"   • {supp}")
        print("\n🔍 Top Identification Candidates:")
        for cand in sp.get("top_candidates", [])[:3]:
            print(f"   #{cand['rank']} {cand['species']:<40} {cand['confidence']:>6.2f}%")
        print("=" * 65)
        return

    print(f"🔬 PATHOLOGY:  {diag['condition_name']} ({diag['confidence']}% within group)")
    print(f"🎯 HEALTHY?    {'YES (Clean Leaf)' if diag['is_healthy'] else 'NO (Disease Detected)'}")
    print(f"🔍 SPOTS:      {spots['status_message']}")
    if spots["has_spots"]:
        print(f"   Infected Area: {spots['infected_area_pct']}% | Spot Clusters: {spots['spots_count']}")
    print("=" * 65)

    if result.get("care_guide") and result["care_guide"].get("immediate_actions"):
        print(f"\n📋 Treatment Action: {result['care_guide']['immediate_actions'][0]}")

if __name__ == "__main__":
    main()

