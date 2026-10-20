"""
Inference script for predicting plant species on new images.
Provides both a CLI and an importable function for web/API integration.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CORETECH_ROOT = PROJECT_ROOT.parent
if str(CORETECH_ROOT) not in sys.path:
    sys.path.insert(0, str(CORETECH_ROOT))

import numpy as np
import torch
from PIL import Image
from src import config
from src.dataset import get_transforms
from src.model import build_model

try:
    from preprocessing.methods import PlantPreprocessors
    _preprocessor = PlantPreprocessors()
except Exception as e:
    _preprocessor = None

_cached_model = None
_cached_classes = None

def load_inference_model():
    """Loads and caches the model for fast subsequent inference calls."""
    global _cached_model, _cached_classes
    if _cached_model is not None:
        return _cached_model, _cached_classes

    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {config.BEST_MODEL_PATH}")

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    _cached_classes = checkpoint["classes"]

    model = build_model(num_classes=len(_cached_classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()

    _cached_model = model
    return _cached_model, _cached_classes

def predict_image(image_input, top_k: int = 5, return_preprocessed: bool = False):
    """
    Predicts plant species for a given PIL Image or file path.
    Applies Preprocessing Method 1 (Plant Species Identification) to isolate the plant
    from background noise, pots, and soil while preserving leaf margins.
    Returns:
        List of dicts: [{"rank": 1, "species": "...", "confidence": 98.5}, ...]
    """
    model, classes = load_inference_model()
    _, eval_tf = get_transforms(config.IMAGE_SIZE)

    if isinstance(image_input, (str, bytes)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError("Expected filepath or PIL Image instance")

    # Method 1: Plant Species Identification Preprocessing
    clean_img = img
    if _preprocessor is not None:
        try:
            res = _preprocessor.process_for_species_identification(
                np.array(img),
                bg_color=(128, 128, 128),
                padding_ratio=0.12,
                filter_pot_soil=True,
                preserve_leaf_shape=True
            )
            clean_img = Image.fromarray(res["cropped_image"])
        except Exception as e:
            print(f"Warning: Species Preprocessing Method 1 fallback: {e}")
            clean_img = img

    tensor = eval_tf(clean_img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = probs.topk(min(top_k, len(classes)))

    results = []
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
        results.append({
            "rank": rank,
            "species": classes[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    if return_preprocessed:
        return results, clean_img
    return results

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Plant Identification Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to display")
    args = parser.parse_args()

    print(f"\nAnalyzing image: {args.image}...")
    results = predict_image(args.image, top_k=args.top_k)

    print("\n--- Predictions ---")
    for r in results:
        bar = "#" * int(r["confidence"] / 5)
        print(f"#{r['rank']} {r['species']:<40} {r['confidence']:>6.2f}%  {bar}")

if __name__ == "__main__":
    main()

