"""
Inference Predictor module for Plant Identification Experimentation.
Integrates Preprocessing Method 1 (botanical leaf isolation).
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = PROJECT_ROOT
CORETECH_ROOT = PROJECT_ROOT.parent

if str(CORETECH_ROOT) not in sys.path:
    sys.path.insert(0, str(CORETECH_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import get_transforms
from src.model import build_model

try:
    from preprocessing.methods import PlantPreprocessors
    _preprocessor = PlantPreprocessors()
except Exception:
    _preprocessor = None

_cached_model = None
_cached_classes = None
_cached_temperature = 1.0
_cached_resolution = config.IMAGE_SIZE

def get_calibrated_temperature() -> float:
    global _cached_temperature
    cal_path = config.ARTIFACTS_DIR / "calibration.json"
    if cal_path.exists():
        try:
            import json
            with open(cal_path, "r", encoding="utf-8") as f:
                _cached_temperature = float(json.load(f).get("calibrated_temperature", 1.0))
        except Exception:
            _cached_temperature = 1.0
    return _cached_temperature

def load_inference_model(reload: bool = False):
    global _cached_model, _cached_classes, _cached_resolution
    if reload:
        _cached_model = None
        _cached_classes = None

    if _cached_model is not None:
        return _cached_model, _cached_classes

    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {config.BEST_MODEL_PATH}")

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    _cached_classes = checkpoint["classes"]
    use_pretrained = checkpoint.get("config", {}).get("pretrained", True)
    backbone = checkpoint.get("backbone", "mobilenet_v3_large")
    _cached_resolution = checkpoint.get("resolution", config.IMAGE_SIZE)

    model = build_model(num_classes=len(_cached_classes), pretrained=use_pretrained, backbone=backbone)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()

    _cached_model = model
    return _cached_model, _cached_classes

def get_tta_crops(img: Image.Image, image_size: int = 384) -> torch.Tensor:
    """
    Generates 5 complementary views of the image for Test-Time Augmentation:
      1. Standard center crop
      2. Tight center crop
      3. Top-Left / Upper flower/leaf
      4. Bottom-Right / Lower stem/base
      5. Horizontally mirrored view
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    to_tensor = transforms.ToTensor()

    resized_base = transforms.Resize(image_size)(img)
    resized_zoomed = transforms.Resize(int(image_size * 1.15))(img)

    crops = [
        transforms.CenterCrop(image_size)(resized_zoomed),
        transforms.CenterCrop(image_size)(resized_base),
        resized_zoomed.crop((0, 0, image_size, image_size)),
        resized_zoomed.crop((resized_zoomed.width - image_size, resized_zoomed.height - image_size, resized_zoomed.width, resized_zoomed.height)),
        transforms.CenterCrop(image_size)(img.transpose(Image.FLIP_LEFT_RIGHT))
    ]

    return torch.stack([norm(to_tensor(c)) for c in crops], dim=0)

def predict_image(image_input, top_k: int = 5, return_preprocessed: bool = False, use_tta: bool = True, use_ensemble: bool = True):
    global _cached_resolution
    swin_ckpt = EXPERIMENT_ROOT / "artifacts" / "best_model_swin.pth"
    if use_ensemble and swin_ckpt.exists():
        from src.ensemble_predict import predict_ensemble
        results = predict_ensemble(image_input, top_k=top_k, use_tta=use_tta)
        if return_preprocessed:
            if isinstance(image_input, (str, Path)):
                img = Image.open(image_input).convert("RGB")
            else:
                img = image_input.convert("RGB")
            return results, img
        return results

    model, classes = load_inference_model()
    _, eval_tf = get_transforms(_cached_resolution)

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError("Expected filepath or PIL Image instance")

    clean_img = img
    if return_preprocessed and _preprocessor is not None:
        try:
            res = _preprocessor.process_for_species_identification(
                np.array(img),
                bg_color=(128, 128, 128),
                padding_ratio=0.12,
                filter_pot_soil=True,
                preserve_leaf_shape=True
            )
            clean_img = Image.fromarray(res["cropped_image"])
        except Exception:
            clean_img = img

    temp = get_calibrated_temperature()

    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            if use_tta:
                # 5-Crop Test-Time Augmentation
                tta_batch = get_tta_crops(img, image_size=_cached_resolution).to(config.DEVICE)
                tta_outputs = model(tta_batch)
                probs = torch.softmax(tta_outputs / temp, dim=1).mean(dim=0)
            else:
                tensor = eval_tf(img).unsqueeze(0).to(config.DEVICE)
                outputs = model(tensor)
                probs = torch.softmax(outputs / temp, dim=1)[0]

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
    parser = argparse.ArgumentParser(description="Experimental Plant Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to input leaf image")
    parser.add_argument("--top_k", type=int, default=5, help="Number of predictions")
    args = parser.parse_args()

    results = predict_image(args.image, top_k=args.top_k)
    print(f"\nPredictions for {args.image}:")
    for r in results:
        print(f"#{r['rank']} {r['species']:<40} {r['confidence']:>6.2f}%")

if __name__ == "__main__":
    main()

