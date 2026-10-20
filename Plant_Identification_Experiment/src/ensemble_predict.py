"""
FloraNet Heterogeneous Dual-Model Ensemble Predictor
Combines ConvNeXt-Small (50.2M params, 384px) and Swin-Transformer-Tiny (28.3M params, 224px).
Performs temperature-calibrated weighted softmax consensus.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config
from src.model import build_model
from src.predict import get_tta_crops, get_calibrated_temperature

_convnext_model = None
_swin_model = None
_classes = None
_swin_transforms = None

def load_ensemble_models(reload: bool = False):
    global _convnext_model, _swin_model, _classes, _swin_transforms

    if _convnext_model is not None and not reload:
        return _convnext_model, _swin_model, _classes

    convnext_path = EXPERIMENT_ROOT / "artifacts" / "best_model.pth"
    swin_path = EXPERIMENT_ROOT / "artifacts" / "best_model_swin.pth"

    # 1. Load ConvNeXt-Small
    ckpt_c = torch.load(convnext_path, map_location=config.DEVICE, weights_only=False)
    _classes = ckpt_c.get("classes", [])
    _convnext_model = build_model(
        num_classes=len(_classes),
        pretrained=False,
        backbone="convnext_small"
    ).to(config.DEVICE)
    _convnext_model.load_state_dict(ckpt_c["model_state_dict"])
    _convnext_model.eval()

    # 2. Load Swin-Transformer-Tiny (if available)
    if swin_path.exists():
        try:
            ckpt_s = torch.load(swin_path, map_location=config.DEVICE, weights_only=False)
            _swin_model = build_model(
                num_classes=len(_classes),
                pretrained=False,
                backbone="swin_t"
            ).to(config.DEVICE)
            _swin_model.load_state_dict(ckpt_s["model_state_dict"])
            _swin_model.eval()
            swin_res = ckpt_s.get("image_size", 224)
        except Exception as e:
            print(f"[Ensemble] Swin checkpoint load warning: {e}")
            _swin_model = None
            swin_res = 224
    else:
        _swin_model = None
        swin_res = 224

    _swin_transforms = transforms.Compose([
        transforms.Resize((int(swin_res * 1.08), int(swin_res * 1.08))),
        transforms.CenterCrop(swin_res),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return _convnext_model, _swin_model, _classes

def predict_ensemble(
    image_input,
    top_k: int = 5,
    convnext_weight: float = 0.75,
    swin_weight: float = 0.25,
    use_tta: bool = True
) -> List[Dict]:
    """
    Performs dual-model ensemble prediction.
    If Swin model is not yet trained, falls back cleanly to ConvNeXt with TTA.
    """
    c_model, s_model, classes = load_ensemble_models()

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError("Expected filepath or PIL Image instance")

    temp = get_calibrated_temperature()

    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            # ConvNeXt Inference (with 5-crop TTA)
            if use_tta:
                c_crops = get_tta_crops(img, image_size=384).to(config.DEVICE)
                c_logits = c_model(c_crops)
                probs_c = torch.softmax(c_logits / temp, dim=1).mean(dim=0)
            else:
                eval_tf = transforms.Compose([
                    transforms.Resize(416),
                    transforms.CenterCrop(384),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                c_tensor = eval_tf(img).unsqueeze(0).to(config.DEVICE)
                c_logits = c_model(c_tensor)
                probs_c = torch.softmax(c_logits / temp, dim=1)[0]

            if s_model is not None:
                # Swin Transformer Inference
                s_tensor = _swin_transforms(img).unsqueeze(0).to(config.DEVICE)
                s_logits = s_model(s_tensor)
                probs_s = torch.softmax(s_logits / temp, dim=1)[0]

                # Weighted Consensus
                final_probs = (convnext_weight * probs_c) + (swin_weight * probs_s)
            else:
                final_probs = probs_c

    top_probs, top_indices = final_probs.topk(min(top_k, len(classes)))

    results = []
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
        results.append({
            "rank": rank,
            "species": classes[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    return results

