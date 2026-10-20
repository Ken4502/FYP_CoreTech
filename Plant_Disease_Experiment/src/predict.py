"""
PhytoCare Disease Diagnostic & Clinical Treatment Predictor
Features:
  1. Adaptive 5-Crop Test-Time Augmentation (TTA)
  2. Platt Softmax Temperature Calibration (T)
  3. Grad-CAM Lesion Heatmap & Bounding Box Spot Overlays
  4. Curated Botanical First-Aid & Chemical Prescription Engine
"""

import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config
from src.model import build_disease_model
from src.dataset import get_pathology_transforms
from src.spot_detector import LesionSpotDetector

_cached_model = None
_cached_classes = None
_cached_spot_detector = None
_cached_kb = None
_cached_temp = 1.0
_cached_health_gate = None

BIOLOGICAL_EXCLUSIONS = {
    "Cactus": [
        "Universal_Whiteflies",        # Cannot pierce thick cuticle / spines; exclusive to soft-leaved dicots
        "Universal_Leaf_Miners",       # Larvae require laminar leaf blade mesophyll
        "Universal_Downy_Mildew",      # Obligate foliar oomycete on moist dicot foliage
        "Universal_Alternaria_Leaf_Spot",
        "Universal_Powdery_Mildew",    # Obligate leaf parasite; cannot infect succulent xerophytes
        "Universal_Botrytis_Gray_Mold", # Foliar blight on soft flowers/leaves; not natural ribbed cactus areoles
        "Universal_Rust_Fungi"         # Pucciniales cannot infect Cactaceae
    ],
    "Aloe": [
        "Universal_Leaf_Miners",
        "Universal_Downy_Mildew",
        "Universal_Powdery_Mildew",
        "Universal_Rust_Fungi",         # Aloe has species-specific Aloe_Rust (Uromyces aloes)
        "Universal_Botrytis_Gray_Mold"
    ],
    "Snake_Plant": [
        "Universal_Leaf_Miners",
        "Universal_Downy_Mildew",
        "Universal_Powdery_Mildew",
        "Universal_Rust_Fungi"
    ],
    "Spider_Plant": [
        "Universal_Leaf_Miners",
        "Universal_Downy_Mildew",
        "Universal_Powdery_Mildew",
        "Universal_Rust_Fungi"
    ],
    "Money_Plant": [
        "Universal_Leaf_Miners",
        "Universal_Downy_Mildew",
        "Universal_Powdery_Mildew",
        "Universal_Rust_Fungi"
    ]
}

def load_health_gate():
    """Loads and caches the Stage-1 MobileNetV3 Foliar Health Gate."""
    global _cached_health_gate
    if _cached_health_gate is not None:
        return _cached_health_gate

    gate_path = config.ARTIFACTS_DIR / "health_gate.pth"
    if not gate_path.exists():
        return None

    try:
        from torchvision import models
        import torch.nn as nn
        ckpt = torch.load(gate_path, map_location=config.DEVICE, weights_only=False)
        gate_model = models.mobilenet_v3_small(weights=None)
        in_f = gate_model.classifier[3].in_features
        gate_model.classifier[3] = nn.Linear(in_f, 2)
        gate_model.load_state_dict(ckpt["model_state_dict"])
        gate_model = gate_model.to(config.DEVICE)
        gate_model.eval()
        _cached_health_gate = gate_model
        return _cached_health_gate
    except Exception as e:
        print(f"[Warning] Failed to load health gate: {e}")
        return None

def evaluate_health_gate(img: Image.Image):
    """
    Evaluates Stage-1 Foliar Health Gate.
    Returns (p_healthy, p_disease) in [0.0, 100.0].
    """
    gate = load_health_gate()
    if gate is None:
        return 50.0, 50.0

    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = tf(img).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        probs = torch.softmax(gate(tensor), dim=1)[0]
        return round(probs[0].item() * 100, 2), round(probs[1].item() * 100, 2)

def load_diagnostic_system(reload: bool = False):
    global _cached_model, _cached_classes, _cached_spot_detector, _cached_kb, _cached_temp

    if _cached_model is not None and not reload:
        return _cached_model, _cached_classes, _cached_spot_detector, _cached_kb

    model_path = config.BEST_MODEL_CONVNEXT if config.BEST_MODEL_CONVNEXT.exists() else config.BEST_MODEL_ENSEMBLE
    if not model_path.exists():
        print(f"[Warning] Disease checkpoint not found at {model_path}, building untrained model.")
        _cached_model = build_disease_model(num_classes=config.NUM_CLASSES, backbone="convnext_small").to(config.DEVICE)
        _cached_classes = [f"Class_{i}" for i in range(config.NUM_CLASSES)]
        _cached_temp = 1.0
    else:
        ckpt = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
        _cached_classes = ckpt.get("classes", [])
        _cached_temp = ckpt.get("temperature", 0.90)
        _cached_model = build_disease_model(num_classes=len(_cached_classes), backbone="convnext_small", transfer_floranet_weights=False).to(config.DEVICE)
        _cached_model.load_state_dict(ckpt["model_state_dict"])
        _cached_model.eval()

    _cached_spot_detector = LesionSpotDetector(_cached_model, device=config.DEVICE)

    if config.DISEASE_KB_PATH.exists():
        with open(config.DISEASE_KB_PATH, "r", encoding="utf-8") as f:
            _cached_kb = json.load(f)
    else:
        _cached_kb = {}

    return _cached_model, _cached_classes, _cached_spot_detector, _cached_kb

def get_5crop_tta_batch(img: Image.Image, img_size: int = 384) -> torch.Tensor:
    """
    Extracts multi-scale and micro-fovea native-resolution crops to guarantee
    both macroscopic leaf structure and microscopic pests (spider mites, mealybugs, thrips)
    are resolved with maximum optical fidelity.
    """
    from torchvision import transforms
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    w, h = img.size
    min_dim = min(w, h)

    # 1. Standard Center
    c_box = ((w - min_dim)//2, (h - min_dim)//2, (w + min_dim)//2, (h + min_dim)//2)
    crop1 = img.crop(c_box).resize((img_size, img_size), Image.BILINEAR)

    # 2. Tight Zoom (Focal lesion center, 70% dimension)
    zoom_dim = int(min_dim * 0.70)
    z_box = ((w - zoom_dim)//2, (h - zoom_dim)//2, (w + zoom_dim)//2, (h + zoom_dim)//2)
    crop2 = img.crop(z_box).resize((img_size, img_size), Image.BILINEAR)

    # 3. Micro-Fovea Zoom (45% dimension - 2.2x optical magnification for microscopic mites & wax filaments)
    micro_dim = max(int(min_dim * 0.45), 100)
    m_box = ((w - micro_dim)//2, (h - micro_dim)//2, (w + micro_dim)//2, (h + micro_dim)//2)
    crop3 = img.crop(m_box).resize((img_size, img_size), Image.BILINEAR)

    # 4. Upper Canopy / Tip
    crop4 = img.crop((0, 0, w, int(h * 0.70))).resize((img_size, img_size), Image.BILINEAR)

    # 5. Lower Stem / Axil (where mealybugs & scale insects shelter)
    crop5 = img.crop((0, int(h * 0.30), w, h)).resize((img_size, img_size), Image.BILINEAR)

    # 6. Mirror Horizontal Flip
    crop6 = crop1.transpose(Image.FLIP_LEFT_RIGHT)

    return torch.stack([norm(to_tensor(c)) for c in [crop1, crop2, crop3, crop4, crop5, crop6]], dim=0)

def diagnose_plant(image_input, target_group: str = None, top_k: int = 5, use_tta: bool = True, detect_spots: bool = True):
    """
    Performs full clinical pathology diagnosis with optional intra-group botanical gating.
    Returns:
      - predictions: List of {rank, condition_name, confidence, is_healthy, severity}
      - treatment_advice: Curated botanical clinical guide
      - lesion_detection: {highlighted_image, boxes, severity_pct}
    """
    model, classes, spot_detector, kb = load_diagnostic_system()

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError("Expected filepath or PIL Image instance")

    # 1. Run Stage-1 Foliar Health Gate
    p_healthy, p_disease = evaluate_health_gate(img)

    # 2. Determine intra-group candidate indices with Biological Exclusion filtering
    valid_indices = None
    if target_group and target_group.lower() != "auto":
        g_clean = target_group.strip().lower().replace(" ", "_")
        g_cap = target_group.capitalize()
        excluded = [e.lower() for e in BIOLOGICAL_EXCLUSIONS.get(target_group, BIOLOGICAL_EXCLUSIONS.get(g_cap, []))]
        
        if g_clean == "universal":
            valid_indices = [i for i, c in enumerate(classes) if c.lower().startswith("universal_") and c.lower() not in excluded]
        else:
            valid_indices = [
                i for i, c in enumerate(classes)
                if (c.lower().startswith(g_clean) or (g_clean in ["cactus", "opuntia"] and "cactus" in c.lower()) or c.lower().startswith("universal_"))
                and c.lower() not in excluded
            ]
        if not valid_indices:
            valid_indices = None

    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            if use_tta:
                batch = get_5crop_tta_batch(img, img_size=384).to(config.DEVICE)
                outputs = model(batch)
                if valid_indices is not None:
                    outputs = outputs[:, valid_indices]
                    probs = torch.softmax(outputs / _cached_temp, dim=1).mean(dim=0)
                    candidate_classes = [classes[i] for i in valid_indices]
                    candidate_global_indices = valid_indices
                else:
                    probs = torch.softmax(outputs / _cached_temp, dim=1).mean(dim=0)
                    candidate_classes = classes
                    candidate_global_indices = list(range(len(classes)))
            else:
                _, eval_tf = get_pathology_transforms(384)
                tensor = eval_tf(img).unsqueeze(0).to(config.DEVICE)
                outputs = model(tensor)
                if valid_indices is not None:
                    outputs = outputs[:, valid_indices]
                    probs = torch.softmax(outputs / _cached_temp, dim=1)[0]
                    candidate_classes = [classes[i] for i in valid_indices]
                    candidate_global_indices = valid_indices
                else:
                    probs = torch.softmax(outputs / _cached_temp, dim=1)[0]
                    candidate_classes = classes
                    candidate_global_indices = list(range(len(classes)))

    top_probs, local_indices = probs.topk(min(top_k, len(candidate_classes)))

    predictions = []
    for rank, (prob, loc_idx) in enumerate(zip(top_probs, local_indices), start=1):
        cls_name = candidate_classes[loc_idx.item()]
        is_healthy = "healthy" in cls_name.lower()
        predictions.append({
            "rank": rank,
            "class_name": cls_name,
            "confidence": round(prob.item() * 100, 2),
            "is_healthy": is_healthy
        })

    top_cls = predictions[0]["class_name"]
    top_idx = candidate_global_indices[local_indices[0].item()]

    # Treatment Advisory
    treatment_info = kb.get(top_cls, {
        "display_name": top_cls.replace("_", " "),
        "severity": "Unknown",
        "primary_symptoms": "Pathological symptoms detected.",
        "first_aid": "Isolate the plant to prevent spore dispersal.",
        "organic_treatment": "Apply neem oil spray and improve ventilation.",
        "chemical_treatment": "Copper-based fungicide if fungal spots spread.",
        "prevention_rules": ["Water at the base, never over foliage", "Ensure well-draining potting mix"]
    })

    # Spatial Lesion Spot Detection & Highlighting
    lesion_vis_img = None
    lesion_boxes = []
    severity_pct = 0.0

    if detect_spots and not predictions[0]["is_healthy"]:
        try:
            vis_img, boxes, sev_pct = spot_detector.detect_lesions(img, target_class_idx=top_idx)
            lesion_vis_img = vis_img
            lesion_boxes = boxes
            severity_pct = sev_pct
        except Exception as e:
            print(f"[Spot Detection Warning]: {e}")
            lesion_vis_img = img

    return {
        "predictions": predictions,
        "health_gate": {
            "p_healthy": p_healthy,
            "p_disease": p_disease
        },
        "treatment_advice": treatment_info,
        "lesion_highlight": {
            "boxes_count": len(lesion_boxes),
            "boxes": lesion_boxes,
            "severity_pct": severity_pct,
            "annotated_image": lesion_vis_img
        }
    }

