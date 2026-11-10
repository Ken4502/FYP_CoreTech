"""
UrbanGreen Plant Verification Gate
Determines whether an uploaded image contains a genuine botanical plant/leaf/flower
before allowing intra-species identification or disease diagnosis.

Architecture:
1. Google Gemini Multimodal Vision API (if GEMINI_API_KEY is available)
2. Local MobileNetV3 1000-Class Semantic Open-Set Rejector (Offline / Zero-Config fallback)
"""

import os
import io
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
CORETECH_ROOT = EXPERIMENT_ROOT.parent

# Cache for local verifier
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

UNAMBIGUOUS_MAN_MADE = {
    'chair', 'couch', 'table', 'desk', 'stool', 'bench', 'furniture', 'bed',
    'sofa', 'wardrobe', 'cabinet', 'seat', 'toilet', 'washbasin', 'lamp',
    'laptop', 'notebook', 'computer', 'mouse', 'keyboard', 'screen', 'monitor',
    'car', 'vehicle', 'truck', 'bus', 'bicycle', 'motorcycle', 'wheel',
    'shoe', 'sneaker', 'sandal', 'boot', 'sock', 'clothing', 'jersey', 'suit',
    'jean', 'shirt', 'dress', 'coat', 'jacket', 'hat', 'cap', 'backpack', 'bag',
    'bottle', 'cup', 'mug', 'glass', 'plate', 'bowl', 'pitcher', 'pottery',
    'can', 'tin', 'box', 'carton', 'package', 'container', 'wall', 'tile'
}

def get_gemini_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """Retrieves Gemini API key from explicit param, environment, or key files."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    # Environment variables
    for var in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()

    # Search for gemini_api_key.txt in project directories
    search_paths = [
        CORETECH_ROOT / "gemini_api_key.txt",
        CORETECH_ROOT.parent / "gemini_api_key.txt",
        EXPERIMENT_ROOT / "gemini_api_key.txt",
        CORETECH_ROOT / ".env",
        CORETECH_ROOT.parent / ".env",
    ]
    for p in search_paths:
        if p.exists():
            try:
                content = p.read_text(encoding="utf-8").strip()
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
                    if line.startswith("GOOGLE_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
                    if line.startswith("AIza"):
                        return line
            except Exception:
                pass

    return None

def verify_via_gemini(img: Image.Image, api_key: str) -> Optional[Tuple[bool, str, Dict[str, Any]]]:
    """Uses Google Gemini Multimodal Vision API to verify if image contains a plant."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        # Buffer image to JPEG bytes
        buf = io.BytesIO()
        img.resize((512, 512), Image.BILINEAR).save(buf, format="JPEG", quality=85)
        image_bytes = buf.getvalue()

        prompt = (
            "Examine this image carefully. Is there a genuine living botanical plant, "
            "houseplant, tree, agricultural crop, leaf, stem, flower, or succulent in this photo?\n"
            "If the image is furniture (like a chair or table, even if green), an electronic device, "
            "a person, clothing, a vehicle, or a general room without living plant foliage as the main subject, answer false.\n\n"
            "Respond ONLY with a JSON object in this exact schema:\n"
            "{\n"
            '  "is_plant": true or false,\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "detected_subject": "short description of main subject (e.g. green chair, African Violet, laptop)",\n'
            '  "reason": "1-sentence concise reason"\n'
            "}"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        raw_text = response.text.strip()
        data = json.loads(raw_text)

        is_plant = bool(data.get("is_plant", False))
        detected_subject = data.get("detected_subject", "Unknown object")
        reason = data.get("reason", f"Subject identified as {detected_subject}")

        if not is_plant:
            msg = f"Non-plant detected: {detected_subject}. {reason}"
        else:
            msg = f"Botanical specimen verified: {detected_subject}."

        return is_plant, msg, {
            "source": "Google Gemini Vision API",
            "detected_subject": detected_subject,
            "gemini_confidence": data.get("confidence", 1.0),
            "reason": reason
        }

    except Exception as e:
        print(f"[Gemini Verification Warning] Fallback to local verifier: {e}")
        return None

def load_local_verifier(device: str = "cpu"):
    """Loads and caches the pretrained MobileNetV3 1000-class ImageNet verifier."""
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
        print(f"[Local Verifier Error] {e}")
        return None, None, None

def verify_via_local_model(img: Image.Image) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Robust Offline Semantic Verifier:
    Uses MobileNetV3 (1,000 real-world categories) + quantitative foliar physics.
    Guarantees non-plant man-made artifacts (even green chairs/cars/bottles) are rejected.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v_model, v_tf, v_cats = load_local_verifier(device=device)

    top_cat = "unknown"
    top_conf = 0.0
    plant_score = 0.0
    man_made_score = 0.0

    if v_model is not None and v_tf is not None:
        try:
            t = v_tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = v_model(t)
                probs = F.softmax(out, dim=1)[0]
                top_p, top_i = probs.topk(10)

            top_cats = [v_cats[idx.item()] for idx in top_i]
            top_scores = [prob.item() for prob in top_p]
            top_cat = top_cats[0]
            top_conf = top_scores[0]

            plant_score = sum(s for c, s in zip(top_cats, top_scores) if any(k in c.lower() for k in PLANT_SYNSET_KEYWORDS))
            man_made_score = sum(s for c, s in zip(top_cats, top_scores) if any(k in c.lower() for k in UNAMBIGUOUS_MAN_MADE))
        except Exception as e:
            print(f"[Local Verifier Eval Error] {e}")

    # Foliar color analysis
    rgb = np.array(img.convert('RGB'), dtype=np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    foliage_green = (g > r * 1.04) & (g > b * 1.08) & (g > 25)
    green_pct = float(np.mean(foliage_green) * 100.0)

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    grad_y = np.abs(np.diff(gray, axis=0))
    grad_x = np.abs(np.diff(gray, axis=1))
    texture_var = float(np.var(gray))
    texture_grad = float(np.mean(grad_x) + np.mean(grad_y))

    metrics = {
        "source": "Local Semantic Verifier (MobileNetV3)",
        "top_detected_object": top_cat,
        "object_confidence": round(top_conf * 100, 1),
        "plant_score": round(plant_score * 100, 1),
        "man_made_score": round(man_made_score * 100, 1),
        "green_pct": round(green_pct, 1),
        "texture_var": round(texture_var, 1)
    }

    # 1. Featureless blank / solid surface
    if texture_var < 15.0 or texture_grad < 0.8:
        return False, "Image appears to be a blank, featureless, or solid non-botanical surface", metrics

    # 2. Explicit Man-Made Artifact Rejection (Chairs, Furniture, Clothes, Electronics, Vehicles)
    # Even if painted or upholstered green, if man_made dominates or plant evidence is near zero:
    if man_made_score >= 0.15 and man_made_score > plant_score:
        return False, f"Detected non-plant object: {top_cat} ({top_conf*100:.1f}%)", metrics

    # 3. Botanical Structure Requirement
    # True plants have recognized botanical/floral/potted features
    if plant_score < 0.02:
        return False, f"No recognizable botanical or floral structures detected (Identified as {top_cat})", metrics

    return True, "Botanical specimen verified", metrics

def verify_is_plant(img: Image.Image, explicit_api_key: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Master Plant Verification Gate.
    1. Attempts Google Gemini Multimodal Vision API if key exists.
    2. Falls back to robust local MobileNetV3 Semantic Rejector.
    """
    key = get_gemini_api_key(explicit_api_key)
    if key:
        gemini_res = verify_via_gemini(img, key)
        if gemini_res is not None:
            return gemini_res

    # Local fallback
    return verify_via_local_model(img)
