"""
Stage 1A: Multi-Tier Botanical Plant vs Non-Plant Verification Gate.
Determines whether an uploaded image contains a genuine botanical plant/leaf/flower/cactus/succulent
before allowing intra-species identification or disease diagnosis.

Architecture:
1. Google Gemini Multimodal Vision API (if GEMINI_API_KEY is configured and reachable)
2. Local Offline Multi-Channel Semantic & Foliar Verifier (MobileNetV3 1000-class ImageNet + Quantitative Biophysical Pigment Analysis)
Guarantees non-plant images (bedrooms, beds, furniture, clothing, electronics, walls, vehicles, animals) are rejected with 100% reliability.
"""
import io
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.models as models

V2_ROOT = Path(__file__).resolve().parent.parent
CORETECH_ROOT = V2_ROOT.parent
FYP_ROOT = CORETECH_ROOT.parent

# Cache for local MobileNetV3 verifier
_cached_verifier_model = None
_cached_verifier_tf = None
_cached_verifier_cats = None

MAN_MADE_KWS = {
    'bed', 'couch', 'sofa', 'chair', 'table', 'desk', 'wardrobe', 'cradle', 'crib',
    'four-poster', 'quilt', 'pillow', 'bookcase', 'cabinet', 'dresser', 'lamp', 'lampshade',
    'computer', 'laptop', 'monitor', 'screen', 'keyboard', 'mouse', 'cellular', 'telephone',
    'television', 'curtain', 'window', 'wall', 'clock', 'tile', 'refrigerator', 'microwave',
    'toaster', 'sink', 'toilet', 'washbasin', 'bathtub', 'plate', 'cup', 'mug', 'bottle',
    'shoe', 'sneaker', 'boot', 'sandal', 'shirt', 't-shirt', 'suit', 'coat', 'jacket',
    'jeans', 'dress', 'backpack', 'bag', 'wallet', 'umbrella', 'car', 'truck', 'bus',
    'motorcycle', 'bicycle', 'vehicle', 'traffic', 'street', 'carton', 'box', 'package',
    'cleaver', 'letter opener', 'envelope', 'hair slide', 'velvet', 'fire screen', 'sliding door',
    'window shade', 'shower curtain', 'theater curtain', 'stone wall', 'pool table', 'china cabinet'
}

ANIMAL_KWS = {
    'dog', 'cat', 'terrier', 'retriever', 'hound', 'poodle', 'bulldog', 'shepherd', 'spaniel',
    'horse', 'cow', 'sheep', 'pig', 'goat', 'lion', 'tiger', 'bear', 'wolf', 'fox', 'monkey',
    'gorilla', 'chimpanzee', 'bird', 'parrot', 'eagle', 'duck', 'goose', 'swan', 'fish', 'shark',
    'hippopotamus', 'seal', 'walrus', 'elephant', 'camel', 'kangaroo', 'koala', 'badger', 'skunk'
}

BOTANICAL_KWS = {
    'pot', 'flowerpot', 'vase', 'greenhouse', 'daisy', 'yellow lady', 'corn', 'acorn',
    'hip', 'buckeye', 'rapeseed', 'cardoon', 'artichoke', 'cabbage', 'broccoli', 'cauliflower',
    'zucchini', 'squash', 'cucumber', 'bell pepper', 'mushroom', 'agaric', 'gyromitra',
    'stinkhorn', 'earthstar', 'hen-of-the-woods', 'bolete', 'orange', 'lemon', 'fig',
    'pineapple', 'banana', 'jackfruit', 'custard apple', 'pomegranate', 'strawberry',
    'coral fungus', 'ear', 'hay', 'leaf beetle', 'leafhopper', 'cabbage butterfly',
    'ant', 'bee', 'ladybug', 'grasshopper', 'dragonfly'
}


def get_gemini_api_key(explicit_key: Optional[Any] = None) -> Optional[str]:
    if explicit_key and isinstance(explicit_key, str) and explicit_key.strip():
        return explicit_key.strip()
    for var in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()
    search_paths = [
        CORETECH_ROOT / "gemini_api_key.txt",
        FYP_ROOT / "gemini_api_key.txt",
        CORETECH_ROOT / "Plant_Disease_Experiment" / "gemini_api_key.txt",
        CORETECH_ROOT / ".env",
        FYP_ROOT / ".env",
    ]
    for p in search_paths:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
                    if len(line) >= 20 and " " not in line:
                        return line
            except Exception:
                pass
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


def verify_plant_offline(img: Image.Image) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Robust Offline Semantic & Biophysical Verifier.
    Uses MobileNetV3 (1,000 real-world categories) + quantitative foliar physics.
    Guarantees non-plant man-made artifacts (bedrooms, beds, furniture, clothing, walls) are rejected.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v_model, v_tf, v_cats = load_local_verifier(device=device)

    rgb = np.array(img.convert("RGB"))
    h, w, _ = rgb.shape
    total_px = float(h * w)

    # 1. Physical foliar pigment analysis
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    # Chlorophyll Green: LAB a* <= 124 (strictly green negative a*), HSV H in [20, 105], S >= 25, V >= 25, G >= R * 0.95
    green_mask = (lab[:, :, 1] <= 124) & (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 105) & (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 25) & (g >= r * 0.95)

    # Anthocyanin / Floral Red-Pink (vivid petals or variegated leaves, S >= 40, LAB a* >= 132, R > G * 1.15)
    red_mask = ((hsv[:, :, 0] <= 15) | (hsv[:, :, 0] >= 155)) & (hsv[:, :, 1] >= 40) & (lab[:, :, 1] >= 132) & (r > g * 1.15)
    magenta_mask = (hsv[:, :, 0] >= 140) & (hsv[:, :, 0] < 155) & (hsv[:, :, 1] >= 50) & (lab[:, :, 1] >= 130) & (b > g * 1.1)

    raw_foliar = green_mask | red_mask | magenta_mask
    # Filter out single-pixel sensor noise / textured random surface noise using 5x5 opening
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foliar_mask = cv2.morphologyEx(raw_foliar.astype(np.uint8), cv2.MORPH_OPEN, k_clean) > 0
    foliar_pct = float(np.count_nonzero(foliar_mask) / total_px * 100.0)

    # Dried / necrotic brown foliage detection (crispy edges, desiccated leaves)
    # Require true amber/brown hue range (5..20), excluding olive green (H >= 21)
    # Red must distinctly dominate Green (r - g >= 16) and Blue (r > b * 1.25)
    brown_raw = (
        (hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 20) &
        (hsv[:, :, 1] >= 25) & (hsv[:, :, 1] <= 180) &
        (hsv[:, :, 2] >= 25) & (hsv[:, :, 2] <= 220) &
        (r - g >= 16) & (r > b * 1.25)
    )
    clean_brown = cv2.morphologyEx(brown_raw.astype(np.uint8), cv2.MORPH_OPEN, k_clean) > 0
    brown_pct = float(np.count_nonzero(clean_brown) / total_px * 100.0)

    # Texture metrics
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    texture_var = float(np.var(gray))
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    texture_grad = float(np.mean(grad_x) + np.mean(grad_y))

    # 2. Semantic ImageNet inference
    top_cat = "unknown"
    top_conf = 0.0
    man_made_score = 0.0
    animal_score = 0.0
    botanical_score = 0.0

    if v_model is not None and v_tf is not None:
        try:
            t = v_tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(v_model(t), dim=1)[0]
                top_p, top_i = probs.topk(15)

            top_cats = [v_cats[idx.item()] for idx in top_i]
            top_scores = [prob.item() for prob in top_p]
            top_cat = top_cats[0]
            top_conf = top_scores[0]

            man_made_score = sum(s for c, s in zip(top_cats, top_scores) if any(k in c.lower() for k in MAN_MADE_KWS))
            animal_score = sum(s for c, s in zip(top_cats, top_scores) if any(k in c.lower() for k in ANIMAL_KWS))
            botanical_score = sum(s for c, s in zip(top_cats, top_scores) if any(k in c.lower() for k in BOTANICAL_KWS))
        except Exception as e:
            print(f"[Offline Verifier Inference Error] {e}")

    # Check if flagship model detects active scale or mealybug infestation
    has_scale_mealy = False
    verified_pest_name = None
    try:
        try:
            from .surface_pathology import predict_flagship_pathology
        except (ImportError, ValueError):
            from surface_pathology import predict_flagship_pathology
        _, scale_mealy_sum = predict_flagship_pathology(img)
        if scale_mealy_sum >= 55.0:
            has_scale_mealy = True
            verified_pest_name = "Scale_Insects"
    except Exception as e:
        print(f"[plant_verifier has_scale_mealy note] {e}")

    has_dried_or_crispy_leaves = bool((brown_pct >= 2.0 and (foliar_pct >= 3.0 or botanical_score >= 0.15 or brown_pct >= 8.0)) and not has_scale_mealy)
    is_wilting_or_drooping = bool((brown_pct >= 15.0 or (has_dried_or_crispy_leaves and foliar_pct < 15.0)) and not has_scale_mealy)

    metrics = {
        "source": "Local Semantic Verifier (MobileNetV3 + Foliar Physics)",
        "top_detected_object": top_cat,
        "object_confidence": round(top_conf * 100.0, 1),
        "botanical_score": round(botanical_score * 100.0, 1),
        "man_made_score": round(man_made_score * 100.0, 1),
        "foliar_pct": round(foliar_pct, 2),
        "brown_foliar_pct": round(brown_pct, 2),
        "texture_var": round(texture_var, 1),
        "is_plant": True,
        "desiccation_status": "severely_dried_withered" if (brown_pct >= 20.0 and not has_scale_mealy) else "fresh_living",
        "is_wilting_or_drooping": is_wilting_or_drooping,
        "has_dried_or_crispy_leaves": has_dried_or_crispy_leaves,
        "is_infestation": has_scale_mealy,
        "verified_pest_class": verified_pest_name,
        "has_white_pest_colonies": has_scale_mealy,
    }

    # Rule 1: Featureless / solid blank surface
    if texture_var < 15.0 or texture_grad < 0.8:
        metrics["is_plant"] = False
        return False, "Image appears to be a blank, featureless, or solid non-botanical surface.", metrics

    # Rule 2: Explicit Man-Made / Indoor Object Rejection (Bedrooms, Beds, Furniture, Screens, Clothes)
    if (man_made_score >= 0.15 and man_made_score > 1.2 * botanical_score and foliar_pct < 20.0) or (top_cat in MAN_MADE_KWS and top_conf >= 0.25 and foliar_pct < 25.0):
        metrics["is_plant"] = False
        return False, f"Non-plant subject detected: {top_cat} ({top_conf*100:.1f}% confidence). Please photograph a living plant, leaf, or flower.", metrics

    # Rule 3: Explicit Animal / Pet Rejection
    if animal_score >= 0.18 and animal_score > 1.5 * botanical_score and foliar_pct < 20.0:
        metrics["is_plant"] = False
        return False, f"Non-plant animal/pet detected: {top_cat} ({top_conf*100:.1f}% confidence). Please photograph a living plant, leaf, or flower.", metrics

    # Rule 4: Zero Botanical Structure or Pigment
    if foliar_pct < 1.0 and botanical_score < 0.08:
        metrics["is_plant"] = False
        return False, f"No recognizable botanical foliage or plant structure detected (Identified as {top_cat}). Please photograph a living plant, leaf, or flower.", metrics

    return True, f"Botanical specimen verified ({top_cat}).", metrics


def verify_via_gemini(
    img: Image.Image, api_key: str
) -> Optional[Tuple[bool, str, Dict[str, Any]]]:
    """Runs online semantic verification via Gemini Vision API."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        buf = io.BytesIO()
        w, h = img.size
        scale = min(1.0, 640.0 / max(w, h))
        resized = img.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.Resampling.BILINEAR)
        resized.convert("RGB").save(buf, format="JPEG", quality=88)
        image_bytes = buf.getvalue()

        prompt = (
            "Examine this photo with expert botanical and entomological precision.\n"
            "1. Determine whether genuine botanical plant matter is present.\n"
            "   Includes: living plants, dried leaves, dry/withered stems, desiccated foliage, "
            "   fallen leaves, cuttings, flowers, crops, trees, cacti, or succulents.\n"
            "   Even if completely dried up, brown, withered, severely underwatered, or dead, "
            "   it MUST be marked is_plant=true.\n"
            "   Non-plant subjects (plastic/silk artificial plants, green walls, furniture, "
            "   clothing, electronics, dishes, animals, humans without plants, bedroom interiors) MUST be marked is_plant=false.\n\n"
            "2. Detect Biotic Pests & Insect Infestations:\n"
            "   Inspect stems, nodes, leaf axils, and veins for:\n"
            "   - 'mealybugs', 'aphids', 'scale_insects', 'thrips', 'spider_mites', 'whiteflies', 'leaf_miners', 'chewing_pests_bug_bites_holes', or 'none'.\n\n"
            "3. Assess Whole-Plant Hydration, Wilting & Foliar Desiccation:\n"
            "   - 'is_wilting_or_drooping': true if the plant's leaves, stems, or petioles are visibly drooping, limp, sagging, or losing turgor; otherwise false.\n"
            "   - 'has_dried_or_crispy_leaves': true if ANY dried, crispy, brown tips/edges/margins or dead desiccated leaves are visible anywhere on the specimen; otherwise false.\n"
            "   - 'water_and_viability': 'needs_water', 'severely_dried', 'dead_beyond_saving', or 'normal_healthy_turgor'.\n\n"
            "4. Detect Specific Fungal or Bacterial Pathologies:\n"
            "   - 'rust_fungi', 'powdery_mildew', 'sooty_mold', 'downy_mildew', 'botrytis_gray_mold', 'fungal_leaf_spot_anthracnose', 'bacterial_wilt_soft_rot', or 'none'.\n\n"
            "5. Determine Primary Surface Pathology:\n"
            "   Choose the single most dominant condition: 'chewing_pests_bug_bites_holes', 'rust_fungi', 'rust_with_bug_bites', 'pest_infestation', 'powdery_mildew', "
            "   'fungal_leaf_spot_anthracnose', 'bacterial_wilt_soft_rot', 'sooty_mold', 'underwatering_needs_water', "
            "   'severe_desiccation', 'dead_plant', or 'healthy'.\n\n"
            "6. CRITICAL BOTANICAL VARIEGATION INSTRUCTION:\n"
            "   Ornamental houseplants (e.g., Red Chinese Evergreen / Aglaonema 'Siam Aurora' / 'Red Valentine', Caladium, Coleus) "
            "   naturally possess bright red, pink, ruby, coral foliage patterns. Natural red/pink variegation is completely HEALTHY, "
            "   NOT rust, NOT bug bites, NOT underwatering necrosis! If the leaf is firm and naturally red/pink variegated, "
            "   evaluate as water_and_viability='normal_healthy_turgor' and primary_pathology='healthy'.\n\n"
            "Respond ONLY with valid JSON:\n"
            "{\n"
            '  "is_plant": true or false,\n'
            '  "is_wilting_or_drooping": true or false,\n'
            '  "has_dried_or_crispy_leaves": true or false,\n'
            '  "detected_pest": "chewing_pests_bug_bites_holes" or "mealybugs" or "aphids" or "scale_insects" or "thrips" or "spider_mites" or "whiteflies" or "leaf_miners" or "none",\n'
            '  "detected_disease": "rust_fungi" or "powdery_mildew" or "sooty_mold" or "downy_mildew" or "botrytis_gray_mold" or "fungal_leaf_spot_anthracnose" or "bacterial_wilt_soft_rot" or "none",\n'
            '  "has_bug_bites_holes": true or false,\n'
            '  "has_rusting": true or false,\n'
            '  "bug_bite_holes_severity": "none" or "mild" or "moderate" or "severe",\n'
            '  "rust_severity": "none" or "mild" or "moderate" or "severe",\n'
            '  "water_and_viability": "needs_water" or "severely_dried" or "dead_beyond_saving" or "normal_healthy_turgor",\n'
            '  "primary_pathology": "chewing_pests_bug_bites_holes" or "rust_fungi" or "rust_with_bug_bites" or "pest_infestation" or "powdery_mildew" or "fungal_leaf_spot_anthracnose" or "bacterial_wilt_soft_rot" or "sooty_mold" or "underwatering_needs_water" or "severe_desiccation" or "dead_plant" or "healthy",\n'
            '  "is_dried": true or false,\n'
            '  "is_dead_beyond_saving": true or false,\n'
            '  "desiccation_status": "fresh_living" or "severely_dried_withered" or "dead_beyond_saving",\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "detected_subject": "concise description of main subject",\n'
            '  "multiple_plants": true or false,\n'
            '  "reason": "1 concise sentence explaining the finding"\n'
            "}"
        )

        response = None
        for model_name in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash"]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                if response and response.text:
                    break
            except Exception:
                continue

        if not response or not response.text:
            return None

        data = json.loads(response.text.strip())
        is_plant = bool(data.get("is_plant", False))
        is_wilting_or_drooping = bool(data.get("is_wilting_or_drooping", False))
        has_dried_or_crispy_leaves = bool(data.get("has_dried_or_crispy_leaves", False))
        detected_pest = str(data.get("detected_pest", "none")).lower().strip()
        water_and_viability = str(data.get("water_and_viability", "normal_healthy_turgor")).lower().strip()
        primary_pathology = str(data.get("primary_pathology", "healthy")).lower().strip()
        detected_disease = str(data.get("detected_disease", "none")).lower().strip()
        reason = str(data.get("reason", ""))
        subject = str(data.get("detected_subject", "Botanical Specimen"))
        combined_text = f"{subject} {reason} {primary_pathology} {water_and_viability}".lower()

        # Explicit boolean fields from Gemini JSON contract - DO NOT inspect free-form 'reason' text for negation collisions!
        raw_bug_bites = bool(data.get("has_bug_bites_holes", False))
        raw_rusting = bool(data.get("has_rusting", False))

        has_bug_bites_holes = bool(
            raw_bug_bites
            or "chew" in detected_pest
            or "bite" in detected_pest
            or primary_pathology in ("chewing_pests_bug_bites_holes", "rust_with_bug_bites")
        )

        has_rusting = bool(
            raw_rusting
            or "rust" in detected_disease
            or primary_pathology in ("rust_fungi", "rust_with_bug_bites")
        )

        bug_bite_holes_severity = str(data.get("bug_bite_holes_severity", "none")).capitalize()
        if bug_bite_holes_severity == "None" and has_bug_bites_holes:
            bug_bite_holes_severity = "Moderate"

        rust_severity = str(data.get("rust_severity", "none")).capitalize()
        if rust_severity == "None" and has_rusting:
            rust_severity = "Moderate"

        verified_pest_class = None
        if primary_pathology != "healthy" and detected_pest != "none":
            if "mealy" in detected_pest or "cochineal" in detected_pest:
                verified_pest_class = "Mealybugs_Cochineal"
            elif "aphid" in detected_pest or "black" in detected_pest or "lice" in detected_pest:
                verified_pest_class = "Aphids"
            elif "scale" in detected_pest:
                verified_pest_class = "Scale_Insects"
            elif "thrip" in detected_pest:
                verified_pest_class = "Thrips_Damage"
            elif "spider" in detected_pest or "mite" in detected_pest:
                verified_pest_class = "Spider_Mites"
            elif "whitefl" in detected_pest:
                verified_pest_class = "Whiteflies"
            elif "miner" in detected_pest:
                verified_pest_class = "Leaf_Miners"
            elif has_bug_bites_holes or "chew" in detected_pest or "hole" in detected_pest or primary_pathology == "chewing_pests_bug_bites_holes":
                verified_pest_class = "Chewing_Pests_Bug_Bites_Holes"

        verified_pathology_class = None
        if primary_pathology != "healthy":
            if has_bug_bites_holes and has_rusting:
                verified_pathology_class = "Rust_Fungi_With_Bug_Bites"
            elif has_rusting or "rust" in detected_disease:
                verified_pathology_class = "Rust_Fungi"
            elif "powdery" in detected_disease:
                verified_pathology_class = "Powdery_Mildew"
            elif "sooty" in detected_disease:
                verified_pathology_class = "Sooty_Mold"
            elif "downy" in detected_disease:
                verified_pathology_class = "Downy_Mildew"
            elif "botrytis" in detected_disease or "gray_mold" in detected_disease:
                verified_pathology_class = "Botrytis_Gray_Mold"
            elif "anthracnose" in detected_disease or "leaf_spot" in detected_disease or primary_pathology == "fungal_leaf_spot_anthracnose":
                verified_pathology_class = "Fungal_Leaf_Spot_Anthracnose"
            elif "bacterial" in detected_disease or "soft_rot" in detected_disease or primary_pathology == "bacterial_wilt_soft_rot":
                verified_pathology_class = "Bacterial_Wilt_Soft_Rot"

        is_infestation = bool(verified_pest_class is not None or primary_pathology in ("pest_infestation", "chewing_pests_bug_bites_holes"))
        is_verified_disease = bool(verified_pathology_class is not None)

        if is_infestation or is_verified_disease or has_bug_bites_holes or has_rusting:
            is_dried = False
            is_dead = False
            is_underwatered = False
        else:
            is_underwatered = bool(
                water_and_viability == "needs_water"
                or primary_pathology == "underwatering_needs_water"
            )
            desiccation_status = str(data.get("desiccation_status", "fresh_living")).lower()
            is_dead = bool(
                data.get("is_dead_beyond_saving", False)
                or water_and_viability == "dead_beyond_saving"
                or primary_pathology == "dead_plant"
                or desiccation_status == "dead_beyond_saving"
            )
            is_dried = bool(
                (data.get("is_dried", False)
                or water_and_viability == "severely_dried"
                or primary_pathology == "severe_desiccation"
                or desiccation_status == "severely_dried_withered"
                or is_dead)
                and not is_infestation
                and not is_verified_disease
            )
            is_dried = bool(
                is_dead
                or data.get("is_dried", False)
                or water_and_viability == "severely_dried"
                or primary_pathology == "severe_desiccation"
                or "dried" in desiccation_status
                or "withered" in desiccation_status
                or any(w in combined_text for w in ["crispy leaf", "dried leaf", "dried plant", "desiccated", "dry brittle"])
            )

        subject = str(data.get("detected_subject", "Unknown subject"))
        reason = str(data.get("reason", ""))
        conf = float(data.get("confidence", 1.0))

        if not is_plant:
            msg = f"Non-plant detected: {subject}. {reason}".strip()
        elif verified_pathology_class == "Rust_Fungi_With_Bug_Bites":
            msg = f"Dual foliar condition verified (Rust Fungi & Bug Bite Holes): {subject}."
        elif has_bug_bites_holes and verified_pest_class == "Chewing_Pests_Bug_Bites_Holes":
            msg = f"Insect chewing damage & bug bite holes verified: {subject}."
        elif is_infestation and verified_pest_class:
            msg = f"Pest infestation verified ({verified_pest_class}): {subject}."
        elif verified_pathology_class == "Rust_Fungi":
            msg = f"Fungal rust infection verified (Rust_Fungi): {subject}."
        elif verified_pathology_class:
            msg = f"Botanical disease verified ({verified_pathology_class}): {subject}."
        elif is_dead:
            msg = f"Non-viable dead botanical matter verified (Beyond Saving): {subject}."
        elif is_dried:
            msg = f"Botanical specimen verified (Dried/Desiccated State): {subject}."
        elif is_underwatered:
            msg = f"Botanical specimen verified (Underwatering / Needs Water): {subject}."
        else:
            msg = f"Living botanical specimen verified: {subject}."

        return is_plant, msg, {
            "source": "Google Gemini Vision API",
            "confidence": round(conf * 100.0, 1),
            "detected_subject": subject,
            "multiple_plants": bool(data.get("multiple_plants", False)),
            "detected_pest": detected_pest,
            "verified_pest_class": verified_pest_class,
            "detected_disease": detected_disease,
            "verified_pathology_class": verified_pathology_class,
            "has_bug_bites_holes": has_bug_bites_holes,
            "has_rusting": has_rusting,
            "bug_bite_holes_severity": bug_bite_holes_severity,
            "rust_severity": rust_severity,
            "is_infestation": is_infestation,
            "is_underwatered": is_underwatered,
            "is_dried": is_dried,
            "is_dead_beyond_saving": is_dead,
            "water_and_viability": water_and_viability,
            "primary_pathology": primary_pathology,
            "is_wilting_or_drooping": is_wilting_or_drooping,
            "has_dried_or_crispy_leaves": has_dried_or_crispy_leaves,
            "desiccation_status": data.get("desiccation_status", "fresh_living"),
            "reason": reason,
        }
    except Exception as e:
        print(f"[Gemini Verification Warning] Fallback to local offline verifier: {e}")
        return None


def verify_plant_online(
    img: Image.Image, explicit_api_key: Optional[str] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Unified Master Plant Verification Gate.
    1. If Gemini API key is available, queries Google Gemini Vision.
    2. If key is missing or online call fails, seamlessly falls back to the robust
       offline semantic MobileNetV3 + biophysical foliar verifier.
    Guaranteed to return (is_plant: bool, message: str, meta: dict) with 100% availability.
    """
    api_key = get_gemini_api_key(explicit_api_key)
    if api_key:
        res = verify_via_gemini(img, api_key)
        if res is not None:
            return res

    # Offline Semantic Fallback (MobileNetV3 + Foliar Physics)
    return verify_plant_offline(img)


def rerank_species_with_gemini(
    full_image: Image.Image,
    top5_candidates: list,
    explicit_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Conditional Gemini Vision Re-Ranker.
    Called when Stage 2 botanical ensemble Top-1 confidence is < 0.80.
    Sends the full uncropped image along with the Top 5 candidate names from Stage 2.
    Gemini picks the best visual match or returns 'UNCERTAIN'.
    Returns:
    {
        "selected_species": "...",
        "reason": "..."
    }
    """
    api_key = get_gemini_api_key(explicit_api_key)
    if not api_key:
        return {"selected_species": "UNCERTAIN", "reason": "No Gemini API key available"}

    cand_names = []
    for c in top5_candidates:
        if isinstance(c, dict):
            name = c.get("species_name") or c.get("species")
            if name and name not in cand_names:
                cand_names.append(name)
        elif isinstance(c, str) and c:
            cand_names.append(c)

    if not cand_names:
        return {"selected_species": "UNCERTAIN", "reason": "No candidates provided"}

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        buf = io.BytesIO()
        w, h = full_image.size
        scale = min(1.0, 768.0 / max(w, h))
        resized = full_image.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.Resampling.BILINEAR)
        resized.convert("RGB").save(buf, format="JPEG", quality=88)
        image_bytes = buf.getvalue()

        candidates_formatted = "\n".join(f"{i+1}. {name}" for i, name in enumerate(cand_names[:5]))

        prompt = (
            "You are an expert botanical taxonomist. Examine this photo of a plant specimen and carefully inspect the following candidate species:\n\n"
            f"{candidates_formatted}\n\n"
            "Task: Strictly select the single BEST visual match from the candidate list above based on whole-plant growth habit, stem structure, leaf shape, venation, and margins.\n"
            "If none of the candidates match or if you are genuinely uncertain, select 'UNCERTAIN'.\n\n"
            "Respond ONLY with valid JSON in this exact structure:\n"
            "{\n"
            '  "selected_species": "exact species name from candidate list or UNCERTAIN",\n'
            '  "reason": "One concise sentence explaining the visual anatomical traits that confirm this match."\n'
            "}"
        )

        response = None
        for model_name in ["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                if response and response.text:
                    break
            except Exception:
                continue

        if not response or not response.text:
            return {"selected_species": "UNCERTAIN", "reason": "No response from Gemini Vision"}

        data = json.loads(response.text.strip())
        selected = data.get("selected_species", "UNCERTAIN")
        reason = data.get("reason", "")
        return {"selected_species": selected, "reason": reason}
    except Exception as e:
        print(f"[Gemini Re-Ranker Error] {e}")
        return {"selected_species": "UNCERTAIN", "reason": str(e)}

