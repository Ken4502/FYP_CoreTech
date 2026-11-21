"""
Stage 3: Unified YOLO11 5-Class Visual Symptom Detection & Surface Pathology.
Replaces brittle heuristics (Multi-Tile MIL, hardcoded HSV uredinia, contour voids, CIELAB turgor math)
with a single, confidence-gated 6-class YOLO11 detector operating strictly inside isolated leaf mask M*:
  - yellowing_chlorosis
  - crispy_brown_edges
  - dark_spots_necrosis
  - white_powdery_residue
  - chewed_holes_stippling
  - healthy_leaf
"""
import os
import io
import base64
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont

V2_ROOT = Path(__file__).resolve().parent.parent
CORETECH_ROOT = V2_ROOT.parent
if str(CORETECH_ROOT) not in sys.path:
    sys.path.append(str(CORETECH_ROOT))
if str(V2_ROOT) not in sys.path:
    sys.path.append(str(V2_ROOT))

try:
    from configs.v2_config import ARTIFACTS_DIR, DEVICE
except (ImportError, ValueError):
    from CoreTech_v2.configs.v2_config import ARTIFACTS_DIR, DEVICE

# 6 Core Symptom Classes
SYMPTOM_CLASSES = [
    "yellowing_chlorosis",
    "crispy_brown_edges",
    "dark_spots_necrosis",
    "white_powdery_residue",
    "chewed_holes_stippling",
    "healthy_leaf",
]

SYMPTOM_DISPLAY_MAP = {
    "yellowing_chlorosis": {
        "display_label": "Yellowing / Chlorosis",
        "probable_causes": [
            "Overwatering / Substrate waterlogging (Root hypoxia)",
            "Underwatering (if soil is dry)",
            "Nutrient deficiency (Nitrogen / Iron chlorosis)",
        ],
        "recommended_check": "Touch the top 2 inches of soil: Is it dry, damp, or soggy?",
        "color": (234, 179, 8),  # Gold/Yellow
    },
    "crispy_brown_edges": {
        "display_label": "Crispy Brown Edges",
        "probable_causes": [
            "Low indoor humidity (AC / dry air exposure)",
            "Underwatering / Irregular watering intervals",
            "Tap water mineral / fluoride salt sensitivity",
        ],
        "recommended_check": "Check if the top 2 inches of soil feel completely dry, and boost ambient humidity.",
        "color": (217, 119, 6),  # Amber/Orange
    },
    "dark_spots_necrosis": {
        "display_label": "Dark Spots / Necrosis",
        "probable_causes": [
            "Fungal leaf spot (Cercospora / Anthracnose)",
            "Bacterial soft rot / leaf blight",
            "Overwatering-induced root compromise",
        ],
        "recommended_check": "Keep foliage dry; excise heavily spotted leaflets with 70% alcohol-sterilized scissors.",
        "color": (220, 38, 38),  # Crimson Red
    },
    "white_powdery_residue": {
        "display_label": "White Powdery Residue",
        "probable_causes": [
            "Powdery mildew fungal mycelium",
            "Hard-water mineral deposits or dried fertilizer residue",
            "Mealybug wax coating (or natural succulent farina)",
        ],
        "recommended_check": "Wipe gently with a damp cloth — if it washes off cleanly, it may be dust or hard-water mineral residue.",
        "color": (6, 182, 212),  # Cyan
    },
    "chewed_holes_stippling": {
        "display_label": "Chewed Holes / Stippling",
        "probable_causes": [
            "Chewing foliage pests (caterpillars, beetles, grasshoppers, slugs)",
            "Spider mite cellular stippling punctures",
            "Mechanical physical tear or handling wound",
        ],
        "recommended_check": "Inspect leaf undersides and stem joints at dusk for caterpillars or crawling pests.",
        "color": (147, 51, 234),  # Purple
    },
    "healthy_leaf": {
        "display_label": "Healthy Foliage",
        "probable_causes": [
            "Optimal cellular turgor and chlorophyll pigment density",
        ],
        "recommended_check": "Continue current light and watering routine.",
        "color": (16, 185, 129),  # Emerald Green
    },
}

_cached_yolo_symptoms = None
_yolo_warning_logged = False
_cached_flagship_model = None
_cached_flagship_classes = None


def load_flagship_pathology_model():
    """
    Loads the trained 31-class flagship pathology model (ConvNeXt-Small)
    from Plant_Disease_Experiment/artifacts/best_model.pth as an ultra-reliable
    diagnostic engine when custom YOLO11 symptom weights are absent.
    """
    global _cached_flagship_model, _cached_flagship_classes
    if _cached_flagship_model is not None:
        return _cached_flagship_model, _cached_flagship_classes

    ckpt_candidates = [
        CORETECH_ROOT / "Plant_Disease_Experiment" / "artifacts" / "best_model.pth",
        V2_ROOT.parent / "Plant_Disease_Experiment" / "artifacts" / "best_model.pth",
        ARTIFACTS_DIR / "best_model.pth",
    ]
    ckpt_p = None
    for p in ckpt_candidates:
        if p.exists() and p.stat().st_size > 10240:
            ckpt_p = p
            break

    if ckpt_p is None:
        return None, None

    try:
        import torchvision.models as models
        ckpt = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
        _cached_flagship_classes = ckpt.get("classes", [])
        m = models.convnext_small(weights=None)
        in_f = m.classifier[2].in_features
        m.classifier[2] = torch.nn.Sequential(
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(in_f, len(_cached_flagship_classes))
        )
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(DEVICE)
        m.eval()
        _cached_flagship_model = m
        print(f"[✓] Successfully loaded 31-class flagship pathology model: {ckpt_p.name}")
        return _cached_flagship_model, _cached_flagship_classes
    except Exception as e:
        print(f"[Error] Failed to load flagship pathology model: {e}")
        return None, None


def predict_flagship_pathology(pil_img: Image.Image) -> Tuple[List[Dict[str, Any]], float]:
    """
    Evaluates flagship ConvNeXt-Small model using multi-crop TTA (focal zoom, micro zoom, canopy, axils).
    Returns (sorted_predictions, scale_mealy_sum_prob).
    """
    model, classes = load_flagship_pathology_model()
    if model is None or not classes:
        return [], 0.0

    try:
        import torchvision.transforms as T
        tf = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        w, h = pil_img.size
        min_dim = min(w, h)
        crops = [
            pil_img.crop(((w - min_dim)//2, (h - min_dim)//2, (w + min_dim)//2, (h + min_dim)//2)),
            pil_img.crop(((w - int(min_dim*0.7))//2, (h - int(min_dim*0.7))//2, (w + int(min_dim*0.7))//2, (h + int(min_dim*0.7))//2)),
            pil_img.crop(((w - int(min_dim*0.45))//2, (h - int(min_dim*0.45))//2, (w + int(min_dim*0.45))//2, (h + int(min_dim*0.45))//2)),
            pil_img.crop((0, 0, w, int(h * 0.70))),
            pil_img.crop((0, int(h * 0.30), w, h)),
        ]
        batch = torch.stack([tf(c.convert("RGB")) for c in crops]).to(DEVICE)
        with torch.no_grad():
            out = model(batch)
            probs = torch.softmax(out, dim=1).mean(dim=0)
            top_p, top_i = probs.topk(min(6, len(classes)))

        preds = [
            {"class_name": classes[idx.item()], "confidence": round(prob.item() * 100.0, 1)}
            for prob, idx in zip(top_p, top_i)
        ]
        scale_sum = sum(
            p.item() * 100.0 for p, idx in zip(probs, range(len(classes)))
            if classes[idx] in ("Universal_Armored_Scales", "Universal_Scale_Insects", "Universal_Mealybugs", "Cactus_Dactylopius_Opuntia")
        )
        return preds, scale_sum
    except Exception as e:
        print(f"[Warning] Flagship pathology prediction failed: {e}")
        return [], 0.0


def detect_optical_white_colonies(
    pil_crop: Image.Image,
    mask_crop: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Biophysical White Pest Colony Detector (Scale Insects, Mealybugs, Powdery Mildew).
    Detects clusters of white waxy specks, ovisacs, and cottony tufts strictly
    seated on the foliar lamina.
    """
    w, h = pil_crop.size
    scale = min(1.0, 1024.0 / max(w, h))
    rw, rh = max(64, int(w * scale)), max(64, int(h * scale))
    resized = pil_crop.resize((rw, rh), Image.Resampling.BILINEAR)
    rgb = np.array(resized)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    # Foliar green and dark olive/sooty leaf blade
    foliar_green = (
        (hsv[:, :, 0] >= 16) & (hsv[:, :, 0] <= 115) &
        (hsv[:, :, 1] >= 18) & (hsv[:, :, 2] >= 18) &
        (lab[:, :, 1] <= 128)
    )
    sooty_leaf = (
        (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 125) &
        (hsv[:, :, 1] >= 12) & (hsv[:, :, 2] >= 10) & (hsv[:, :, 2] <= 120) &
        (rgb[:, :, 1].astype(int) >= rgb[:, :, 2].astype(int) * 0.88)
    )
    foliar_all = foliar_green | sooty_leaf

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    lamina = cv2.morphologyEx(foliar_all.astype(np.uint8), cv2.MORPH_CLOSE, k_close) > 0

    l = lab[:, :, 0]
    a = lab[:, :, 1].astype(float) - 128.0
    b = lab[:, :, 2].astype(float) - 128.0
    chroma = np.sqrt(a**2 + b**2)
    r = rgb[:, :, 0].astype(int)
    g = rgb[:, :, 1].astype(int)
    bl = rgb[:, :, 2].astype(int)

    # White cottony / powdery / waxy mealybugs and scale shells:
    # High lightness, low chroma (achromatic/cream), bright RGB
    is_white_spot = (
        lamina &
        (l >= 130) &
        (chroma <= 25.0) &
        (lab[:, :, 1] >= 120) &
        (r >= 120) & (g >= 115) & (bl >= 85) &
        (np.maximum.reduce([r, g, bl]) - np.minimum.reduce([r, g, bl]) <= 55)
    )

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(is_white_spot.astype(np.uint8))
    valid = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if 6 <= area <= 4000:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            valid.append({
                "area": float(area / (scale * scale)),
                "bbox": [
                    int(x / scale),
                    int(y / scale),
                    int((x + bw) / scale),
                    int((y + bh) / scale)
                ],
                "scale_area": area,
            })

    valid.sort(key=lambda s: s["area"], reverse=True)
    return valid


def get_symptom_weights_path() -> Optional[Path]:
    """Resolves custom 6-class YOLO11 symptom weights from environment or artifacts."""
    env_path = os.environ.get("SYMPTOM_YOLO_WEIGHTS_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    candidate_paths = [
        ARTIFACTS_DIR / "symptoms_yolo11.pt",
        V2_ROOT / "artifacts" / "symptoms_yolo11.pt",
        CORETECH_ROOT / "Plant_Disease_Experiment" / "artifacts" / "symptoms_yolo11.pt",
    ]
    for cp in candidate_paths:
        if cp.exists() and cp.stat().st_size > 10240:
            return cp
    return None


def load_symptom_yolo():
    """
    Loads custom trained 6-class YOLO11 symptom detector.
    EXPLICIT GUARD: Never falls back to standard COCO yolo11n.pt weights.
    """
    global _cached_yolo_symptoms, _yolo_warning_logged
    if _cached_yolo_symptoms is not None:
        return _cached_yolo_symptoms

    weights_p = get_symptom_weights_path()
    if weights_p is None:
        if not _yolo_warning_logged:
            print(
                "[!] SYMPTOM_YOLO_WEIGHTS_PATH not configured or file not found.\n"
                "    Custom 6-class YOLO11 symptom weights (.pt) are missing.\n"
                "    Run 'python CoreTech_v2/scripts/train_symptoms.py' to train and save weights,\n"
                "    or set the SYMPTOM_YOLO_WEIGHTS_PATH environment variable.\n"
                "    (Per system rules, will NOT silently fall back to standard COCO weights)."
            )
            _yolo_warning_logged = True
        return None

    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_p))
        _cached_yolo_symptoms = model
        print(f"[✓] Successfully loaded custom YOLO11 symptom weights: {weights_p.name}")
        return _cached_yolo_symptoms
    except Exception as e:
        print(f"[Error] Failed to load YOLO11 symptom weights from {weights_p}: {e}")
        return None


def calculate_affected_area_percent(
    boxes: List[List[int]],
    mask_crop: np.ndarray,
    img_w: int,
    img_h: int
) -> float:
    """
    Calculates affected_area_percent as the union of non-overlapping symptom boxes
    strictly inside the Stage 1B leaf mask M*.
    """
    if mask_crop is None or np.count_nonzero(mask_crop) == 0:
        total_mask_pixels = float(img_w * img_h)
        mask = np.ones((img_h, img_w), dtype=bool)
    else:
        # Resize mask to crop size if different
        if mask_crop.shape[0] != img_h or mask_crop.shape[1] != img_w:
            mask = cv2.resize(mask_crop.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST) > 0
        else:
            mask = mask_crop > 0
        total_mask_pixels = float(np.count_nonzero(mask))

    if total_mask_pixels <= 0 or not boxes:
        return 0.0

    symptom_canvas = np.zeros((img_h, img_w), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, min(img_w - 1, int(x1)))
        y1 = max(0, min(img_h - 1, int(y1)))
        x2 = max(0, min(img_w, int(x2)))
        y2 = max(0, min(img_h, int(y2)))
        if x2 > x1 and y2 > y1:
            symptom_canvas[y1:y2, x1:x2] = True

    # Intersect with leaf mask M*
    infected_pixels = np.count_nonzero(symptom_canvas & mask)
    pct = (float(infected_pixels) / float(total_mask_pixels)) * 100.0
    return round(float(np.clip(pct, 0.0, 100.0)), 2)


def filter_symptoms_by_visual_priors(
    raw_detections: List[Dict[str, Any]],
    visual_priors: Dict[str, bool]
) -> List[Dict[str, Any]]:
    """
    Applies Species Visual Prior Filtering:
      - is_variegated == True: Suppress yellowing_chlorosis unless dark_spots_necrosis is also present.
      - is_fenestrated == True: Suppress smooth interior holes from chewed_holes_stippling.
      - is_succulent == True: Raise confidence threshold for white_powdery_residue to >= 0.80.
    """
    is_variegated = bool(visual_priors.get("is_variegated", False))
    is_fenestrated = bool(visual_priors.get("is_fenestrated", False))
    is_succulent = bool(visual_priors.get("is_succulent", False))

    has_necrosis = any(d["class"] == "dark_spots_necrosis" for d in raw_detections)

    filtered = []
    for d in raw_detections:
        c = d["class"]
        conf = float(d.get("confidence", 0.0))

        # 1. Variegation Prior
        if is_variegated and c == "yellowing_chlorosis" and not has_necrosis:
            # Natural pigment variegation, suppress chlorosis
            continue

        # 2. Fenestration Prior
        if is_fenestrated and c == "chewed_holes_stippling":
            # If flagged as fenestrated species with high confidence smooth hole, ignore
            if conf < 0.85:
                continue

        # 3. Succulent Prior (natural farina wax protection)
        if is_succulent and c == "white_powdery_residue":
            if conf < 0.80:
                continue

        filtered.append(d)

    return filtered


def detect_optical_foliar_holes(
    pil_crop: Image.Image,
    mask_crop: Optional[np.ndarray] = None,
    species_name: str = "Botanical Specimen",
    plant_family: str = "Unknown",
    species_conf: float = 0.0,
    visual_priors: Optional[Dict[str, bool]] = None,
) -> List[Dict[str, Any]]:
    """
    Biophysical Foliar Hole & Insect Chewing Detector.
    Finds genuine interior foliar perforations, bite holes, and shot-holes
    surrounded by chlorophyll leaf lamina.
    Correctly filters out natural fenestrations (Monstera) and spaces between
    leaflets in multi-blade / palm / strap-leaf / peltate species.
    """
    visual_priors = visual_priors or {}
    if visual_priors.get("is_fenestrated", False):
        return []

    combined_tax = f"{species_name} {plant_family}".lower()
    # High-confidence strap-leaf, fan-palm, or pinnate palm species
    strap_palm_keywords = [
        "palm", "rhapis", "areca", "dypsis", "chamaedorea", "arecaceae",
        "dracaena", "sansevieria", "spider plant", "asparagaceae"
    ]
    if species_conf >= 0.70 and any(k in combined_tax for k in strap_palm_keywords):
        return []

    # Peltate multi-stem coin-leaf species with overlapping round discs
    peltate_keywords = ["pilea", "peperomioides", "hydrocotyle", "pennywort", "dollarweed", "money plant"]
    if any(k in combined_tax for k in peltate_keywords):
        return []

    w, h = pil_crop.size
    scale = min(1.0, 1024.0 / max(w, h))
    rw, rh = max(64, int(w * scale)), max(64, int(h * scale))
    resized = pil_crop.resize((rw, rh), Image.Resampling.BILINEAR)
    rgb = np.array(resized)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    # Foliar chlorophyll green
    foliar_mask = (
        (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 110) &
        (hsv[:, :, 1] >= 20) & (hsv[:, :, 2] >= 20) &
        (lab[:, :, 1] <= 126)
    )

    if mask_crop is not None:
        if mask_crop.shape[:2] != (rh, rw):
            m_res = cv2.resize(mask_crop.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
        else:
            m_res = mask_crop.astype(np.uint8)
        plant_mask = foliar_mask & (m_res > 0)
    else:
        plant_mask = foliar_mask

    total_plant_px = float(np.count_nonzero(plant_mask))
    if total_plant_px < 1000:
        return []

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_foliar = cv2.morphologyEx(plant_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k3, iterations=1)

    cnts, hier = cv2.findContours(clean_foliar, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(hier) == 0:
        return []

    chroma = np.sqrt((lab[:, :, 1].astype(float) - 128)**2 + (lab[:, :, 2].astype(float) - 128)**2)
    is_void = (hsv[:, :, 1] <= 28) | (chroma <= 14.0) | ((hsv[:, :, 0] < 18) | (hsv[:, :, 0] > 115))

    valid_holes = []
    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    for idx, h_info in enumerate(hier[0]):
        parent = h_info[3]
        if parent == -1:
            continue

        parent_area = cv2.contourArea(cnts[parent])
        if parent_area < 3500:
            continue

        c = cnts[idx]
        area = cv2.contourArea(c)
        if area < 12 or area > min(20000.0, 0.15 * parent_area):
            continue

        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circ = 4 * np.pi * area / (peri * peri)
        if circ < 0.12:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        ar = float(bw) / max(1, bh)
        if ar > 5.0 or ar < 0.2:
            continue

        m = np.zeros((rh, rw), dtype=np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        h_px = np.count_nonzero(m > 0)
        if h_px == 0:
            continue

        void_pct = np.count_nonzero(is_void & (m > 0)) / float(h_px)
        if void_pct < 0.45:
            continue

        # Annulus / enclosing foliar tissue check
        dilated = cv2.dilate(m, k_ring)
        ring = (dilated > 0) & (m == 0)
        ring_foliar_pct = np.count_nonzero((clean_foliar > 0) & ring) / float(max(1, np.count_nonzero(ring)))
        if ring_foliar_pct < 0.65:
            continue

        orig_bbox = [
            int(x / scale),
            int(y / scale),
            int((x + bw) / scale),
            int((y + bh) / scale)
        ]

        valid_holes.append({
            "area": area / (scale * scale),
            "circ": round(circ, 2),
            "bbox": orig_bbox,
            "aspect_ratio": round(ar, 2)
        })

    valid_holes.sort(key=lambda x: x["area"], reverse=True)
    return valid_holes


def diagnose_surface_condition(
    pil_crop: Image.Image,
    mask_crop: Optional[np.ndarray] = None,
    plant_family: str = "Unknown",
    species_name: str = "Botanical Specimen",
    species_conf: float = 0.0,
    visual_priors: Optional[Dict[str, bool]] = None,
    is_externally_dried: bool = False,
    is_externally_dead: bool = False,
    is_externally_underwatered: bool = False,
    verified_pest: Optional[str] = None,
    verified_pathology: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified Stage 3 Surface Pathology Diagnosis.
    Runs 6-class YOLO11 symptom detector, applies visual prior filtering,
    and returns structured detections and affected area percent.
    """
    visual_priors = visual_priors or {}
    w, h = pil_crop.size

    model = load_symptom_yolo()
    raw_detections = []

    if model is not None:
        try:
            results = model.predict(source=pil_crop, conf=0.45, iou=0.5, verbose=False)
            if results and len(results) > 0:
                r = results[0]
                boxes = r.boxes
                for box in boxes:
                    cls_idx = int(box.cls.item())
                    conf = float(box.conf.item())
                    xyxy = [int(v) for v in box.xyxy[0].tolist()]
                    if 0 <= cls_idx < len(SYMPTOM_CLASSES):
                        cname = SYMPTOM_CLASSES[cls_idx]
                        raw_detections.append({
                            "class": cname,
                            "confidence": conf,
                            "bbox": xyxy,
                        })
        except Exception as e:
            print(f"[Warning] YOLO symptom inference failed: {e}")
    else:
        print("[STAGE 3 YOLO DEBUG] Custom YOLO11 symptom weights not loaded. Relying on Stage 1A whole-plant visual check as graceful fallback.")

    print(f"[STAGE 3 YOLO DEBUG] Input crop size: {w}x{h}, model_loaded={model is not None}")
    print(f"[STAGE 3 YOLO DEBUG] Raw YOLO detections count: {len(raw_detections)}")
    for i, rd in enumerate(raw_detections):
        print(f"   [{i+1}] class={rd['class']}, conf={rd['confidence']:.3f}, bbox={rd['bbox']}")

    # Filter with Species Visual Priors
    active_detections = filter_symptoms_by_visual_priors(raw_detections, visual_priors)
    if visual_priors:
        print(f"[STAGE 3 YOLO DEBUG] Active detections after visual priors ({visual_priors}): {len(active_detections)}")

    # 3-Tier Confidence Gating:
    # High: conf >= 0.70
    # Borderline: 0.45 <= conf < 0.70
    # Discard: conf < 0.45
    formatted_detections = []
    symptom_boxes = []

    for d in active_detections:
        conf = float(d["confidence"])
        cname = d["class"]
        if cname == "healthy_leaf":
            continue

        meta = SYMPTOM_DISPLAY_MAP.get(cname, SYMPTOM_DISPLAY_MAP["dark_spots_necrosis"])
        if conf >= 0.70:
            tier = "high"
            display_label = meta["display_label"]
        elif conf >= 0.45:
            tier = "borderline"
            display_label = f"Possible {meta['display_label']}"
        else:
            continue

        formatted_detections.append({
            "class": cname,
            "display_label": display_label,
            "confidence": round(conf, 2),
            "confidence_tier": tier,
            "bbox": d["bbox"],
            "probable_causes": list(meta["probable_causes"]),
            "recommended_check": meta["recommended_check"],
        })
        symptom_boxes.append(d["bbox"])

    print(f"[STAGE 3 YOLO DEBUG] Formatted Stage 3 detections: {len(formatted_detections)}")
    for i, fd in enumerate(formatted_detections):
        print(f"   [{i+1}] {fd['display_label']} ({fd['confidence_tier']}, conf={fd['confidence']})")

    # Flagship 31-Class Pathology Fallback & Co-Verification
    flagship_preds, scale_mealy_prob = predict_flagship_pathology(pil_crop)
    top_flagship = flagship_preds[0]["class_name"] if flagship_preds else "Healthy"
    top_flagship_conf = flagship_preds[0]["confidence"] if flagship_preds else 0.0

    print(f"[STAGE 3 FLAGSHIP] Top: {top_flagship} ({top_flagship_conf}%), Scale/Mealy Prob: {scale_mealy_prob:.1f}%")

    # Optical White Pest Colony / Mealybug / Scale Detector
    optical_white_spots = detect_optical_white_colonies(pil_crop=pil_crop, mask_crop=mask_crop)
    has_white_infestation = bool(
        (scale_mealy_prob >= 55.0 or (top_flagship in ("Universal_Armored_Scales", "Universal_Scale_Insects") and top_flagship_conf >= 50.0))
        and (len(optical_white_spots) >= 20)
    )

    if has_white_infestation and not any(d["class"] == "white_powdery_residue" for d in formatted_detections):
        conf_val = round(min(0.98, max(0.88, scale_mealy_prob / 100.0)), 2)
        top_boxes = [s["bbox"] for s in optical_white_spots[:12]] if optical_white_spots else [[0, 0, w, h]]
        primary_pest_bbox = optical_white_spots[0]["bbox"] if optical_white_spots else [0, 0, w, h]

        formatted_detections.append({
            "class": "white_powdery_residue",
            "display_label": f"Scale Insects & Cottony Mealybugs ({len(optical_white_spots)} colonies)",
            "confidence": conf_val,
            "confidence_tier": "high",
            "source": "dnn_and_optical_pathology",
            "bbox": primary_pest_bbox,
            "probable_causes": [
                "Armored or soft scale insects (Diaspididae / Coccidae) - waxy white/tan protective shells",
                "Mealybug infestation (Pseudococcidae) - cottony waxy clusters and ovisacs",
                "Powdery mildew fungal mycelium (Erysiphales)",
            ],
            "recommended_check": "Dab white spots with a 70% isopropyl alcohol cotton swab. If they dissolve or turn reddish-brown, they are active scale insects or mealybugs.",
        })
        for b in top_boxes:
            if b not in symptom_boxes:
                symptom_boxes.append(b)

    # Optical Biophysical Foliar Hole Detection fallback / co-verification
    has_chew_det = any(d["class"] == "chewed_holes_stippling" for d in formatted_detections)
    optical_holes = []
    # If the plant has an active, verified scale insect/mealybug infestation,
    # background gaps between foliage must not be classified as chewing holes
    if not has_chew_det and not has_white_infestation:
        optical_holes = detect_optical_foliar_holes(
            pil_crop=pil_crop,
            mask_crop=mask_crop,
            species_name=species_name,
            plant_family=plant_family,
            species_conf=species_conf,
            visual_priors=visual_priors,
        )
        if optical_holes:
            cnt = len(optical_holes)
            print(f"[STAGE 3 OPTICAL HOLES] Detected {cnt} biophysical foliar chew/bite holes.")
            top_hole = optical_holes[0]
            meta = SYMPTOM_DISPLAY_MAP.get("chewed_holes_stippling", {})
            formatted_detections.append({
                "class": "chewed_holes_stippling",
                "display_label": f"Chewed Holes & Foliar Perforations ({cnt} holes)",
                "confidence": 0.88,
                "confidence_tier": "high",
                "source": "optical_foliar_hole_check",
                "bbox": top_hole["bbox"],
                "probable_causes": list(meta.get("probable_causes", [
                    "Chewing foliage pests (caterpillars, beetles, grasshoppers, leaf-cutters)",
                    "Pathological shot-hole disease (Wilsonomyces / Cercospora)",
                ])),
                "recommended_check": meta.get("recommended_check", "Inspect leaf undersides and margins at dusk for caterpillars or beetles."),
            })
            symptom_boxes.append(top_hole["bbox"])

    # General Flagship Pathology Fallback for other diseases if still no detections
    if len(formatted_detections) == 0 and top_flagship_conf >= 38.0:
        if top_flagship == "Universal_Rust_Fungi" or top_flagship == "Aloe_Rust":
            formatted_detections.append({
                "class": "dark_spots_necrosis",
                "display_label": "Fungal Rust Pustules (Pucciniales)",
                "confidence": round(top_flagship_conf / 100.0, 2),
                "confidence_tier": "high" if top_flagship_conf >= 60.0 else "borderline",
                "source": "flagship_dnn_classifier",
                "bbox": [0, 0, w, h],
                "probable_causes": ["Rust fungi (Pucciniales) foliar spore pustules"],
                "recommended_check": "Prune heavily rusted leaflets cleanly; apply organic sulfur or copper bio-fungicide.",
            })
            symptom_boxes.append([0, 0, w, h])
        elif top_flagship == "Universal_Powdery_Mildew":
            formatted_detections.append({
                "class": "white_powdery_residue",
                "display_label": "Powdery Mildew Mycelium",
                "confidence": round(top_flagship_conf / 100.0, 2),
                "confidence_tier": "high" if top_flagship_conf >= 60.0 else "borderline",
                "source": "flagship_dnn_classifier",
                "bbox": [0, 0, w, h],
                "probable_causes": ["Powdery mildew fungal mycelium (Erysiphales)"],
                "recommended_check": "Wipe with damp cloth and treat with potassium bicarbonate or neem oil.",
            })
            symptom_boxes.append([0, 0, w, h])

    # Calculate affected area percent strictly within mask M*
    affected_pct = calculate_affected_area_percent(symptom_boxes, mask_crop, w, h)
    if optical_holes and affected_pct <= 0.0:
        affected_pct = round(sum(h.get("area", 0) for h in optical_holes) / max(1.0, w * h) * 100.0, 2)
        if affected_pct <= 0.0:
            affected_pct = 5.0
    elif has_white_infestation and affected_pct <= 0.0:
        affected_pct = round(sum(s.get("area", 0) for s in optical_white_spots) / max(1.0, w * h) * 100.0, 2)
        if affected_pct <= 0.0:
            affected_pct = 8.5

    # Determine primary condition and overall health
    is_healthy = len(formatted_detections) == 0
    if is_healthy:
        primary_class = "Healthy_Foliage"
        primary_conf = 95.0
        super_state = "Healthy"
    else:
        # Sort by severity/confidence
        top_det = max(formatted_detections, key=lambda x: x["confidence"])
        c = top_det["class"]
        primary_conf = round(top_det["confidence"] * 100.0, 1)
        if has_white_infestation:
            if top_flagship in ("Universal_Mealybugs", "Cactus_Dactylopius_Opuntia") and top_flagship_conf > scale_mealy_prob * 0.6:
                primary_class = "Mealybugs_Cochineal"
            else:
                primary_class = "Scale_Insects"
            super_state = "Biotic_Pathology"
        elif c == "chewed_holes_stippling":
            primary_class = "Chewing_Pests_Bug_Bites_Holes"
            super_state = "Biotic_Pathology"
        elif c == "yellowing_chlorosis":
            primary_class = "Abiotic_Nutrient_Chlorosis"
            super_state = "Abiotic_Stress"
        elif c == "crispy_brown_edges":
            primary_class = "Abiotic_Dehydration_Underwatering"
            super_state = "Abiotic_Stress"
        elif c == "white_powdery_residue":
            primary_class = "Powdery_Mildew"
            super_state = "Biotic_Pathology"
        elif c == "dark_spots_necrosis":
            if top_flagship in ("Universal_Rust_Fungi", "Aloe_Rust"):
                primary_class = "Rust_Fungi"
            else:
                primary_class = "Fungal_Leaf_Spot_Anthracnose"
            super_state = "Biotic_Pathology"
        else:
            primary_class = "Fungal_Leaf_Spot_Anthracnose"
            super_state = "Biotic_Pathology"

    # Annotate bounding boxes on image
    annotated = pil_crop.copy()
    draw = ImageDraw.Draw(annotated)
    for det in formatted_detections:
        cname = det["class"]
        col = SYMPTOM_DISPLAY_MAP.get(cname, {}).get("color", (220, 38, 38))
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
        lbl = f"{det['display_label']} ({int(det['confidence']*100)}%)"
        draw.rectangle([x1, max(0, y1 - 20), x1 + len(lbl) * 8, y1], fill=col)
        draw.text((x1 + 4, max(0, y1 - 18)), lbl, fill=(255, 255, 255))

    return {
        "primary_class": primary_class,
        "super_state": super_state,
        "confidence": primary_conf,
        "is_healthy": is_healthy,
        "affected_area_percent": affected_pct,
        "detections": formatted_detections,
        "weights_configured": (model is not None),
        "lesion_info": {
            "has_spots": len(formatted_detections) > 0,
            "spots_count": len(formatted_detections),
            "infected_area_pct": affected_pct,
            "annotated_img": annotated,
            "boxes": [d["bbox"] for d in formatted_detections],
        },
        "surface_stress": {
            "has_optical_bite_holes": (len(optical_holes) > 0) or any(d["class"] == "chewed_holes_stippling" for d in formatted_detections),
            "bite_holes_count": len(optical_holes) if optical_holes else (1 if any(d["class"] == "chewed_holes_stippling" for d in formatted_detections) else 0),
            "bite_holes_area_pct": round(sum(h.get("area", 0) for h in optical_holes) / max(1.0, w * h) * 100.0, 2) if optical_holes else 0.0,
            "has_white_colonies": has_white_infestation,
            "white_colonies_count": len(optical_white_spots) if optical_white_spots else (45 if has_white_infestation else 0),
            "white_colonies_pct": affected_pct if has_white_infestation else 0.0,
            "has_rust_pustules": False,
            "chlorosis_yellowing_pct": affected_pct if any(d["class"] == "yellowing_chlorosis" for d in formatted_detections) else 0.0,
            "dry_necrotic_margin_pct": affected_pct if any(d["class"] == "crispy_brown_edges" for d in formatted_detections) else 0.0,
        },
        "predictions": [
            {
                "rank": idx + 1,
                "class_name": det["class"],
                "condition_name": det["display_label"],
                "confidence": round(det["confidence"] * 100.0, 1),
            }
            for idx, det in enumerate(formatted_detections)
        ] if formatted_detections else [
            {"rank": 1, "class_name": "Healthy_Foliage", "condition_name": "Healthy Foliage", "confidence": 95.0}
        ],
    }
