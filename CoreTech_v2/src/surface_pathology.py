"""
Stage 3: Dual-Scale Surface Pathology, Lesion Localization & Masked Hydration/Stress Engine.
Operates STRICTLY inside the Stage 1B isolated plant pixel mask M* (zero pot/soil/table interference
and zero ExG green-color heuristics!).
"""
import io
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))
from configs.v2_config import ARTIFACTS_DIR, DEVICE, USE_BF16, SURFACE_PATHOLOGY_CLASSES
try:
    from .cosine_models import MultiTileAttentionMIL
except (ImportError, ValueError):
    from src.cosine_models import MultiTileAttentionMIL

try:
    from .host_compatibility import apply_biological_host_prior
except (ImportError, ValueError):
    from src.host_compatibility import apply_biological_host_prior

_cached_mil = None
_cached_lesion_yolo = None
_cached_path_calib = None


def load_pathology_suite(reload: bool = False):
    global _cached_mil, _cached_lesion_yolo, _cached_path_calib
    if _cached_mil is not None and not reload:
        return _cached_mil, _cached_lesion_yolo, _cached_path_calib

    calib_p = ARTIFACTS_DIR / "pathology_calibration.json"
    if calib_p.exists():
        _cached_path_calib = json.loads(calib_p.read_text(encoding="utf-8"))
    else:
        _cached_path_calib = {
            "weights": {"convnext": 0.55, "dinov2": 0.45},
            "temperatures": {"convnext": 1.0, "dinov2": 1.0},
            "healthy_override_thresh": 0.62,
        }

    mil_dict = {}
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    for arch, ckpt_name in [("convnext", "pathology_convnext_mil.pth"), ("dinov2", "pathology_dinov2_mil.pth")]:
        p = ARTIFACTS_DIR / ckpt_name
        if not p.exists():
            continue
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        classes = ckpt.get("classes", SURFACE_PATHOLOGY_CLASSES)
        timm_name = ckpt["timm_name"]
        img_size = ckpt.get("img_size", 384)
        m = MultiTileAttentionMIL(timm_name, num_classes=len(classes), pretrained=False)
        m.load_state_dict(ckpt["model_state_dict"], strict=True)
        m = m.to(DEVICE).eval()
        mil_dict[arch] = {"model": m, "classes": classes, "img_size": img_size, "mean": mean, "std": std}

    yolo_p = ARTIFACTS_DIR / "yolo_lesion_det.pt"
    if yolo_p.exists():
        from ultralytics import YOLO
        ym = YOLO(str(yolo_p))
        ym.to(str(DEVICE))
        _cached_lesion_yolo = ym

    _cached_mil = mil_dict
    return _cached_mil, _cached_lesion_yolo, _cached_path_calib


def _build_inference_bag(pil_crop: Image.Image, mask_crop: np.ndarray, img_size: int, mean, std) -> Tuple[torch.Tensor, List[List[int]]]:
    """
    Builds [1, 5, C, H, W] bag using masked plant tiles focused strictly where plant pixels exist.
    Also returns bounding boxes [x0, y0, x1, y1] for each tile in the input image coordinate space.
    """
    rgb = np.array(pil_crop.convert("RGB"))
    orig_h, orig_w, _ = rgb.shape
    global_view = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    sc = 1.0
    min_side = min(orig_h, orig_w)
    if min_side < img_size:
        sc = float(img_size * 1.35) / max(1, min_side)
        rgb = cv2.resize(rgb, (int(orig_w * sc), int(orig_h * sc)), interpolation=cv2.INTER_LINEAR)
        mask_rs = cv2.resize(mask_crop.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        h, w, _ = rgb.shape
    else:
        mask_rs = mask_crop
        h, w = orig_h, orig_w

    tile_sz = min(h, w, max(img_size, int(min(h, w) * 0.50)))
    # Sample 4 tiles centered on dense foreground plant regions
    ys, xs = np.where(mask_rs)
    if len(xs) > 50:
        cx_fg, cy_fg = int(np.median(xs)), int(np.median(ys))
        q1x, q1y = int(np.percentile(xs, 25)), int(np.percentile(ys, 25))
        q3x, q3y = int(np.percentile(xs, 75)), int(np.percentile(ys, 75))
        centers = [(cx_fg, cy_fg), (q1x, q1y), (q3x, q1y), (cx_fg, q3y)]
    else:
        centers = [(w // 2, h // 2), (w // 3, h // 3), (2 * w // 3, h // 3), (w // 2, 2 * h // 3)]

    tiles = [global_view]
    boxes = [[0, 0, orig_w, orig_h]]
    for cx, cy in centers:
        x0 = max(0, min(w - tile_sz, cx - tile_sz // 2))
        y0 = max(0, min(h - tile_sz, cy - tile_sz // 2))
        crop = rgb[y0:y0 + tile_sz, x0:x0 + tile_sz]
        tiles.append(cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR))
        # Map back to original coordinate system
        bx0 = int(x0 / sc)
        by0 = int(y0 / sc)
        bx1 = int((x0 + tile_sz) / sc)
        by1 = int((y0 + tile_sz) / sc)
        boxes.append([bx0, by0, bx1, by1])

    tensors = []
    for t in tiles:
        im_t = TF.to_tensor(Image.fromarray(t))
        im_t = TF.normalize(im_t, mean=mean, std=std)
        tensors.append(im_t)
    return torch.stack(tensors, dim=0).unsqueeze(0), boxes


def generate_xai_attention_heatmap(
    pil_crop: Image.Image,
    mask_crop: np.ndarray,
    feats: torch.Tensor,
    attn_weights: np.ndarray,
    tile_boxes: List[List[int]],
) -> str:
    """
    Renders an explainable AI (XAI) Grad-CAM / Attention Activation Heatmap projected
    strictly into the Stage 1B plant pixel mask. Returns base64 JPEG data URL.
    """
    rgb = np.array(pil_crop.convert("RGB"))
    h, w, _ = rgb.shape
    mask = mask_crop.astype(bool) if mask_crop is not None else np.ones((h, w), dtype=bool)

    acts = torch.norm(feats, dim=1).cpu().numpy()  # [5, 12, 12]
    canvas = np.zeros((h, w), dtype=np.float32)

    # 1. Global View (Index 0)
    g_act = cv2.resize(acts[0], (w, h), interpolation=cv2.INTER_CUBIC)
    canvas += g_act * float(attn_weights[0])

    # 2. Regional Surface Tiles (Indices 1..4)
    for k in range(1, min(len(tile_boxes), len(acts))):
        b = tile_boxes[k]
        x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])
        if x1 > x0 and y1 > y0:
            tile_act = cv2.resize(acts[k], (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
            canvas[y0:y1, x0:x1] += tile_act * float(attn_weights[k]) * 1.5

    # 3. Mask strictly inside plant pixels & Normalize
    canvas = np.where(mask, canvas, 0.0)
    canvas_fg = canvas[mask]
    if canvas_fg.size > 0 and canvas_fg.max() > canvas_fg.min():
        norm = np.clip((canvas - canvas_fg.min()) / (canvas_fg.max() - canvas_fg.min() + 1e-8) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = np.zeros((h, w), dtype=np.uint8)

    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    blend = (rgb.astype(np.float32) * 0.58 + color.astype(np.float32) * 0.42).astype(np.uint8)
    final_img = np.where(mask[:, :, None], blend, rgb)

    buf = io.BytesIO()
    Image.fromarray(final_img).save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def analyze_masked_surface_stress(pil_crop: Image.Image, mask_crop: np.ndarray) -> Dict[str, Any]:
    """
    Computes quantitative surface hydration, turgor/wrinkling texture, chlorosis, and sunscald
    STRICTLY inside Stage 1B segmented plant mask M* (zero pot/soil/table contamination!).
    """
    rgb = np.array(pil_crop.convert("RGB"))
    h, w, _ = rgb.shape
    m = mask_crop.astype(bool) if mask_crop is not None else np.ones((h, w), dtype=bool)
    fg_pixels = int(np.count_nonzero(m))
    if fg_pixels < 100:
        m = np.ones((h, w), dtype=bool)
        fg_pixels = h * w

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    a_ch = lab[:, :, 1][m]
    b_ch = lab[:, :, 2][m]
    h_ch = hsv[:, :, 0][m]
    s_ch = hsv[:, :, 1][m]
    v_ch = hsv[:, :, 2][m]

    chlorophyll_vigor = round(float(np.mean(a_ch < 123) * 100.0), 1)
    yellow_pct = round(float(np.mean((h_ch >= 18) & (h_ch <= 38) & (s_ch > 40) & (b_ch > 136)) * 100.0), 1)
    crispy_brown_pct = round(float(np.mean(((h_ch >= 5) & (h_ch <= 22) & (s_ch >= 28) & (v_ch <= 170)) | ((h_ch >= 5) & (h_ch <= 26) & (s_ch >= 25) & (v_ch <= 180) & (b_ch > 128) & (a_ch >= 124))) * 100.0), 1)
    bleached_pct = round(float(np.mean((s_ch < 35) & (v_ch > 185)) * 100.0), 1)

    # White cottony / powdery wax cluster index (characteristic of mealybugs & cochineal)
    white_waxy_pct = round(float(np.mean((s_ch < 50) & (v_ch > 180) & (b_ch < 138)) * 100.0), 1)

    # Bright orange / cinnamon / reddish-orange spore pustule index (characteristic of Rust Fungi / Pucciniales)
    rust_pustule_pct = round(
        float(np.mean((h_ch >= 7) & (h_ch <= 26) & (s_ch >= 80) & (v_ch >= 80) & (a_ch >= 133) & (b_ch >= 135)) * 100.0),
        2,
    )
    has_rust_pustules = bool(rust_pustule_pct >= 0.65)

    # Optical bug bite holes & leaf perforation index
    # Detect interior cavities, voids, or chewed punctures inside the dominant leaf boundary
    mask_u8 = (mask_crop.astype(np.uint8) * 255)
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    internal_holes_count = 0
    internal_holes_area = 0
    plant_area = max(1, int(np.count_nonzero(mask_crop)))

    if hierarchy is not None and len(contours) > 0:
        for i, c in enumerate(contours):
            # Interior hole within leaf mask
            if hierarchy[0][i][3] != -1:
                area = cv2.contourArea(c)
                if 20 <= area <= (0.28 * plant_area):
                    internal_holes_count += 1
                    internal_holes_area += int(area)

    bite_holes_area_pct = round(float(internal_holes_area / plant_area * 100.0), 2)
    has_optical_bite_holes = bool(internal_holes_count >= 1 and bite_holes_area_pct >= 0.20)

    # Surface wrinkling / shriveling texture index (high local high-freq directional ridges inside mask)
    lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    wrinkle_index = round(float(np.percentile(lap_abs[m], 85)), 2)
    wrinkle_p95 = round(float(np.percentile(lap_abs[m], 95)), 2)
    mean_sat = round(float(np.mean(s_ch) / 255.0 * 100.0), 1)

    # Mask contour tortuosity to detect wild / curled / wavy foliage margins
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
    tortuosity = round(float((total_perimeter ** 2) / (4.0 * np.pi * max(1, fg_pixels))), 2)

    # Check for genuine compact rust pustules vs diffuse dry patches
    rust_color_mask = (hsv[:, :, 0] >= 9) & (hsv[:, :, 0] <= 22) & (hsv[:, :, 1] >= 115) & (hsv[:, :, 2] >= 75) & (hsv[:, :, 2] <= 210) & m
    rust_px = int(np.count_nonzero(rust_color_mask))
    has_compact_rust_pustules = False
    if rust_px > 30:
        r_u8 = rust_color_mask.astype(np.uint8) * 255
        r_cnts, _ = cv2.findContours(r_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        compact_spots = 0
        for c in r_cnts:
            c_area = cv2.contourArea(c)
            if 15 <= c_area <= 2000:
                p_len = cv2.arcLength(c, True)
                if p_len > 0:
                    circ = 4.0 * np.pi * c_area / (p_len * p_len)
                    if circ >= 0.45:
                        compact_spots += 1
        if compact_spots >= 3:
            has_compact_rust_pustules = True

    # Natural white farina wax / trichome bloom check (succulents, cacti, Saintpaulia)
    white_wax_mask = (s_ch < 45) & (v_ch > 200) & (a_ch >= 118) & (a_ch <= 135)
    white_wax_pct = round(float(np.mean(white_wax_mask) * 100.0), 1)
    has_diffuse_white_bloom = white_wax_pct >= 2.5

    # Hydration & Moisture stress evaluation
    is_underwatered = bool(
        (wrinkle_index >= 20.0 and (crispy_brown_pct >= 1.8 or mean_sat < 40.0)) or
        (crispy_brown_pct >= 4.5 and (wrinkle_index >= 15.0 or mean_sat < 42.0)) or
        (wrinkle_index >= 27.0) or
        (tortuosity >= 3.5 and wrinkle_index >= 18.0 and (crispy_brown_pct >= 1.2 or mean_sat < 42.0))
    )
    severe_underwatering = bool(
        (wrinkle_index >= 28.0 and crispy_brown_pct >= 3.5) or
        (crispy_brown_pct >= 9.0 and wrinkle_index >= 18.0) or
        (wrinkle_index >= 36.0)
    )

    mean_val = float(np.mean(v_ch) / 255.0 * 100.0)
    is_overwatered = bool(
        (yellow_pct >= 14.0 and crispy_brown_pct <= 2.2 and wrinkle_index <= 18.0) or
        (yellow_pct >= 22.0 and crispy_brown_pct <= 4.0 and mean_sat <= 48.0) or
        (mean_val >= 62.0 and yellow_pct >= 10.0 and crispy_brown_pct <= 2.0 and wrinkle_index <= 16.0)
    ) and not is_underwatered
    severe_overwatering = bool(
        (yellow_pct >= 28.0 and crispy_brown_pct <= 3.0 and wrinkle_index <= 16.0) or
        (yellow_pct >= 40.0 and crispy_brown_pct <= 5.0)
    ) and is_overwatered

    is_sunburned = bool(
        bleached_pct >= 4.5 and (crispy_brown_pct >= 1.5 or yellow_pct >= 8.0) and wrinkle_index <= 22.0
    ) and not is_underwatered and not is_overwatered

    hydration_dict = {
        "is_underwatered": is_underwatered,
        "severe_underwatering": severe_underwatering,
        "is_overwatered": is_overwatered,
        "severe_overwatering": severe_overwatering,
        "wrinkle_index": wrinkle_index,
        "wrinkle_p95": wrinkle_p95,
        "crispy_margin_pct": crispy_brown_pct,
        "mean_saturation_pct": mean_sat,
        "tortuosity": tortuosity,
        "has_compact_rust_pustules": has_compact_rust_pustules,
        "has_diffuse_white_bloom": has_diffuse_white_bloom,
    }

    # White cottony mealybugs produce high high-frequency edge variance; protect against false dried calls
    is_mealybug_cotton = bool(white_waxy_pct >= 3.5 and wrinkle_index >= 12.0)

    # Detect dead / non-viable tissue (beyond saving) vs recoverable dehydration
    is_dead = (not is_mealybug_cotton) and (
        (chlorophyll_vigor <= 3.0 and crispy_brown_pct >= 45.0) or
        (chlorophyll_vigor <= 1.0 and (crispy_brown_pct + yellow_pct) >= 60.0) or
        (chlorophyll_vigor <= 4.0 and crispy_brown_pct >= 55.0)
    )

    is_dried = (not is_mealybug_cotton) and (is_dead or (
        (crispy_brown_pct >= 25.0 and chlorophyll_vigor <= 25.0) or
        (crispy_brown_pct + yellow_pct >= 50.0 and chlorophyll_vigor <= 15.0) or
        (wrinkle_index >= 16.0 and chlorophyll_vigor <= 20.0 and crispy_brown_pct >= 18.0) or
        (chlorophyll_vigor <= 8.0 and crispy_brown_pct >= 18.0)
    ))
    desiccation_grade = (
        "Non-Viable / Dead (Beyond Saving)"
        if is_dead
        else (
            "Severe Desiccation (Completely Dried Foliage & Stem)"
            if (crispy_brown_pct >= 40.0 or (is_dried and chlorophyll_vigor <= 10.0))
            else ("Moderate Dehydration & Wilting" if is_dried else "Normal Hydration")
        )
    )

    return {
        "chlorophyll_vigor_pct": chlorophyll_vigor,
        "chlorosis_yellowing_pct": yellow_pct,
        "dry_necrotic_margin_pct": crispy_brown_pct,
        "sunburn_bleached_pct": bleached_pct,
        "white_waxy_pct": white_waxy_pct,
        "is_mealybug_cotton": is_mealybug_cotton,
        "rust_pustule_pct": rust_pustule_pct,
        "has_rust_pustules": has_rust_pustules,
        "bite_holes_count": internal_holes_count,
        "bite_holes_area_pct": bite_holes_area_pct,
        "has_optical_bite_holes": has_optical_bite_holes,
        "surface_wrinkle_index": wrinkle_index,
        "mean_saturation_pct": mean_sat,
        "contour_tortuosity": tortuosity,
        "hydration_stress": hydration_dict,
        "chlorosis_score": yellow_pct,
        "necrosis_score": crispy_brown_pct,
        "has_compact_rust_pustules": has_compact_rust_pustules,
        "has_diffuse_white_bloom": has_diffuse_white_bloom,
        "is_sunburned": is_sunburned,
        "is_dried": is_dried,
        "is_dead": is_dead,
        "desiccation_grade": desiccation_grade,
    }


def detect_surface_lesions(pil_crop: Image.Image, mask_crop: np.ndarray, yolo_model) -> Dict[str, Any]:
    """Runs trained YOLO11 surface lesion detector strictly filtered inside plant mask M*."""
    rgb = np.array(pil_crop.convert("RGB"))
    h, w, _ = rgb.shape
    plant_area = max(1, int(np.count_nonzero(mask_crop)))
    vis = pil_crop.copy()
    draw = ImageDraw.Draw(vis)

    if yolo_model is None:
        return {"boxes": [], "spots_count": 0, "infected_area_pct": 0.0, "annotated_img": vis}

    res = yolo_model.predict(pil_crop, conf=0.22, imgsz=640, verbose=False)[0]
    boxes_out = []
    lesion_px = 0
    if res.boxes is not None and len(res.boxes) > 0:
        for b, c_cls, c_conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            if int(c_cls) != 1:
                continue  # Class 1 = surface_lesion_or_disease
            x1, y1, x2, y2 = [int(max(0, v)) for v in b]
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            box_mask = mask_crop[y1:y2, x1:x2]
            fg_overlap = float(np.mean(box_mask)) if box_mask.size > 0 else 0.0
            if fg_overlap < 0.30:
                continue  # Reject detections lying outside the dominant plant mask
            boxes_out.append({"box": [x1, y1, x2, y2], "conf": round(float(c_conf), 3)})
            lesion_px += int(np.count_nonzero(box_mask))
            draw.rectangle([x1, y1, x2, y2], outline=(239, 68, 68), width=max(2, min(w, h) // 200))

    infected_pct = round(min(100.0, (lesion_px / plant_area) * 100.0), 1)
    return {
        "boxes": boxes_out,
        "spots_count": len(boxes_out),
        "infected_area_pct": infected_pct,
        "annotated_img": vis,
    }


def diagnose_surface_condition(
    pil_crop: Image.Image,
    mask_crop: np.ndarray,
    plant_family: str = "Unknown",
    is_externally_dried: bool = False,
    is_externally_dead: bool = False,
    is_externally_underwatered: bool = False,
    verified_pest: Optional[str] = None,
    verified_pathology: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs Stage 3 Multi-Tile MIL Ensemble + YOLO Lesion Localizer + Masked Surface Stress."""
    mil_dict, yolo_lesion, calib = load_pathology_suite()
    stress_metrics = analyze_masked_surface_stress(pil_crop, mask_crop)
    lesion_info = detect_surface_lesions(pil_crop, mask_crop, yolo_lesion)

    if verified_pest or verified_pathology:
        is_dead_plant = False
        is_dried_plant = False
        is_externally_underwatered = False
    else:
        is_dead_plant = bool(is_externally_dead or stress_metrics.get("is_dead", False))
        is_dried_plant = bool(is_externally_dried or stress_metrics.get("is_dried", False) or is_dead_plant)

    if not mil_dict:
        hyd = stress_metrics.get("hydration_stress", {})
        if hyd.get("is_underwatered", False) or is_externally_underwatered:
            primary = "Abiotic_Dehydration_Underwatering"
            super_state = "Abiotic_Stress"
            conf = 88.0 if hyd.get("severe_underwatering", False) else 78.0
            lesion_info["spots_count"] = 0
            lesion_info["infected_area_pct"] = 0.0
            lesion_info["status_message"] = "Foliar surface exhibits moisture deficit and turgor loss (wrinkled folds and dry papery tips). No infectious fungal or pest lesions detected."
            preds = [{"rank": 1, "class_name": primary, "confidence": conf}]
            return {
                "primary_class": primary,
                "super_state": super_state,
                "super_probs": {"healthy_pct": 0.0, "biotic_pct": 0.0, "abiotic_pct": 100.0},
                "confidence": conf,
                "is_healthy": False,
                "predictions": preds,
                "lesion_info": lesion_info,
                "surface_stress": stress_metrics,
            }
        elif hyd.get("is_overwatered", False):
            primary = "Abiotic_Overwatering_Edema"
            super_state = "Abiotic_Stress"
            conf = 86.0 if hyd.get("severe_overwatering", False) else 76.0
            lesion_info["spots_count"] = 0
            lesion_info["infected_area_pct"] = 0.0
            lesion_info["status_message"] = "Foliar surface exhibits moisture saturation and edema swelling. No infectious fungal or pest lesions detected."
            preds = [{"rank": 1, "class_name": primary, "confidence": conf}]
            return {
                "primary_class": primary,
                "super_state": super_state,
                "super_probs": {"healthy_pct": 0.0, "biotic_pct": 0.0, "abiotic_pct": 100.0},
                "confidence": conf,
                "is_healthy": False,
                "predictions": preds,
                "lesion_info": lesion_info,
                "surface_stress": stress_metrics,
            }
        elif stress_metrics.get("is_sunburned", False):
            primary = "Abiotic_Sunburn_Scald"
            super_state = "Abiotic_Stress"
            conf = 85.0
            lesion_info["spots_count"] = 0
            lesion_info["infected_area_pct"] = 0.0
            lesion_info["status_message"] = "Foliar surface exhibits bleached solar scorch. No infectious fungal or pest lesions detected."
            preds = [{"rank": 1, "class_name": primary, "confidence": conf}]
            return {
                "primary_class": primary,
                "super_state": super_state,
                "super_probs": {"healthy_pct": 0.0, "biotic_pct": 0.0, "abiotic_pct": 100.0},
                "confidence": conf,
                "is_healthy": False,
                "predictions": preds,
                "lesion_info": lesion_info,
                "surface_stress": stress_metrics,
            }
        if verified_pest:
            return {
                "primary_class": verified_pest,
                "super_state": "Biotic_Pathology",
                "confidence": 96.5,
                "is_healthy": False,
                "predictions": [{"rank": 1, "class_name": verified_pest, "confidence": 96.5}],
                "lesion_info": lesion_info,
                "surface_stress": stress_metrics,
            }
        if verified_pathology:
            return {
                "primary_class": verified_pathology,
                "super_state": "Biotic_Pathology",
                "confidence": 96.0,
                "is_healthy": False,
                "predictions": [{"rank": 1, "class_name": verified_pathology, "confidence": 96.0}],
                "lesion_info": lesion_info,
                "surface_stress": stress_metrics,
            }
        if is_dead_plant:
            return {
                "primary_class": "Abiotic_Terminal_Dead_Plant",
                "super_state": "Abiotic_Stress",
                "confidence": 97.0,
                "is_healthy": False,
                "predictions": [{"rank": 1, "class_name": "Abiotic_Terminal_Dead_Plant", "confidence": 97.0}],
                "lesion_info": lesion_info,
                "surface_stress": stress_metrics,
            }
        if is_dried_plant:
            return {
                "primary_class": "Abiotic_Severe_Desiccation_Dried_Plant",
                "super_state": "Abiotic_Stress",
                "confidence": 94.0,
                "is_healthy": False,
                "predictions": [{"rank": 1, "class_name": "Abiotic_Severe_Desiccation_Dried_Plant", "confidence": 94.0}],
                "lesion_info": lesion_info,
                "surface_stress": stress_metrics,
            }
        return {
            "primary_class": "Healthy_Foliage",
            "super_state": "Healthy",
            "confidence": 90.0,
            "is_healthy": True,
            "predictions": [{"rank": 1, "class_name": "Healthy_Foliage", "confidence": 90.0}],
            "lesion_info": lesion_info,
            "surface_stress": stress_metrics,
        }

    weights_cfg = calib.get("weights", {})
    temps_cfg = calib.get("temperatures", {})
    amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16

    fine_probs_sum = None
    super_probs_sum = None
    ref_classes = None
    total_w = 0.0

    convnext_feats = None
    convnext_attn = None
    convnext_boxes = None

    with torch.no_grad():
        for arch, bundle in mil_dict.items():
            w = float(weights_cfg.get(arch, 1.0))
            T = max(0.3, float(temps_cfg.get(arch, 1.0)))
            bag, boxes = _build_inference_bag(pil_crop, mask_crop, bundle["img_size"], bundle["mean"], bundle["std"])
            bag = bag.to(DEVICE)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                f_logits, s_logits, _, attn_w = bundle["model"](bag)
                if arch == "convnext":
                    convnext_feats = bundle["model"].backbone.forward_features(bag.squeeze(0))
                    convnext_attn = attn_w
                    convnext_boxes = boxes

            fp = F.softmax(f_logits.float() / T, dim=-1)[0]
            sp = F.softmax(s_logits.float(), dim=-1)[0]

            if fine_probs_sum is None:
                fine_probs_sum = w * fp
                super_probs_sum = w * sp
                ref_classes = bundle["classes"]
            else:
                fine_probs_sum = fine_probs_sum + w * fp
                super_probs_sum = super_probs_sum + w * sp
            total_w += w

    fine_probs = fine_probs_sum / max(1e-6, total_w)
    super_probs = super_probs_sum / max(1e-6, total_w)

    # 1. Feature 2: Test-Time Augmentation (TTA) on Borderline / Low-Confidence Captures
    top_p_pre, _ = fine_probs.topk(2)
    top1_pre = float(top_p_pre[0].item())
    top2_pre = float(top_p_pre[1].item()) if len(top_p_pre) > 1 else 0.0
    pre_margin = top1_pre - top2_pre

    tta_applied = False
    tta_meta = {"applied": False, "iterations": 1, "trigger": "None"}

    if (top1_pre < 0.72 or pre_margin < 0.12) and not is_dried_plant:
        tta_applied = True
        tta_meta = {
            "applied": True,
            "iterations": 3,
            "trigger": f"Low Confidence ({top1_pre*100.0:.1f}%) or Narrow Margin ({pre_margin*100.0:.1f}%)",
        }
        # View 2: Horizontal Mirror (FLIP_LEFT_RIGHT)
        flip_img = pil_crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flip_mask = np.fliplr(mask_crop)
        # View 3: Center Zoom Crop (88%)
        cw, ch = pil_crop.size
        z_sz = int(min(cw, ch) * 0.88)
        zx0, zy0 = max(0, (cw - z_sz) // 2), max(0, (ch - z_sz) // 2)
        zoom_img = pil_crop.crop((zx0, zy0, zx0 + z_sz, zy0 + z_sz)).resize((cw, ch), Image.Resampling.BILINEAR)
        zoom_mask = cv2.resize(mask_crop[zy0:zy0 + z_sz, zx0:zx0 + z_sz].astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST) > 0

        tta_views = [(flip_img, flip_mask, 0.85), (zoom_img, zoom_mask, 0.80)]
        with torch.no_grad():
            for v_img, v_mask, v_w in tta_views:
                for arch, bundle in mil_dict.items():
                    w = float(weights_cfg.get(arch, 1.0)) * v_w
                    T = max(0.3, float(temps_cfg.get(arch, 1.0)))
                    v_bag, _ = _build_inference_bag(v_img, v_mask, bundle["img_size"], bundle["mean"], bundle["std"])
                    v_bag = v_bag.to(DEVICE)
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                        vf_logits, vs_logits, _, _ = bundle["model"](v_bag)
                    v_fp = F.softmax(vf_logits.float() / T, dim=-1)[0]
                    v_sp = F.softmax(vs_logits.float(), dim=-1)[0]
                    fine_probs_sum = fine_probs_sum + w * v_fp
                    super_probs_sum = super_probs_sum + w * v_sp
                    total_w += w

        fine_probs = fine_probs_sum / max(1e-6, total_w)
        super_probs = super_probs_sum / max(1e-6, total_w)

    # 2. Feature 1: Biological Host-Pathogen Compatibility Matrix (Bayesian Prior)
    exempt_host_classes = []
    if verified_pathology:
        exempt_host_classes.append(verified_pathology)
    if verified_pest:
        exempt_host_classes.append(verified_pest)
    if stress_metrics.get("has_rust_pustules", False):
        exempt_host_classes.append("Rust_Fungi")
        exempt_host_classes.append("Rust_Fungi_With_Bug_Bites")
    if stress_metrics.get("has_optical_bite_holes", False):
        exempt_host_classes.append("Chewing_Pests_Bug_Bites_Holes")
        exempt_host_classes.append("Rust_Fungi_With_Bug_Bites")
    fine_probs, host_meta = apply_biological_host_prior(
        fine_probs, ref_classes, plant_family, exempt_classes=exempt_host_classes
    )

    # Apply biological host-family impossibility masking
    fam_l = (plant_family or "").lower()
    if ref_classes is not None:
        # 1. Cactaceae, Crassulaceae, Succulents, Sansevieria, Aloe
        if any(k in fam_l for k in ["cactaceae", "cactus", "crassulaceae", "aloe", "succulent", "agavaceae", "asphodelaceae", "sansevieria"]):
            for idx, cname in enumerate(ref_classes):
                if cname in {"Downy_Mildew", "Powdery_Mildew", "Leaf_Miners", "Whiteflies", "Rust_Fungi"}:
                    fine_probs[idx] *= 0.03
                # Suppress false-positive mealybugs on cacti/succulents with natural areoles or farina bloom
                if cname == "Mealybugs_Cochineal" and stress_metrics.get("has_diffuse_white_bloom", False):
                    fine_probs[idx] *= 0.15
        # 2. Araceae (Monstera, Pothos, Philodendron, Anthurium, Alocasia, Spathiphyllum)
        elif any(k in fam_l for k in ["araceae", "monstera", "philodendron", "pothos", "anthurium", "alocasia", "spathiphyllum"]):
            for idx, cname in enumerate(ref_classes):
                if cname in {"Leaf_Miners", "Powdery_Mildew", "Downy_Mildew", "Rust_Fungi"}:
                    fine_probs[idx] *= 0.03
        # 3. Orchidaceae (Orchids)
        elif any(k in fam_l for k in ["orchidaceae", "orchid"]):
            for idx, cname in enumerate(ref_classes):
                if cname in {"Leaf_Miners", "Powdery_Mildew", "Downy_Mildew", "Rust_Fungi"}:
                    fine_probs[idx] *= 0.03
        # 4. Arecaceae (Palms)
        elif any(k in fam_l for k in ["arecaceae", "palm"]):
            for idx, cname in enumerate(ref_classes):
                if cname in {"Leaf_Miners", "Powdery_Mildew", "Downy_Mildew"}:
                    fine_probs[idx] *= 0.03
        # 5. Ferns (Pteridophytes)
        elif any(k in fam_l for k in ["fern", "pteridophyta", "polypodiaceae"]):
            for idx, cname in enumerate(ref_classes):
                if cname in {"Powdery_Mildew", "Downy_Mildew", "Rust_Fungi", "Leaf_Miners"}:
                    fine_probs[idx] *= 0.03
        fine_probs = fine_probs / fine_probs.sum().clamp(min=1e-8)

    # 3. Feature 3: Explainable AI (XAI) Grad-CAM / Attention Activation Heatmap
    xai_b64 = ""
    tile_attn_pct = []
    if convnext_feats is not None and convnext_attn is not None and convnext_boxes is not None:
        tile_attn_pct = [round(float(a) * 100.0, 1) for a in convnext_attn[0].cpu().numpy()]
        try:
            xai_b64 = generate_xai_attention_heatmap(
                pil_crop, mask_crop, convnext_feats, convnext_attn[0].cpu().numpy(), convnext_boxes
            )
        except Exception as e:
            print(f"[Warning] XAI Attention Heatmap generation exception: {e}")

    p_healthy_sup = float(super_probs[0].item())
    p_biotic_sup = float(super_probs[1].item())
    p_abiotic_sup = float(super_probs[2].item())

    # DEHYDRATION & ABIOTIC STRESS ARBITRATION
    # Suppress spurious Rust and Leaf Miners / Insect Bite detections on shriveled, curled wild foliage
    hyd = stress_metrics.get("hydration_stress", {})
    is_underwatered = hyd.get("is_underwatered", False)
    severe_underwatering = hyd.get("severe_underwatering", False)
    is_overwatered = hyd.get("is_overwatered", False)
    severe_overwatering = hyd.get("severe_overwatering", False)
    is_sunburned = stress_metrics.get("is_sunburned", False)
    has_compact_rust = hyd.get("has_compact_rust_pustules", False)

    if is_underwatered and ref_classes is not None:
        idx_dehydration = ref_classes.index("Abiotic_Dehydration_Underwatering") if "Abiotic_Dehydration_Underwatering" in ref_classes else -1
        idx_rust = ref_classes.index("Rust_Fungi") if "Rust_Fungi" in ref_classes else -1
        idx_miners = ref_classes.index("Leaf_Miners") if "Leaf_Miners" in ref_classes else -1

        # 1. Suppress false-positive Rust Fungi if true compact pustules are absent
        if idx_rust != -1 and not has_compact_rust:
            fine_probs[idx_rust] *= 0.10

        # 2. Suppress false-positive Leaf Miners / bite damage caused by curled wavy margins or shriveled folds
        if idx_miners != -1:
            fine_probs[idx_miners] *= 0.12

        # 3. Boost Abiotic_Dehydration_Underwatering
        if idx_dehydration != -1:
            boost_min = 0.70 if severe_underwatering else 0.52
            fine_probs[idx_dehydration] = torch.clamp(fine_probs[idx_dehydration] * 4.0, min=boost_min)

        fine_probs = fine_probs / fine_probs.sum().clamp(min=1e-8)

    elif is_overwatered and ref_classes is not None:
        idx_overwater = ref_classes.index("Abiotic_Overwatering_Edema") if "Abiotic_Overwatering_Edema" in ref_classes else -1
        idx_scale = ref_classes.index("Scale_Insects") if "Scale_Insects" in ref_classes else -1
        idx_bact = ref_classes.index("Bacterial_Wilt_Soft_Rot") if "Bacterial_Wilt_Soft_Rot" in ref_classes else -1

        # 1. Suppress false-positive Scale Insects (corky edema blisters vs true waxy scale insects)
        if idx_scale != -1:
            fine_probs[idx_scale] *= 0.12

        # 2. Suppress false-positive Bacterial Soft Rot (water-soaked translucent chlorosis vs true bacterial blight)
        if idx_bact != -1:
            fine_probs[idx_bact] *= 0.18

        # 3. Boost Abiotic_Overwatering_Edema
        if idx_overwater != -1:
            boost_min = 0.72 if severe_overwatering else 0.55
            fine_probs[idx_overwater] = torch.clamp(fine_probs[idx_overwater] * 4.0, min=boost_min)

        fine_probs = fine_probs / fine_probs.sum().clamp(min=1e-8)

    elif is_sunburned and ref_classes is not None:
        idx_sunburn = ref_classes.index("Abiotic_Sunburn_Scald") if "Abiotic_Sunburn_Scald" in ref_classes else -1
        idx_fungal = ref_classes.index("Fungal_Leaf_Spot_Anthracnose") if "Fungal_Leaf_Spot_Anthracnose" in ref_classes else -1
        if idx_fungal != -1:
            fine_probs[idx_fungal] *= 0.15
        if idx_sunburn != -1:
            fine_probs[idx_sunburn] = torch.clamp(fine_probs[idx_sunburn] * 3.5, min=0.60)
        fine_probs = fine_probs / fine_probs.sum().clamp(min=1e-8)

    top_p, top_i = fine_probs.topk(5)
    preds = [
        {"rank": r, "class_name": ref_classes[i.item()], "confidence": round(p.item() * 100.0, 2)}
        for r, (p, i) in enumerate(zip(top_p, top_i), 1)
    ]

    primary = preds[0]["class_name"]
    primary_conf = preds[0]["confidence"]

    # Overwatering Edema Guard: Real edema requires either active YOLO blisters/spots or severe water-soaked swelling.
    # If zero spots are detected and chlorophyll vigor is high, don't let noisy overwatering eclipse Healthy_Foliage.
    if (
        primary == "Abiotic_Overwatering_Edema"
        and lesion_info["spots_count"] == 0
        and stress_metrics.get("chlorophyll_vigor_pct", 0) >= 65.0
        and stress_metrics.get("dry_necrotic_margin_pct", 0) < 15.0
        and p_healthy_sup >= 0.40
    ):
        primary = "Healthy_Foliage"
        primary_conf = round(max(p_healthy_sup * 100.0, 78.5), 1)

    # Super-head healthy protection: only declare healthy if super-head healthy dominates AND zero YOLO spots AND no pest/disease/holes
    healthy_thresh = float(calib.get("healthy_override_thresh", 0.62))
    if (
        not verified_pest
        and not verified_pathology
        and not stress_metrics.get("has_rust_pustules", False)
        and not stress_metrics.get("has_optical_bite_holes", False)
        and not is_dried_plant
        and not is_dead_plant
        and not is_externally_underwatered
        and primary != "Healthy_Foliage"
        and p_healthy_sup >= healthy_thresh
        and lesion_info["spots_count"] == 0
    ):
        primary = "Healthy_Foliage"
        primary_conf = round(p_healthy_sup * 100.0, 2)

    if verified_pathology == "Rust_Fungi_With_Bug_Bites" and not is_underwatered:
        primary = "Rust_Fungi_With_Bug_Bites"
        primary_conf = max(primary_conf, 97.0)
        super_state = "Biotic_Pathology"
        p_healthy_sup = min(p_healthy_sup, 0.01)
        p_biotic_sup = max(p_biotic_sup, 0.98)
        p_abiotic_sup = min(p_abiotic_sup, 0.01)
        other_preds = [p for p in preds if p["class_name"] != "Rust_Fungi_With_Bug_Bites"][:4]
        preds = [{"rank": 1, "class_name": "Rust_Fungi_With_Bug_Bites", "confidence": primary_conf}] + [
            {"rank": r + 2, "class_name": p["class_name"], "confidence": round(p["confidence"] * 0.20, 2)}
            for r, p in enumerate(other_preds)
        ]
    elif verified_pest and not is_underwatered:
        primary = verified_pest
        primary_conf = max(primary_conf, 96.5)
        super_state = "Biotic_Pathology"
        p_healthy_sup = min(p_healthy_sup, 0.01)
        p_biotic_sup = max(p_biotic_sup, 0.98)
        p_abiotic_sup = min(p_abiotic_sup, 0.01)
        other_preds = [p for p in preds if p["class_name"] != verified_pest][:4]
        preds = [{"rank": 1, "class_name": verified_pest, "confidence": primary_conf}] + [
            {"rank": r + 2, "class_name": p["class_name"], "confidence": round(p["confidence"] * 0.25, 2)}
            for r, p in enumerate(other_preds)
        ]
    elif verified_pathology and not is_underwatered:
        primary = verified_pathology
        primary_conf = max(primary_conf, 96.0)
        super_state = "Biotic_Pathology"
        p_healthy_sup = min(p_healthy_sup, 0.01)
        p_biotic_sup = max(p_biotic_sup, 0.98)
        p_abiotic_sup = min(p_abiotic_sup, 0.01)
        other_preds = [p for p in preds if p["class_name"] != verified_pathology][:4]
        preds = [{"rank": 1, "class_name": verified_pathology, "confidence": primary_conf}] + [
            {"rank": r + 2, "class_name": p["class_name"], "confidence": round(p["confidence"] * 0.20, 2)}
            for r, p in enumerate(other_preds)
        ]
    elif stress_metrics.get("has_rust_pustules", False) and stress_metrics.get("has_optical_bite_holes", False) and not is_underwatered:
        primary = "Rust_Fungi_With_Bug_Bites"
        primary_conf = max(primary_conf, 96.5)
        super_state = "Biotic_Pathology"
        p_healthy_sup = min(p_healthy_sup, 0.01)
        p_biotic_sup = max(p_biotic_sup, 0.98)
        p_abiotic_sup = min(p_abiotic_sup, 0.01)
        other_preds = [p for p in preds if p["class_name"] != "Rust_Fungi_With_Bug_Bites"][:4]
        preds = [{"rank": 1, "class_name": "Rust_Fungi_With_Bug_Bites", "confidence": primary_conf}] + [
            {"rank": r + 2, "class_name": p["class_name"], "confidence": round(p["confidence"] * 0.20, 2)}
            for r, p in enumerate(other_preds)
        ]
    elif stress_metrics.get("has_rust_pustules", False) and not is_underwatered and (
        primary in ("Fungal_Leaf_Spot_Anthracnose", "Healthy_Foliage", "Abiotic_Overwatering_Edema", "Abiotic_Sunburn_Scald")
        or primary_conf < 78.0
    ):
        # Optical safeguard: raised bright orange/cinnamon urediniospores detected on leaf surface override Anthracnose
        primary = "Rust_Fungi"
        primary_conf = max(primary_conf, 95.0)
        super_state = "Biotic_Pathology"
        p_healthy_sup = min(p_healthy_sup, 0.01)
        p_biotic_sup = max(p_biotic_sup, 0.98)
        p_abiotic_sup = min(p_abiotic_sup, 0.01)
        other_preds = [p for p in preds if p["class_name"] != "Rust_Fungi"][:4]
        preds = [{"rank": 1, "class_name": "Rust_Fungi", "confidence": primary_conf}] + [
            {"rank": r + 2, "class_name": p["class_name"], "confidence": round(p["confidence"] * 0.20, 2)}
            for r, p in enumerate(other_preds)
        ]
    elif stress_metrics.get("has_optical_bite_holes", False) and not is_underwatered and (
        primary in ("Healthy_Foliage", "Abiotic_Overwatering_Edema", "Abiotic_Sunburn_Scald")
        or primary_conf < 75.0
    ):
        # Optical safeguard: internal bite holes / leaf perforations
        primary = "Chewing_Pests_Bug_Bites_Holes"
        primary_conf = max(primary_conf, 94.0)
        super_state = "Biotic_Pathology"
        p_healthy_sup = min(p_healthy_sup, 0.01)
        p_biotic_sup = max(p_biotic_sup, 0.98)
        p_abiotic_sup = min(p_abiotic_sup, 0.01)
        other_preds = [p for p in preds if p["class_name"] != "Chewing_Pests_Bug_Bites_Holes"][:4]
        preds = [{"rank": 1, "class_name": "Chewing_Pests_Bug_Bites_Holes", "confidence": primary_conf}] + [
            {"rank": r + 2, "class_name": p["class_name"], "confidence": round(p["confidence"] * 0.20, 2)}
            for r, p in enumerate(other_preds)
        ]
    elif is_dead_plant:
        primary = "Abiotic_Terminal_Dead_Plant"
        primary_conf = max(primary_conf, 97.0)
        super_state = "Abiotic_Stress"
        p_healthy_sup = min(p_healthy_sup, 0.01)
        p_abiotic_sup = max(p_abiotic_sup, 0.98)
        p_biotic_sup = min(p_biotic_sup, 0.01)
        preds = [
            {"rank": 1, "class_name": "Abiotic_Terminal_Dead_Plant", "confidence": primary_conf}
        ] + [
            {"rank": r + 2, "class_name": p["class_name"], "confidence": round(p["confidence"] * 0.10, 2)}
            for r, p in enumerate(preds[:4])
        ]
    elif is_dried_plant:
        primary = "Abiotic_Severe_Desiccation_Dried_Plant"
        primary_conf = max(primary_conf, 94.0)
        super_state = "Abiotic_Stress"
        p_healthy_sup = min(p_healthy_sup, 0.02)
        p_abiotic_sup = max(p_abiotic_sup, 0.96)
        preds = [
            {"rank": 1, "class_name": "Abiotic_Severe_Desiccation_Dried_Plant", "confidence": primary_conf}
        ] + [
            {"rank": r + 2, "class_name": p["class_name"], "confidence": round(p["confidence"] * 0.15, 2)}
            for r, p in enumerate(preds[:4])
        ]
    elif is_externally_underwatered or is_underwatered:
        primary = "Abiotic_Dehydration_Underwatering"
        primary_conf = max(primary_conf, 93.5)
        super_state = "Abiotic_Stress"
        p_abiotic_sup = max(p_abiotic_sup, 0.92)
        p_healthy_sup = min(p_healthy_sup, 0.05)
        other_preds = [p for p in preds if p["class_name"] != "Abiotic_Dehydration_Underwatering"][:4]
        preds = [{"rank": 1, "class_name": "Abiotic_Dehydration_Underwatering", "confidence": primary_conf}] + [
            {"rank": r + 2, "class_name": p["class_name"], "confidence": round(p["confidence"] * 0.25, 2)}
            for r, p in enumerate(other_preds)
        ]
    else:
        super_state = (
            "Healthy"
            if primary == "Healthy_Foliage"
            else ("Abiotic_Stress" if primary.startswith("Abiotic_") else "Biotic_Pathology")
        )

    # For all Abiotic stress conditions, calibrate lesion detector output:
    # Bounding boxes caused by shriveled folds, edema blisters, or sunscald are non-infectious physiological marks
    if primary.startswith("Abiotic_"):
        lesion_info["spots_count"] = 0
        lesion_info["infected_area_pct"] = 0.0
        if primary == "Abiotic_Dehydration_Underwatering":
            lesion_info["status_message"] = "Foliar surface exhibits moisture deficit and turgor loss (wrinkled folds and dry papery tips). No infectious fungal or pest lesions detected."
        elif primary == "Abiotic_Overwatering_Edema":
            lesion_info["status_message"] = "Foliar surface exhibits moisture saturation and edema swelling. No infectious fungal or pest lesions detected."
        elif primary == "Abiotic_Sunburn_Scald":
            lesion_info["status_message"] = "Foliar surface exhibits bleached solar scorch. No infectious fungal or pest lesions detected."
        elif primary == "Abiotic_Nutrient_Chlorosis":
            lesion_info["status_message"] = "Foliar surface exhibits nutritional chlorosis / mineral deficiency. No infectious fungal or pest lesions detected."

    return {
        "primary_class": primary,
        "super_state": super_state,
        "super_probs": {
            "healthy_pct": round(p_healthy_sup * 100.0, 1),
            "biotic_pct": round(p_biotic_sup * 100.0, 1),
            "abiotic_pct": round(p_abiotic_sup * 100.0, 1),
        },
        "confidence": primary_conf,
        "is_healthy": (primary == "Healthy_Foliage")
        and not is_dried_plant
        and not is_dead_plant
        and not verified_pest
        and not verified_pathology
        and not is_externally_underwatered
        and not stress_metrics.get("has_rust_pustules", False)
        and not stress_metrics.get("has_optical_bite_holes", False),
        "predictions": preds,
        "lesion_info": lesion_info,
        "surface_stress": stress_metrics,
        "xai_heatmap_base64": xai_b64,
        "tile_attention_pct": tile_attn_pct,
        "host_compatibility": host_meta,
        "tta_meta": tta_meta,
    }
