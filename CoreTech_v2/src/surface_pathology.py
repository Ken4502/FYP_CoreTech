"""
Stage 3: Dual-Scale Surface Pathology, Lesion Localization & Masked Hydration/Stress Engine.
Operates STRICTLY inside the Stage 1B isolated plant pixel mask M* (zero pot/soil/table interference
and zero ExG green-color heuristics!).
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
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


def _build_inference_bag(pil_crop: Image.Image, mask_crop: np.ndarray, img_size: int, mean, std) -> torch.Tensor:
    """Builds [1, 5, C, H, W] bag using masked plant tiles focused strictly where plant pixels exist."""
    rgb = np.array(pil_crop.convert("RGB"))
    h, w, _ = rgb.shape
    global_view = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    min_side = min(h, w)
    if min_side < img_size:
        sc = float(img_size * 1.35) / max(1, min_side)
        rgb = cv2.resize(rgb, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_LINEAR)
        mask_rs = cv2.resize(mask_crop.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        h, w, _ = rgb.shape
    else:
        mask_rs = mask_crop

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
    for cx, cy in centers:
        x0 = max(0, min(w - tile_sz, cx - tile_sz // 2))
        y0 = max(0, min(h - tile_sz, cy - tile_sz // 2))
        crop = rgb[y0:y0 + tile_sz, x0:x0 + tile_sz]
        tiles.append(cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR))

    tensors = []
    for t in tiles:
        im_t = TF.to_tensor(Image.fromarray(t))
        im_t = TF.normalize(im_t, mean=mean, std=std)
        tensors.append(im_t)
    return torch.stack(tensors, dim=0).unsqueeze(0)


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

    # Hydration stress evaluation
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

    hydration_dict = {
        "is_underwatered": is_underwatered,
        "severe_underwatering": severe_underwatering,
        "severe_overwatering": False,
        "wrinkle_index": wrinkle_index,
        "wrinkle_p95": wrinkle_p95,
        "crispy_margin_pct": crispy_brown_pct,
        "mean_saturation_pct": mean_sat,
        "tortuosity": tortuosity,
        "has_compact_rust_pustules": has_compact_rust_pustules,
    }

    return {
        "chlorophyll_vigor_pct": chlorophyll_vigor,
        "chlorosis_yellowing_pct": yellow_pct,
        "dry_necrotic_margin_pct": crispy_brown_pct,
        "sunburn_bleached_pct": bleached_pct,
        "surface_wrinkle_index": wrinkle_index,
        "mean_saturation_pct": mean_sat,
        "contour_tortuosity": tortuosity,
        "hydration_stress": hydration_dict,
        "chlorosis_score": yellow_pct,
        "necrosis_score": crispy_brown_pct,
        "has_compact_rust_pustules": has_compact_rust_pustules,
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
    pil_crop: Image.Image, mask_crop: np.ndarray, plant_family: str = "Unknown"
) -> Dict[str, Any]:
    """Runs Stage 3 Multi-Tile MIL Ensemble + YOLO Lesion Localizer + Masked Surface Stress."""
    mil_dict, yolo_lesion, calib = load_pathology_suite()
    stress_metrics = analyze_masked_surface_stress(pil_crop, mask_crop)
    lesion_info = detect_surface_lesions(pil_crop, mask_crop, yolo_lesion)

    if not mil_dict:
        hyd = stress_metrics.get("hydration_stress", {})
        if hyd.get("is_underwatered", False):
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

    with torch.no_grad():
        for arch, bundle in mil_dict.items():
            w = float(weights_cfg.get(arch, 1.0))
            T = max(0.3, float(temps_cfg.get(arch, 1.0)))
            bag = _build_inference_bag(pil_crop, mask_crop, bundle["img_size"], bundle["mean"], bundle["std"]).to(DEVICE)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                f_logits, s_logits, _, _ = bundle["model"](bag)
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

    # Apply biological host-family impossibility masking (e.g. foliar downy mildew cannot infect ribbed desert cacti)
    fam_l = plant_family.lower()
    if "cactaceae" in fam_l or "cactus" in fam_l:
        for idx, cname in enumerate(ref_classes):
            if cname in {"Downy_Mildew", "Powdery_Mildew", "Leaf_Miners", "Whiteflies", "Rust_Fungi"}:
                fine_probs[idx] *= 0.05
        fine_probs = fine_probs / fine_probs.sum().clamp(min=1e-8)

    p_healthy_sup = float(super_probs[0].item())
    p_biotic_sup = float(super_probs[1].item())
    p_abiotic_sup = float(super_probs[2].item())

    # DEHYDRATION & ABIOTIC STRESS ARBITRATION
    # Suppress spurious Rust and Leaf Miners / Insect Bite detections on shriveled, curled wild foliage
    hyd = stress_metrics.get("hydration_stress", {})
    is_underwatered = hyd.get("is_underwatered", False)
    severe_underwatering = hyd.get("severe_underwatering", False)
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

    top_p, top_i = fine_probs.topk(5)
    preds = [
        {"rank": r, "class_name": ref_classes[i.item()], "confidence": round(p.item() * 100.0, 2)}
        for r, (p, i) in enumerate(zip(top_p, top_i), 1)
    ]

    primary = preds[0]["class_name"]
    primary_conf = preds[0]["confidence"]

    # Super-head healthy protection: only declare healthy if super-head healthy dominates AND zero YOLO spots
    healthy_thresh = float(calib.get("healthy_override_thresh", 0.62))
    if primary != "Healthy_Foliage" and p_healthy_sup >= healthy_thresh and lesion_info["spots_count"] == 0 and primary_conf < 65.0:
        primary = "Healthy_Foliage"
        primary_conf = round(p_healthy_sup * 100.0, 2)

    # If diagnosed as Abiotic_Dehydration_Underwatering, calibrate lesion detector output:
    # Bounding boxes caused by shriveled folds / curled margins are non-infectious mechanical turgor collapse
    if primary == "Abiotic_Dehydration_Underwatering":
        lesion_info["spots_count"] = 0
        lesion_info["infected_area_pct"] = 0.0
        lesion_info["status_message"] = "Foliar surface exhibits moisture deficit and turgor loss (wrinkled folds and dry papery tips). No infectious fungal or pest lesions detected."

    super_state = "Healthy" if primary == "Healthy_Foliage" else ("Abiotic_Stress" if primary.startswith("Abiotic_") else "Biotic_Pathology")

    return {
        "primary_class": primary,
        "super_state": super_state,
        "super_probs": {
            "healthy_pct": round(p_healthy_sup * 100.0, 1),
            "biotic_pct": round(p_biotic_sup * 100.0, 1),
            "abiotic_pct": round(p_abiotic_sup * 100.0, 1),
        },
        "confidence": primary_conf,
        "is_healthy": primary == "Healthy_Foliage",
        "predictions": preds,
        "lesion_info": lesion_info,
        "surface_stress": stress_metrics,
    }
