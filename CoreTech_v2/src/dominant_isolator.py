"""
Stage 1B: Multi-Plant Instance Segmentation & Dominant Plant Isolation.
Separates overlapping plants, excludes pots/soil/furniture, and isolates ONLY the plant
with the highest pixel coverage in the picture before Species ID and Surface Disease analysis.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

V2_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = V2_ROOT / "artifacts"

_cached_seg_model = None
_cached_seg_mode = None  # "trained_yolo" or "yoloe_zeroshot"


def load_plant_segmenter(device: str = "cuda"):
    global _cached_seg_model, _cached_seg_mode
    if _cached_seg_model is not None:
        return _cached_seg_model, _cached_seg_mode

    trained_path = ARTIFACTS_DIR / "yolo_plant_seg.pt"
    yoloe_path = ARTIFACTS_DIR / "yoloe-11l-seg.pt"

    if trained_path.exists():
        from ultralytics import YOLO
        m = YOLO(str(trained_path))
        m.to(device)
        _cached_seg_model = m
        _cached_seg_mode = "trained_yolo"
        return _cached_seg_model, _cached_seg_mode

    if yoloe_path.exists():
        from ultralytics import YOLOE
        m = YOLOE(str(yoloe_path))
        prompts = ["potted plant", "plant foliage", "cactus", "succulent", "houseplant leaves"]
        m.set_classes(prompts, m.get_text_pe(prompts))
        m.to(device)
        _cached_seg_model = m
        _cached_seg_mode = "yoloe_zeroshot"
        return _cached_seg_model, _cached_seg_mode

    return None, None


def isolate_dominant_plant(
    pil_img: Image.Image,
    conf_thresh: float = 0.18,
    min_coverage_pct: float = 3.0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Detects all plant instances in pil_img, selects the instance with highest coverage,
    suppresses secondary background plants, and returns clean crops & aligned masks.
    """
    rgb = np.array(pil_img.convert("RGB"))
    h, w, _ = rgb.shape
    total_px = float(h * w)

    model, mode = load_plant_segmenter(device=device)
    if model is None:
        # Clean fallback (should never happen once trained_yolo is built)
        full_mask = np.ones((h, w), dtype=bool)
        return {
            "detected": True,
            "instances_count": 1,
            "dominant_coverage_pct": 100.0,
            "dominant_conf": 1.0,
            "dominant_box": [0, 0, w, h],
            "crop_img": pil_img,
            "masked_crop_img": pil_img,
            "mask_crop": full_mask,
            "full_mask": full_mask,
            "overlay_img": pil_img,
            "mode": "unmasked_fallback",
        }

    res = model.predict(pil_img, conf=conf_thresh, imgsz=640, verbose=False)[0]
    if res.masks is None or len(res.masks.data) == 0:
        return {
            "detected": False,
            "instances_count": 0,
            "dominant_coverage_pct": 0.0,
            "dominant_conf": 0.0,
            "dominant_box": None,
            "crop_img": pil_img,
            "masked_crop_img": pil_img,
            "mask_crop": np.zeros((h, w), dtype=bool),
            "full_mask": np.zeros((h, w), dtype=bool),
            "overlay_img": pil_img,
            "mode": mode,
        }

    # Resize masks cleanly to original image resolution
    raw_masks = res.masks.data.detach().cpu().numpy()
    confs = res.boxes.conf.detach().cpu().numpy()
    boxes = res.boxes.xyxy.detach().cpu().numpy()

    candidates = []
    cx_img, cy_img = w / 2.0, h / 2.0
    max_dist = np.hypot(cx_img, cy_img) + 1e-6

    for idx in range(len(raw_masks)):
        m = cv2.resize(raw_masks[idx], (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
        px_area = int(np.count_nonzero(m))
        cov_pct = (px_area / total_px) * 100.0
        if cov_pct < 0.8:
            continue

        ys, xs = np.where(m)
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        center_score = 1.0 - (np.hypot(mx - cx_img, my - cy_img) / max_dist)
        # Highest coverage dominates (90% pixel area, 10% mild center tie-breaker)
        score = cov_pct * (0.90 + 0.10 * center_score)
        candidates.append({
            "idx": idx,
            "mask": m,
            "area": px_area,
            "cov_pct": cov_pct,
            "conf": float(confs[idx]),
            "box": [int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])],
            "score": score,
        })

    if not candidates:
        return {
            "detected": False,
            "instances_count": 0,
            "dominant_coverage_pct": 0.0,
            "dominant_conf": 0.0,
            "dominant_box": None,
            "crop_img": pil_img,
            "masked_crop_img": pil_img,
            "mask_crop": np.zeros((h, w), dtype=bool),
            "full_mask": np.zeros((h, w), dtype=bool),
            "overlay_img": pil_img,
            "mode": mode,
        }

    # Deduplicate heavily overlapping detections (if open-vocab YOLOE emitted nested prompts for same plant)
    candidates.sort(key=lambda c: c["area"], reverse=True)
    merged = []
    for c in candidates:
        is_same_plant = False
        for kept in merged:
            inter = np.count_nonzero(c["mask"] & kept["mask"])
            smaller = min(c["area"], kept["area"])
            if inter / max(1, smaller) > 0.65:
                # Merge overlapping fragment masks into the primary plant instance
                kept["mask"] = kept["mask"] | c["mask"]
                kept["area"] = int(np.count_nonzero(kept["mask"]))
                kept["cov_pct"] = (kept["area"] / total_px) * 100.0
                kept["conf"] = max(kept["conf"], c["conf"])
                is_same_plant = True
                break
        if not is_same_plant:
            merged.append(c)

    # Re-score merged instances and pick dominant highest-coverage plant
    for c in merged:
        ys, xs = np.where(c["mask"])
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        center_score = 1.0 - (np.hypot(mx - cx_img, my - cy_img) / max_dist)
        c["score"] = c["cov_pct"] * (0.90 + 0.10 * center_score)

    merged.sort(key=lambda c: c["score"], reverse=True)
    dom = merged[0]
    dom_mask = dom["mask"]

    # Tight bounding box around dominant mask with 6% padding
    ys, xs = np.where(dom_mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad_x = max(8, int((x2 - x1) * 0.06))
    pad_y = max(8, int((y2 - y1) * 0.06))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    # Suppress secondary background plants inside the dominant box (blur + dim non-dominant pixels)
    soft_bg = cv2.GaussianBlur(rgb, (31, 31), 0)
    soft_bg = (soft_bg.astype(np.float32) * 0.45 + 128.0 * 0.55).astype(np.uint8)
    clean_full = np.where(dom_mask[:, :, None], rgb, soft_bg)

    neutral_bg = np.full_like(rgb, 128)
    masked_full = np.where(dom_mask[:, :, None], rgb, neutral_bg)

    crop_img = Image.fromarray(clean_full[y1:y2, x1:x2])
    masked_crop_img = Image.fromarray(masked_full[y1:y2, x1:x2])
    mask_crop = dom_mask[y1:y2, x1:x2]

    # Visual overlay showing dominant selected plant vs dimmed secondary plants
    vis = rgb.copy()
    tint = np.zeros_like(vis)
    tint[dom_mask] = (16, 185, 129)  # Emerald highlight
    vis = np.where(dom_mask[:, :, None], cv2.addWeighted(vis, 0.82, tint, 0.18, 0), vis)
    contours, _ = cv2.findContours(dom_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (16, 185, 129), max(2, min(w, h) // 250))

    return {
        "detected": dom["cov_pct"] >= min_coverage_pct,
        "instances_count": len(merged),
        "dominant_coverage_pct": round(dom["cov_pct"], 1),
        "dominant_conf": round(dom["conf"], 3),
        "dominant_box": [x1, y1, x2, y2],
        "crop_img": crop_img,
        "masked_crop_img": masked_crop_img,
        "mask_crop": mask_crop,
        "full_mask": dom_mask,
        "overlay_img": Image.fromarray(vis),
        "mode": mode,
    }
