"""
Stage 0: Objective Image Quality & Viability Gate (No Color Heuristics).
Evaluates spatial focus sharpness, exposure balance, and minimum resolution,
producing clear graceful guidance if an uploaded photo is unusable.
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image


def evaluate_image_quality(
    pil_img: Image.Image,
    foreground_mask: Optional[np.ndarray] = None,
    min_short_side: int = 224,
    min_laplacian_var: float = 32.0,
    min_tenengrad: float = 8.0,
    max_dark_pct: float = 65.0,
    max_bright_pct: float = 55.0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Runs non-color image quality diagnostics.
    Returns (is_usable, status_code, details_dict).
    status_code in {"OK", "LOW_RESOLUTION", "BLURRY", "TOO_DARK", "OVEREXPOSED"}.
    """
    w, h = pil_img.size
    short_side = min(w, h)
    if short_side < min_short_side:
        return False, "LOW_RESOLUTION", {
            "width": w,
            "height": h,
            "reason": f"Image resolution is too small ({w}x{h}px).",
            "recommendations": [
                "Take a fresh photo using the main camera instead of a compressed thumbnail.",
                "Move closer to the plant so leaf and surface details are clear."
            ],
        }

    # Standardize working size for resolution-invariant focus measurement
    scale = 512.0 / float(max(w, h))
    nw, nh = max(64, int(w * scale)), max(64, int(h * scale))
    rgb = np.array(pil_img.resize((nw, nh), Image.Resampling.BILINEAR))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Restrict sharpness/exposure focus check to foreground plant if mask supplied
    if foreground_mask is not None:
        m = cv2.resize(foreground_mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
        if np.count_nonzero(m) < 200:
            m = np.ones((nh, nw), dtype=bool)
    else:
        # Weight toward central 70% if mask not yet computed
        m = np.zeros((nh, nw), dtype=bool)
        y0, y1 = int(nh * 0.15), int(nh * 0.85)
        x0, x1 = int(nw * 0.15), int(nw * 0.85)
        m[y0:y1, x0:x1] = True

    gray_f = gray.astype(np.float32)
    lap = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap[m]))

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    tenengrad = float(np.mean(grad_mag[m]))

    roi_vals = gray[m]
    dark_pct = float(np.mean(roi_vals < 22) * 100.0)
    bright_pct = float(np.mean(roi_vals > 246) * 100.0)
    mean_luma = float(np.mean(roi_vals))

    # HSV Specular Glare Check (Value high, Saturation near zero)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv_roi = hsv[m]
    glare_mask = (hsv_roi[:, 2] >= 240) & (hsv_roi[:, 1] <= 30)
    glare_pct = float(np.mean(glare_mask) * 100.0) if len(hsv_roi) > 0 else 0.0

    metrics = {
        "width": w,
        "height": h,
        "laplacian_var": round(lap_var, 2),
        "tenengrad_mean": round(tenengrad, 2),
        "dark_clipped_pct": round(dark_pct, 1),
        "bright_clipped_pct": round(bright_pct, 1),
        "mean_luminance": round(mean_luma, 1),
        "glare_pct": round(glare_pct, 1),
    }

    if glare_pct > 15.0:
        return False, "GLARE_DETECTED", {
            **metrics,
            "reason": "Glare detected on leaf — tilt slightly away from direct light.",
            "recommendations": [
                "Tilt the phone or plant slightly away from direct light or harsh window glare.",
                "Diffuse overhead lighting or turn off camera flash to prevent specular reflection."
            ],
        }

    if dark_pct > max_dark_pct or mean_luma < 28.0:
        return False, "TOO_DARK", {
            **metrics,
            "reason": f"Image is severely underexposed ({dark_pct:.0f}% shadow clipped).",
            "recommendations": [
                "Move the plant near a bright window or turn on overhead lighting.",
                "Avoid shooting directly against a bright window (backlighting darkens the plant)."
            ],
        }

    if bright_pct > max_bright_pct or mean_luma > 240.0:
        return False, "OVEREXPOSED", {
            **metrics,
            "reason": f"Severe glare or overexposure washed out surface details ({bright_pct:.0f}% blown highlights).",
            "recommendations": [
                "Step slightly out of direct harsh midday glare or angle away from grow lights.",
                "Tap the leaf surface on your phone screen to lock exposure before capturing."
            ],
        }

    if lap_var < min_laplacian_var and tenengrad < min_tenengrad:
        return False, "BLURRY", {
            **metrics,
            "reason": f"Image appears out of focus or motion-blurred (sharpness={lap_var:.1f}).",
            "recommendations": [
                "Wipe your camera lens gently before capturing.",
                "Hold the phone steady 25-40 cm from the plant and tap the leaf/stem to autofocus.",
                "Take the shot in brighter lighting so shutter speed stays crisp."
            ],
        }

    return True, "OK", metrics


def normalize_illumination_lab(pil_img: Image.Image) -> Image.Image:
    """
    Applies CLAHE on the L* channel in CIELAB color space before passing the image
    downstream to neutralize warm indoor LEDs, shadows, and window backlighting.
    Never uses raw RGB color math.
    """
    rgb = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    norm_lab = cv2.merge((cl, a_channel, b_channel))
    norm_rgb = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(norm_rgb)

