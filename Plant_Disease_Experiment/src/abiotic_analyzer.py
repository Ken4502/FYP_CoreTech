"""
Abiotic Stress & Foliar Senescence Analyzer for PhytoCare Ultra
Analyzes foliar colorimetry (CIELAB & HSV spaces), chlorophyll degradation,
margin crispness, and spatial chlorosis gradients to diagnose:
  1. Underwatering / Drought Stress (Crispy margin necrosis + chlorophyll desaturation)
  2. Natural Leaf Senescence / Old Age (Smooth uniform yellow gradient + intact veins + 0 fungal spots)
  3. Overwatering / Edema (Blotchy water-soaked translucent tissue)
  4. Optimal Foliar Hydration & Vigor
"""

import numpy as np
from PIL import Image
import cv2

def analyze_abiotic_stress(pil_img: Image.Image, spots_count: int = 0) -> dict:
    """
    Performs algorithmic colorimetric and morphological analysis on plant foliage.
    """
    # Convert PIL Image to BGR (OpenCV format)
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    h, w, _ = bgr.shape
    total_pixels = h * w

    # 1. Segment Foliage Mask (Exclude white/dark sterile studio background)
    # Background is typically near-white (V > 230, S < 30) or near-black (V < 30)
    bg_white = (hsv[:, :, 1] < 28) & (hsv[:, :, 2] > 215)
    bg_black = (hsv[:, :, 2] < 25)
    leaf_mask = ~(bg_white | bg_black)

    leaf_pixel_count = np.count_nonzero(leaf_mask)
    if leaf_pixel_count < total_pixels * 0.05:
        # Fallback if masking is too aggressive
        leaf_mask = np.ones((h, w), dtype=bool)
        leaf_pixel_count = total_pixels

    # Extract channels for leaf pixels only
    h_leaf = hsv[:, :, 0][leaf_mask]
    s_leaf = hsv[:, :, 1][leaf_mask]
    v_leaf = hsv[:, :, 2][leaf_mask]
    l_leaf = lab[:, :, 0][leaf_mask]
    a_leaf = lab[:, :, 1][leaf_mask]  # Negative = Green, Positive = Magenta/Red
    b_leaf = lab[:, :, 2][leaf_mask]  # Positive = Yellow, Negative = Blue

    # 2. Metric A: Chlorophyll Green Vigor (CIELAB a* channel)
    # Deep healthy green: a* <= 118 (OpenCV scales LAB to 0-255, 128 is neutral 0)
    green_pixels = np.count_nonzero(a_leaf < 122)
    vigor_pct = round((green_pixels / leaf_pixel_count) * 100, 1)

    # 3. Metric B: Chlorosis / Yellowing Ratio
    # Yellow foliage: HSV Hue 20 to 38 (OpenCV hue 0-180) and b* > 138 (Yellow)
    yellow_mask_full = (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 38) & (hsv[:, :, 1] > 35) & leaf_mask
    yellow_pixel_count = np.count_nonzero(yellow_mask_full)
    chlorosis_pct = round((yellow_pixel_count / leaf_pixel_count) * 100, 1)

    # 4. Metric C: Brown Dry Margin Crispness (Underwatering Necrosis)
    # Brown papery crisp: Hue 8 to 18, Saturation 40-180, Low Value (V 40-140)
    brown_mask_full = (hsv[:, :, 0] >= 6) & (hsv[:, :, 0] <= 18) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] <= 160) & leaf_mask
    brown_pixel_count = np.count_nonzero(brown_mask_full)
    crispness_pct = round((brown_pixel_count / leaf_pixel_count) * 100, 1)

    # 5. Metric D: Mean Saturation & Moisture Index
    mean_sat = float(np.mean(s_leaf)) / 255.0 * 100.0
    mean_val = float(np.mean(v_leaf)) / 255.0 * 100.0

    # 6. Spatial Smoothness Analysis: Senescence vs Lesion
    # Natural senescence is a smooth, continuous spatial gradient with low edge variance
    # Pathological spots have high Laplacian edge sharpness and jagged contours
    is_smooth_gradient = False
    if yellow_pixel_count > leaf_pixel_count * 0.08:
        # Check gradient smoothness of yellow area
        yellow_img = np.zeros((h, w), dtype=np.uint8)
        yellow_img[yellow_mask_full] = 255
        blurred = cv2.GaussianBlur(yellow_img, (15, 15), 0)
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If there are 1 or 2 large smooth connected zones (not 10 scattered spots)
        if len(contours) in [1, 2, 3] and spots_count == 0:
            is_smooth_gradient = True

    # 7. Diagnostic Logic & Classification
    water_status = "Optimal Hydration"
    water_severity = "Normal"
    aging_status = "Young / Mature Vigorous"
    aging_severity = "Normal"
    remedial_advice = []
    status_summary = ""

    # Check Underwatering (Drought Stress)
    if crispness_pct >= 5.0 or (crispness_pct >= 2.5 and mean_sat < 35.0):
        water_status = "Foliar Dehydration (Underwatering)"
        water_severity = "High" if crispness_pct > 12.0 else "Moderate"
        status_summary = f"Foliar moisture analysis indicates water deficit stress. Peripheral leaf margins show {crispness_pct}% papery brown crisping and cellular turgor loss."
        remedial_advice = [
            "Perform a deep bottom-soak (submerge bottom 2 inches of nursery pot in lukewarm water for 30-45 minutes).",
            "Prune completely crispy, dead tip margins with clean scissors (leave a thin margin of brown to avoid cutting live green veins).",
            "Increase ambient humidity using a pebble tray or room humidifier to slow down foliar transpiration."
        ]
    elif mean_sat < 32.0 and vigor_pct < 60.0:
        water_status = "Mild Dehydration / Desaturation"
        water_severity = "Low"
        status_summary = "Foliage shows slight desaturation and reduced chlorophyll glossiness, consistent with early moisture depletion."
        remedial_advice = [
            "Check soil moisture 2 inches below surface; water thoroughly if soil feels dry and crumbly."
        ]
    # Check Overwatering / Edema
    elif mean_val > 70.0 and chlorosis_pct > 20.0 and spots_count == 0:
        water_status = "Overwatering / Moisture Saturation"
        water_severity = "Moderate"
        status_summary = "High tissue translucency and diffuse yellowing without crisp margins suggest roots are suffocating from excessive soil water retention."
        remedial_advice = [
            "Cease watering immediately and allow the root ball to dry out.",
            "Empty any standing water pooled in the decorative saucer.",
            "Aerate potting soil by gently poking 4-5 holes around root perimeter with a chopstick to introduce oxygen."
        ]
    else:
        water_status = "Optimal Foliar Turgor & Hydration"
        water_severity = "Normal"
        status_summary = f"Healthy foliar turgor with {vigor_pct}% green chlorophyll index and normal leaf hydration."
        remedial_advice = [
            "Maintain current watering schedule.",
            "Always feel soil dryness before watering."
        ]

    # Check Natural Leaf Senescence (Old Age)
    if is_smooth_gradient and spots_count == 0 and chlorosis_pct >= 8.0:
        aging_status = "Natural Leaf Senescence (Old Age)"
        aging_severity = "Physiological (Harmless)"
        status_summary += f" Smooth continuous yellowing ({chlorosis_pct}%) without necrotic lesion margins indicates natural nutrient recycling (nitrogen resorption) on mature lower leaves."
        remedial_advice.append("This is healthy plant maturation! Allow the leaf to fully yellow so the plant finishes recycling nutrients, then gently snip at the petiole base.")
    elif chlorosis_pct >= 25.0 and spots_count > 0:
        aging_status = "Pathological Chlorosis (Disease-Induced)"
        aging_severity = "Pathology"
    else:
        aging_status = "Active Vegetative Photosynthesis"
        aging_severity = "Normal"

    return {
        "water_status": water_status,
        "water_severity": water_severity,
        "aging_status": aging_status,
        "aging_severity": aging_severity,
        "chlorosis_pct": chlorosis_pct,
        "crispness_pct": crispness_pct,
        "vigor_pct": vigor_pct,
        "moisture_index": round(mean_sat, 1),
        "status_summary": status_summary,
        "abiotic_actions": remedial_advice
    }

