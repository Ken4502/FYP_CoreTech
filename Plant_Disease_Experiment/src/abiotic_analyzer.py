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
    bg_white = (hsv[:, :, 1] < 25) & (hsv[:, :, 2] > 225)
    bg_black = (hsv[:, :, 2] < 20)
    raw_leaf_mask = ~(bg_white | bg_black)

    # Fill internal leaf holes so bleached sunburn patches, powdery mildew, and lesions are not discarded as background
    contours, _ = cv2.findContours(raw_leaf_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_mask = np.zeros((h, w), dtype=bool)
    if contours:
        valid_cnts = [c for c in contours if cv2.contourArea(c) >= total_pixels * 0.02]
        if valid_cnts:
            filled_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(filled_mask, valid_cnts, -1, 1, thickness=cv2.FILLED)
            leaf_mask = filled_mask > 0
        else:
            leaf_mask = raw_leaf_mask
    else:
        leaf_mask = raw_leaf_mask

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

    # 6. Metric E: Surface Wrinkle Index & Wild Leaf Contour Tortuosity
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    wrinkle_index = round(float(np.percentile(lap_abs[leaf_mask], 85)), 2) if leaf_pixel_count > 0 else 0.0

    contours, _ = cv2.findContours(leaf_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
    tortuosity = round(float((total_perimeter ** 2) / (4.0 * np.pi * max(1, leaf_pixel_count))), 2)

    # 7. Spatial Smoothness Analysis: Senescence vs Lesion
    # Natural senescence is a smooth, continuous spatial gradient with low edge variance
    # Pathological spots have high Laplacian edge sharpness and jagged contours
    is_smooth_gradient = False
    if yellow_pixel_count > leaf_pixel_count * 0.08:
        # Check gradient smoothness of yellow area
        yellow_img = np.zeros((h, w), dtype=np.uint8)
        yellow_img[yellow_mask_full] = 255
        blurred = cv2.GaussianBlur(yellow_img, (15, 15), 0)
        c_yellow, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If there are 1 or 2 large smooth connected zones (not 10 scattered spots)
        if len(c_yellow) in [1, 2, 3] and spots_count == 0:
            is_smooth_gradient = True

    # 8. Diagnostic Logic & Classification
    water_status = "Optimal Hydration"
    water_severity = "Normal"
    aging_status = "Young / Mature Vigorous"
    aging_severity = "Normal"
    remedial_advice = []
    status_summary = ""

    # Check Underwatering (Drought Stress)
    is_underwatered = bool(
        crispness_pct >= 4.0 or
        (crispness_pct >= 1.8 and mean_sat < 38.0) or
        (wrinkle_index >= 20.0 and (crispness_pct >= 1.2 or mean_sat < 40.0)) or
        (wrinkle_index >= 27.0) or
        (tortuosity >= 3.5 and wrinkle_index >= 18.0 and (crispness_pct >= 1.0 or mean_sat < 42.0))
    )
    severe_underwatering = bool(
        (wrinkle_index >= 28.0 and crispness_pct >= 3.5) or
        (crispness_pct >= 9.0 and wrinkle_index >= 18.0) or
        (wrinkle_index >= 36.0)
    )

    # Check Overwatering / Edema
    is_overwatered = bool(
        (chlorosis_pct >= 14.0 and crispness_pct <= 2.2 and wrinkle_index <= 18.0) or
        (chlorosis_pct >= 20.0 and crispness_pct <= 4.0 and mean_sat <= 48.0) or
        (mean_val >= 62.0 and chlorosis_pct >= 10.0 and crispness_pct <= 2.0 and wrinkle_index <= 16.0)
    ) and not is_underwatered
    severe_overwatering = bool(
        (chlorosis_pct >= 28.0 and crispness_pct <= 3.0 and wrinkle_index <= 16.0) or
        (chlorosis_pct >= 40.0 and crispness_pct <= 5.0)
    ) and is_overwatered

    # Check Sunburn / Solar Scorch
    bleached_mask = (s_leaf < 35) & (v_leaf > 185)
    bleached_pct = round(float(np.mean(bleached_mask) * 100.0), 1) if leaf_pixel_count > 0 else 0.0
    is_sunburned = bool(
        (bleached_pct >= 5.0 and wrinkle_index <= 22.0) or
        (bleached_pct >= 3.5 and (crispness_pct >= 1.2 or chlorosis_pct >= 6.0) and wrinkle_index <= 22.0)
    ) and not is_underwatered and not is_overwatered

    if is_underwatered:
        water_status = "Foliar Dehydration (Underwatering)"
        water_severity = "High" if severe_underwatering else "Moderate"
        status_summary = (
            f"Foliar moisture analysis indicates water deficit stress ({wrinkle_index} wrinkle index, "
            f"{crispness_pct}% papery brown margin crisping). Symptoms represent physiological dehydration, "
            f"NOT infectious fungal rust or insect bite damage."
        )
        remedial_advice = [
            "Perform a deep bottom-soak (submerge bottom 2 inches of nursery pot in lukewarm water for 30-45 minutes).",
            "Do NOT apply chemical fungicides or pesticides; foliage is suffering from drought stress, not rust or insect chew.",
            "Prune only completely crispy, dead tip margins with clean scissors (leave a thin brown buffer to protect healthy live veins).",
            "Increase ambient humidity using a pebble tray or room humidifier to slow down foliar transpiration."
        ]
    elif is_overwatered:
        water_status = "Overwatering / Moisture Saturation"
        water_severity = "High" if severe_overwatering else "Moderate"
        status_summary = (
            f"Foliar moisture analysis indicates waterlogging and root hypoxia ({chlorosis_pct}% diffuse "
            f"yellow chlorosis, high tissue translucency). Corky bumps or edema blisters represent physiological "
            f"cellular rupture, NOT scale insects or bacterial blight."
        )
        remedial_advice = [
            "Cease watering immediately and allow the top 50% of potting mix to dry out thoroughly.",
            "Empty any standing water pooled in the decorative saucer and ensure drainage holes are completely open.",
            "Aerate potting soil by gently poking 4-5 holes around root perimeter with a chopstick to introduce root oxygen.",
            "Do NOT apply chemical fungicides or pesticides; the condition is root suffocation from moisture excess."
        ]
    elif is_sunburned:
        water_status = "Solar Scorch / Sunburn"
        water_severity = "Moderate"
        status_summary = f"Foliar analysis indicates solar radiation scorch ({bleached_pct}% bleached leaf tissue). Non-infectious sunscald, NOT fungal leaf spot."
        remedial_advice = [
            "Move plant immediately back from direct midday sun into bright indirect, filtered light.",
            "Do NOT prune mildly scorched leaves that still retain healthy green photosynthetic tissue.",
            "Ensure consistent root hydration to help foliage withstand ambient heat."
        ]
    elif mean_sat < 32.0 and vigor_pct < 60.0:
        water_status = "Mild Dehydration / Desaturation"
        water_severity = "Low"
        status_summary = "Foliage shows slight desaturation and reduced chlorophyll glossiness, consistent with early moisture depletion."
        remedial_advice = [
            "Check soil moisture 2 inches below surface; water thoroughly if soil feels dry and crumbly."
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

    stress_level = water_severity if (is_underwatered or is_overwatered) else ("Moderate" if (water_status != "Optimal Foliar Turgor & Hydration" or is_sunburned) else "Low")
    if is_underwatered:
        stress_type = "Underwatering Dehydration"
    elif is_overwatered:
        stress_type = "Overwatering / Waterlogging"
    elif is_sunburned:
        stress_type = "Solar Scorch / Sunburn"
    else:
        stress_type = water_status if water_status != "Optimal Foliar Turgor & Hydration" else "Normal"

    recommendation = remedial_advice[0] if remedial_advice else "Optimal foliar hydration detected."

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
        "abiotic_actions": remedial_advice,
        # Flutter & Web unified keys:
        "chlorosis_score": chlorosis_pct,
        "necrosis_score": crispness_pct,
        "stress_level": stress_level,
        "stress_type": stress_type,
        "recommendation": recommendation,
        "is_underwatered": is_underwatered,
        "severe_underwatering": severe_underwatering,
        "is_overwatered": is_overwatered,
        "severe_overwatering": severe_overwatering,
        "is_sunburned": is_sunburned,
        "bleached_pct": bleached_pct,
        "surface_wrinkle_index": wrinkle_index,
        "contour_tortuosity": tortuosity,
    }

