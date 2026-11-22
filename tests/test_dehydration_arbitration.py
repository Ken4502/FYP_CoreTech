"""
Test Suite: Dehydration / Abiotic Stress Arbitration vs False-Positive Rust & Insect Bites
Validates that:
1. Dehydrated leaves with high wrinkle index / curling wild margins / dry margins are correctly diagnosed
   as Abiotic_Dehydration_Underwatering / Surface Dehydration Stress.
2. Spurious Rust Fungi (Pucciniales) and Leaf Miners / Insect Bite chew detections are suppressed.
3. Abiotic payload fields expected by Flutter (chlorosis_score, necrosis_score, stress_level, stress_type, recommendation)
   are populated cleanly.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

CORETECH_ROOT = Path(r"c:\Users\user\Desktop\FYP\CoreTech")
EXPERIMENT_ROOT = CORETECH_ROOT / "Plant_Disease_Experiment"

sys.path.insert(0, str(CORETECH_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))

def create_synthetic_dehydrated_leaf(width=400, height=400) -> Image.Image:
    """Creates a synthetic image of a dehydrated, curled/wrinkled leaf with crispy dry brown margins."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Background (light gray indoor wall)
    img[:] = (235, 238, 240)

    # Draw wavy/wild leaf blade
    center = (width // 2, height // 2)
    pts = []
    num_pts = 40
    for i in range(num_pts):
        angle = 2.0 * np.pi * i / num_pts
        # Base elliptical radius + wild wavy perturbation (curling/rippling)
        r_base = 130 + 35 * np.cos(2 * angle) + 20 * np.sin(5 * angle)
        x = int(center[0] + r_base * np.cos(angle))
        y = int(center[1] + r_base * np.sin(angle))
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)

    # Fill base wilted green/olive color (desaturated green: H~40, S~70, V~130 -> BGR)
    cv2.fillPoly(img, [pts], (45, 115, 75))

    # Add pronounced wrinkle ridges / veins across the leaf blade (high Laplacian variance)
    for x in range(center[0] - 110, center[0] + 110, 14):
        cv2.line(img, (x, center[1] - 120), (x + np.random.randint(-15, 15), center[1] + 120), (35, 85, 55), 2)
    for y in range(center[1] - 100, center[1] + 100, 16):
        cv2.line(img, (center[0] - 120, y), (center[0] + 120, y + np.random.randint(-12, 12)), (40, 95, 60), 2)

    # Add papery dry brown crispy necrosis along perimeter edges
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    margin_mask = (mask > 0) & (eroded == 0)
    # Brown papery BGR: (30, 75, 140)
    img[margin_mask] = (35, 80, 145)

    # Convert BGR to RGB for PIL
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def test_abiotic_analyzer():
    print("[*] Testing abiotic_analyzer.py...")
    from src.abiotic_analyzer import analyze_abiotic_stress
    leaf_img = create_synthetic_dehydrated_leaf()
    res = analyze_abiotic_stress(leaf_img, spots_count=2)
    print("    Abiotic results:", res)
    assert res["is_underwatered"] is True, f"Expected is_underwatered=True, got {res['is_underwatered']}"
    assert "Dehydration" in res["water_status"] or "Underwatering" in res["water_status"]
    assert res["stress_level"] in ["Moderate", "High"]
    assert "stress_type" in res and "recommendation" in res
    assert res["chlorosis_score"] >= 0.0
    assert res["necrosis_score"] >= 0.0
    print("    [✓] abiotic_analyzer successfully detected foliar dehydration & turgor loss.")

def test_v2_surface_pathology():
    print("\n[*] Testing CoreTech_v2 surface_pathology.py dehydration arbitration...")
    from CoreTech_v2.src.surface_pathology import analyze_masked_surface_stress, diagnose_surface_condition
    leaf_img = create_synthetic_dehydrated_leaf()
    # Mask is non-background
    rgb = np.array(leaf_img)
    mask = (rgb[:, :, 0] < 220) | (rgb[:, :, 1] < 220) | (rgb[:, :, 2] < 220)
    stress = analyze_masked_surface_stress(leaf_img, mask)
    print("    Surface stress metrics:", stress)
    assert stress["hydration_stress"]["is_underwatered"] is True, "Failed to flag is_underwatered in V2"
    assert stress["has_compact_rust_pustules"] is False, "Incorrectly flagged compact rust pustules on dry margin"

    diag = diagnose_surface_condition(leaf_img, mask, plant_family="Solanaceae")
    print("    Diagnosed primary class:", diag["primary_class"])
    print("    Super state:", diag["super_state"])
    print("    Top predictions:", diag["predictions"][:3])
    assert diag["primary_class"] == "Abiotic_Dehydration_Underwatering", f"Expected Abiotic_Dehydration_Underwatering, got {diag['primary_class']}"
    assert diag["super_state"] == "Abiotic_Stress"
    assert diag["lesion_info"]["spots_count"] == 0, "Lesion spots should be suppressed for dehydration"
    print("    [✓] V2 surface pathology successfully arbitrated Dehydration over Rust/Miners.")

def test_api_predict():
    print("\n[*] Testing full FastAPI /api/predict endpoint with synthetic/real dehydrated specimen...")
    import io
    from fastapi.testclient import TestClient
    from web.app import app

    client = TestClient(app)
    real_img = CORETECH_ROOT / "CoreTech_v2/data/live_internet_stress_benchmark/underwatering_wilted/underwatering_wilted_407184271.jpg"
    if real_img.exists():
        leaf_img = Image.open(real_img)
    else:
        leaf_img = create_synthetic_dehydrated_leaf()
    buf = io.BytesIO()
    leaf_img.save(buf, format="JPEG")
    buf.seek(0)

    resp = client.post("/api/predict", files={"file": ("dehydrated_leaf.jpg", buf, "image/jpeg")}, data={"plant": "auto"})
    assert resp.status_code == 200, f"Expected status 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    print("    Diagnosis result:", data.get("diagnosis"))
    print("    Abiotic result:", data.get("abiotic"))
    print("    PDI Rating:", data.get("pdi_rating"))
    print("    Co-alerts:", data.get("co_alerts"))

    diag = data["diagnosis"]
    assert "Dehydration" in diag["condition_name"] or "Underwatering" in diag["condition_name"], f"Expected Dehydration in condition_name, got {diag['condition_name']}"
    assert diag["category"] == "Abiotic Water Stress" or "Abiotic" in diag["category"]
    assert diag["is_healthy"] is False
    assert data["pdi_rating"]["grade"] in (0, 2), "Expected clean/Grade 0 or Grade 2 (no pathogen quarantine) for dehydration"

    abiotic = data.get("abiotic", {})
    assert "chlorosis_score" in abiotic and "necrosis_score" in abiotic
    assert "stress_level" in abiotic and "recommendation" in abiotic
    print("    [✓] /api/predict successfully diagnosed Surface Dehydration & populated Flutter abiotic model.")

if __name__ == "__main__":
    test_abiotic_analyzer()
    test_v2_surface_pathology()
    test_api_predict()
    print("\n[✓] ALL ARBITRATION TESTS PASSED CLEANLY!")
