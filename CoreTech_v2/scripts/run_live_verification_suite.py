"""
Live Verification Suite for CoreTech v2.
Tests both Botanical Identification (269 Taxa) and Surface Pathology Diagnosis (19 Classes)
across real test specimens using the unified pipeline_v2 and the web server endpoint.
"""
import sys
import time
import json
from pathlib import Path
from PIL import Image

CORETECH_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CORETECH_ROOT))
sys.path.insert(0, str(CORETECH_ROOT / "Plant_Disease_Experiment"))

from fastapi.testclient import TestClient
from web.app import app
from CoreTech_v2.src.pipeline_v2 import run_v2_diagnosis

TEST_SPECIMENS = [
    {
        "category": "Botanical ID (Succulent / Cactaceae)",
        "ground_truth": "Golden Barrel Cactus (Echinocactus grusonii)",
        "path": r"C:\Users\User\Desktop\FYP\CoreTech\CoreTech_v2\data\botanical_id\test\Golden Barrel Cactus (Echinocactus grusonii)\exp_test_inat_407191142.jpg",
    },
    {
        "category": "Botanical ID (Succulent / Asphodelaceae)",
        "ground_truth": "Aloe Vera",
        "path": r"C:\Users\User\Desktop\FYP\CoreTech\CoreTech_v2\data\botanical_id\test\Aloe Vera\exp_test_242.jpg",
    },
    {
        "category": "Botanical ID (Indoor Foliage / Moraceae)",
        "ground_truth": "Fiddle Leaf Fig (Ficus lyrata)",
        "path": r"C:\Users\User\Desktop\FYP\CoreTech\CoreTech_v2\data\botanical_id\test\Fiddle Leaf Fig (Ficus lyrata)\exp_test_mega_161004404.jpg",
    },
    {
        "category": "Pathology (Fungal Foliar Disease)",
        "ground_truth": "Powdery Mildew",
        "path": r"C:\Users\User\Desktop\FYP\CoreTech\CoreTech_v2\data\surface_pathology\test\Powdery_Mildew\leg_Universal_Po_08c759b468.jpg",
    },
    {
        "category": "Pathology (Acarid Pest Infestation)",
        "ground_truth": "Spider Mites",
        "path": r"C:\Users\User\Desktop\FYP\CoreTech\CoreTech_v2\data\surface_pathology\test\Spider_Mites\leg_Universal_Sp_02273c5d0f.jpg",
    },
    {
        "category": "Pathology (Abiotic Environmental Stress)",
        "ground_truth": "Abiotic Sunburn / Solar Scorch",
        "path": r"C:\Users\User\Desktop\FYP\CoreTech\CoreTech_v2\data\surface_pathology\test\Abiotic_Sunburn_Scald\leg_Aloe_Sunburn_04375ecab1.jpg",
    },
    {
        "category": "Pathology (Healthy Control Specimen)",
        "ground_truth": "Healthy Foliage (Clean)",
        "path": r"C:\Users\User\Desktop\FYP\CoreTech\CoreTech_v2\data\surface_pathology\test\Healthy_Foliage\hhneg_0186601a4d.jpg",
    },
]

def main():
    print("=" * 80)
    print("🌿 FYP CORETECH v2: LIVE DUAL VERIFICATION SUITE")
    print("Testing Botanical Identification (269 Taxa) & Pathology Diagnosis (19 Classes)")
    print("=" * 80)

    # 1. Warm-up run to ensure all weights are resident in VRAM
    print("\n[*] Initializing & warming up model suite in VRAM...")
    t_warm = time.time()
    warm_img = Image.open(TEST_SPECIMENS[0]["path"]).convert("RGB")
    _ = run_v2_diagnosis(warm_img)
    print(f"[✓] Model suite resident in VRAM in {time.time() - t_warm:.2f}s.\n")

    # 2. TestClient initialization for API endpoint verification
    client = TestClient(app)

    results = []

    for idx, item in enumerate(TEST_SPECIMENS, 1):
        cat = item["category"]
        gt = item["ground_truth"]
        fpath = Path(item["path"])

        print(f"{'-' * 80}")
        print(f"[Specimen #{idx}] {cat}")
        print(f"  • Ground Truth  : {gt}")
        print(f"  • File Path     : {fpath.name}")

        if not fpath.exists():
            print(f"  [ERROR] File not found: {fpath}")
            continue

        pil_img = Image.open(fpath).convert("RGB")

        # Test A: Direct Engine Execution (pipeline_v2.py)
        t0 = time.time()
        res_engine = run_v2_diagnosis(pil_img)
        lat_engine_ms = (time.time() - t0) * 1000.0

        # Test B: Full Web API Endpoint Execution (/api/predict)
        t_api = time.time()
        with open(fpath, "rb") as f:
            api_resp = client.post("/api/predict", files={"file": (fpath.name, f, "image/jpeg")})
        lat_api_ms = (time.time() - t_api) * 1000.0
        api_data = api_resp.json() if api_resp.status_code == 200 else {}

        # Parse Engine Results
        sp = res_engine.get("species", {})
        diag = res_engine.get("diagnosis", {})
        spots = res_engine.get("spots", {})
        iso = res_engine.get("isolation_meta", {})
        care = res_engine.get("care_guide", {})

        pred_species = sp.get("name", "Unknown")
        pred_conf = sp.get("confidence", 0.0)
        pred_diag = diag.get("condition_name", "Unknown")
        diag_conf = diag.get("confidence", 0.0)
        is_healthy = diag.get("is_healthy", False)
        sev = diag.get("severity", "None")
        spots_cnt = spots.get("spots_count", 0)
        inf_area = spots.get("infected_area_pct", 0.0)
        coverage = iso.get("dominant_coverage_pct", 0.0)
        pdi_label = api_data.get("pdi_rating", {}).get("label", "N/A")

        print(f"  • Predicted Plant  : {pred_species} (Confidence: {pred_conf:.1f}%)")
        print(f"  • Dominant Coverage: {coverage:.1f}% of frame (Isolated by YOLO11m-Seg)")
        print(f"  • Surface Diagnosis: {pred_diag} (Confidence: {diag_conf:.1f}%)")
        print(f"  • Healthy Status   : {'HEALTHY' if is_healthy else 'DISEASED / STRESSED'} (Severity: {sev})")
        print(f"  • Lesion Spotting  : {spots_cnt} box(es) localized, {inf_area:.1f}% infected tissue area")
        print(f"  • PDI Rating Meter : {pdi_label}")
        print(f"  • Latency Profile  : Engine: {lat_engine_ms:.1f}ms | API Endpoint: {lat_api_ms:.1f}ms")
        print(f"  • Immediate Action : {care.get('immediate_actions', ['N/A'])[0] if care.get('immediate_actions') else 'N/A'}")

        results.append({
            "specimen": idx,
            "category": cat,
            "ground_truth": gt,
            "pred_species": pred_species,
            "species_conf": round(pred_conf, 1),
            "pred_diagnosis": pred_diag,
            "diagnosis_conf": round(diag_conf, 1),
            "is_healthy": is_healthy,
            "spots_count": spots_cnt,
            "latency_engine_ms": round(lat_engine_ms, 1),
            "latency_api_ms": round(lat_api_ms, 1),
            "api_status": api_resp.status_code
        })

    print(f"\n{'=' * 80}")
    print("LIVE VERIFICATION SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"{'ID':<3} | {'Ground Truth':<32} | {'Predicted Species':<28} | {'Predicted Diagnosis':<28} | {'Engine ms':<9}")
    print(f"{'-' * 110}")
    for r in results:
        print(f"{r['specimen']:<3} | {r['ground_truth']:<32} | {r['pred_species'][:26]:<28} | {r['pred_diagnosis'][:26]:<28} | {r['latency_engine_ms']:<9}")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    main()
