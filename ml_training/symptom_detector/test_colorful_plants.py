import io
import sys
from pathlib import Path
from PIL import Image

CORETECH_ROOT = Path(r"c:\Users\User\Desktop\FYP\CoreTech")
sys.path.insert(0, str(CORETECH_ROOT))
sys.path.insert(0, str(CORETECH_ROOT / "Plant_Disease_Experiment"))

from fastapi.testclient import TestClient
from web.app import app

def run_colorful_plant_test():
    client = TestClient(app)

    samples = [
        (CORETECH_ROOT / "Datasets/house_plant_species/Chinese evergreen (Aglaonema)/10.jpg", "Aglaonema (Chinese Evergreen)"),
        (CORETECH_ROOT / "Datasets/house_plant_species/Calathea/1.jpg", "Calathea (Patterned Foliage)"),
        (CORETECH_ROOT / "Datasets/house_plant_species/Polka Dot Plant (Hypoestes phyllostachya)/1.jpeg", "Hypoestes (Polka Dot Plant)"),
    ]

    print("=" * 72)
    print("REAL-WORLD HEALTHY COLORFUL & VARIEGATED PLANT AUDIT")
    print("=" * 72)

    for p, display_name in samples:
        if not p.exists():
            print(f"[!] File not found: {p}")
            continue

        img = Image.open(p)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = client.post("/api/predict", files={"file": (p.name, buf, "image/jpeg")}, data={"plant": "auto"})
        if resp.status_code != 200:
            print(f"[!] Error {resp.status_code}: {resp.text}")
            continue

        data = resp.json()
        diag = data.get("diagnosis", {})
        species = data.get("species", {})
        pdi = data.get("pdi_rating", {})
        abiotic = data.get("abiotic", {})
        dets = data.get("detections", [])

        print(f"\n[Specimen] {display_name} ({p.name})")
        print(f"  * Detected Species:   {species.get('name')} (conf={species.get('confidence')})")
        print(f"  * Visual Priors:      {species.get('visual_priors')}")
        print(f"  * Diagnosis Name:     {diag.get('condition_name')}")
        print(f"  * Class ID:           {diag.get('class_id')}")
        print(f"  * Category:           {diag.get('category')}")
        print(f"  * Is Healthy:         {diag.get('is_healthy')}")
        print(f"  * PDI Rating:         Grade {pdi.get('grade')} | {pdi.get('label')}")
        print(f"  * Needs Soil Check:   {data.get('needs_user_soil_check')}")
        print(f"  * Abiotic Status:     {abiotic.get('stress_level')} | {abiotic.get('stress_type')}")
        print(f"  * Recommendation:     {abiotic.get('recommendation')}")
        print(f"  * Stage 3 Detections: {[(d.get('class'), d.get('confidence'), d.get('display_label')) for d in dets]}")

if __name__ == "__main__":
    run_colorful_plant_test()
