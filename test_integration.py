import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

CORETECH_DIR = Path(__file__).resolve().parent
if str(CORETECH_DIR) not in sys.path:
    sys.path.insert(0, str(CORETECH_DIR))

# Test imports
print("1. Testing imports from CoreTech/preprocessing...")
from preprocessing.methods import PlantPreprocessors
from preprocessing.normalization import apply_clahe, apply_white_balance
from preprocessing.evaluator import MaskEvaluator

preprocessor = PlantPreprocessors()
print("   PlantPreprocessors successfully instantiated.")

sample_path = CORETECH_DIR.parent / "CoreTechPreProcessing" / "plantimage.jpg"
if not sample_path.exists():
    sample_path = CORETECH_DIR.parent / "CoreTechPreProcessing" / "samples" / "potted_plant_sample.jpg"

print(f"\n2. Testing Preprocessing on sample image: {sample_path}...")
img_pil = Image.open(sample_path).convert("RGB")
img_rgb = np.array(img_pil)

# Method 1
res_species = preprocessor.process_for_species_identification(
    img_rgb,
    bg_color=(128, 128, 128),
    padding_ratio=0.12,
    filter_pot_soil=True,
    preserve_leaf_shape=True
)
print("   Method 1 (Species ID) completed successfully!")
print(f"   Output shape: {res_species['cropped_image'].shape}, BBox: {res_species['bbox']}")

# Method 2
res_disease = preprocessor.process_for_disease_identification(
    img_rgb,
    bg_color=(128, 128, 128),
    padding_ratio=0.08,
    filter_pot_soil=True,
    preserve_leaf_shape=True
)
print("   Method 2 (Disease ID) completed successfully!")
print(f"   Output shape: {res_disease['cropped_image'].shape}, BBox: {res_disease['bbox']}")

# Test Plant Identification Model
print("\n3. Testing Plant_Identification module with integrated Preprocessing Method 1...")
plant_id_dir = CORETECH_DIR / "Plant_Identification"
if str(plant_id_dir) not in sys.path:
    sys.path.insert(0, str(plant_id_dir))

from Plant_Identification.src.predict import predict_image
id_results = predict_image(img_pil, top_k=3)
print("   Plant_Identification predict_image executed successfully:")
for r in id_results:
    print(f"    - Rank {r['rank']}: {r['species']} ({r['confidence']}%)")

# Test Plant Disease Model
print("\n4. Testing Plant_Disease module with integrated Preprocessing Method 2 & Spot Detection...")
plant_disease_dir = CORETECH_DIR / "Plant_Disease"
if str(plant_disease_dir) not in sys.path:
    sys.path.insert(0, str(plant_disease_dir))

from Plant_Disease.src.predict import diagnose_image
disease_result = diagnose_image(img_pil, target_species="auto", top_k=3)
print("   Plant_Disease diagnose_image executed successfully:")
print(f"    - Identified: {disease_result['species']['name']} ({disease_result['species']['confidence']}%)")
print(f"    - Disease Available: {disease_result['disease_available']}")
if disease_result['disease_available']:
    print(f"    - Pathology Diagnosis: {disease_result['diagnosis']['condition_name']} ({disease_result['diagnosis']['confidence']}%)")
    print(f"    - Spot Detection: {disease_result['spots']['status_message']}")
else:
    print(f"    - Message: {disease_result['status_message']}")

print("\n5. Testing Branch B (Supported Plant Variety: Money_Plant) -> Full Pathology & Spot Detection...")
disease_result_b = diagnose_image(img_pil, target_species="Money_Plant", top_k=3)
print("   Plant_Disease Branch B executed successfully:")
print(f"    - Disease Available: {disease_result_b['disease_available']}")
print(f"    - Pathology Diagnosis: {disease_result_b['diagnosis']['condition_name']} ({disease_result_b['diagnosis']['confidence']}%)")
print(f"    - Spots Count: {disease_result_b['spots']['spots_count']}")
print(f"    - Infected Area: {disease_result_b['spots']['infected_area_pct']}%")
print(f"    - Spot Detection Status: {disease_result_b['spots']['status_message']}")
print(f"    - Preprocessed Base64 included: {bool(disease_result_b.get('preprocessed_image_base64'))}")

print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
