"""
Reliability and Clinical Pathology Verification Suite
Tests:
1. Biological Family Exclusions (Dicot vs Monocot parasites, farina/bloom, etc.)
2. Abiotic Stress Analysis (Underwatering/Wrinkle, Overwatering/Edema, Sunburn/Bleach)
3. Surface Pathology Arbitration & Guardrails
4. Disease Knowledge Base completeness and structure
"""
import sys
import os
import json
import unittest
import numpy as np
from PIL import Image

# Setup paths
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(TEST_DIR)
CORETECH_DIR = os.path.dirname(EXP_DIR)
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, CORETECH_DIR)

from src.predict import get_biological_exclusions
from src.abiotic_analyzer import analyze_abiotic_stress
from CoreTech_v2.src.surface_pathology import diagnose_surface_condition

class TestReliabilityAndPathology(unittest.TestCase):

    def test_biological_family_exclusions(self):
        """Verify biological exclusions accurately filter obligate parasites based on plant morphology."""
        # 1. Cacti and Succulents (Cactaceae / Crassulaceae / Haworthia / Sansevieria)
        succulent_excl = [x.lower() for x in get_biological_exclusions("Echeveria elegans")]
        self.assertTrue(any("leaf_miners" in x for x in succulent_excl))
        self.assertTrue(any("powdery_mildew" in x for x in succulent_excl))
        self.assertTrue(any("downy_mildew" in x for x in succulent_excl))

        cactus_excl = [x.lower() for x in get_biological_exclusions("Opuntia microdasys")]
        self.assertTrue(any("leaf_miners" in x for x in cactus_excl))
        self.assertTrue(any("powdery_mildew" in x for x in cactus_excl))

        # 2. Araceae (Monstera, Philodendron, Epipremnum, Anthurium, Zamioculcas)
        aroid_excl = [x.lower() for x in get_biological_exclusions("Monstera deliciosa")]
        self.assertTrue(any("leaf_miners" in x for x in aroid_excl))
        self.assertTrue(any("powdery_mildew" in x for x in aroid_excl))

        # 3. Orchidaceae
        orchid_excl = [x.lower() for x in get_biological_exclusions("Phalaenopsis aphrodite")]
        self.assertTrue(any("leaf_miners" in x for x in orchid_excl))
        self.assertTrue(any("powdery_mildew" in x for x in orchid_excl))

        # 4. Arecaceae / Palms
        palm_excl = [x.lower() for x in get_biological_exclusions("Dypsis lutescens (Areca Palm)")]
        self.assertTrue(any("leaf_miners" in x for x in palm_excl))
        self.assertTrue(any("powdery_mildew" in x for x in palm_excl))

        # 5. Ferns / Pteridophytes
        fern_excl = [x.lower() for x in get_biological_exclusions("Nephrolepis exaltata (Boston Fern)")]
        self.assertTrue(any("leaf_miners" in x for x in fern_excl))
        self.assertTrue(any("powdery_mildew" in x for x in fern_excl))
        self.assertTrue(any("rust" in x for x in fern_excl))

    def test_abiotic_dehydration_analyzer(self):
        """Verify that wrinkled, curled, brown-tipped leaves trigger dehydration."""
        # Create synthetic wrinkled brown-bordered image
        arr = np.full((300, 300, 3), (70, 110, 50), dtype=np.uint8)
        # Add high-frequency edge texture (wrinkles)
        for i in range(0, 300, 4):
            arr[i, :, :] = (120, 80, 40)
        # Add crispy brown papery margins
        arr[:40, :, :] = (140, 80, 30)
        arr[-40:, :, :] = (140, 80, 30)

        img = Image.fromarray(arr)
        res = analyze_abiotic_stress(img, spots_count=0)

        self.assertIn("is_underwatered", res)
        self.assertIn("surface_wrinkle_index", res)
        self.assertIn("crispness_pct", res)
        self.assertTrue(res["is_underwatered"])
        self.assertEqual(res["stress_type"], "Underwatering Dehydration")

    def test_abiotic_overwatering_analyzer(self):
        """Verify that water-soaked, translucent chlorotic tissue triggers overwatering."""
        # Create synthetic pale yellowish translucent image
        arr = np.full((300, 300, 3), (160, 190, 80), dtype=np.uint8)
        # Add water-soaked translucent patches
        arr[50:200, 50:200, :] = (170, 200, 95)

        img = Image.fromarray(arr)
        res = analyze_abiotic_stress(img, spots_count=0)

        self.assertIn("is_overwatered", res)
        self.assertIn("severe_overwatering", res)
        self.assertTrue(res["is_overwatered"])
        self.assertEqual(res["stress_type"], "Overwatering / Waterlogging")

    def test_abiotic_sunburn_analyzer(self):
        """Verify that bleached, high-brightness surface patches trigger sunburn."""
        # Create normal green leaf image with large bleached center
        arr = np.full((300, 300, 3), (60, 140, 50), dtype=np.uint8)
        # Add bleached tan/white patch
        arr[80:200, 80:200, :] = (235, 230, 210)

        img = Image.fromarray(arr)
        res = analyze_abiotic_stress(img, spots_count=0)

        self.assertIn("is_sunburned", res)
        self.assertIn("bleached_pct", res)
        self.assertTrue(res["is_sunburned"])
        self.assertEqual(res["stress_type"], "Solar Scorch / Sunburn")

    def test_surface_pathology_arbitration(self):
        """Verify surface pathology fallback arbitration and overwatering/edema override."""
        # Synthetic green image and full mask
        arr = np.full((224, 224, 3), (50, 150, 60), dtype=np.uint8)
        img = Image.fromarray(arr)
        mask = np.ones((224, 224), dtype=np.uint8)

        diag = diagnose_surface_condition(img, mask, plant_family="Monstera deliciosa")
        self.assertIn("primary_class", diag)
        self.assertIn("confidence", diag)
        self.assertIn("surface_stress", diag)

        # For Monstera, leaf miners must NEVER be returned due to biological exclusion
        self.assertNotEqual(diag["primary_class"], "Leaf_Miners")
        self.assertNotEqual(diag["primary_class"], "Powdery_Mildew")

    def test_knowledge_base_integrity(self):
        """Verify that all new abiotic conditions are properly documented in disease_knowledge_base.json."""
        kb_path = os.path.join(EXP_DIR, "artifacts", "disease_knowledge_base.json")
        self.assertTrue(os.path.exists(kb_path))
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)

        required_keys = [
            "Abiotic_Dehydration_Underwatering",
            "Abiotic_Overwatering_Edema",
            "Abiotic_Sunburn_Scald",
            "Abiotic_Nutrient_Chlorosis"
        ]

        for k in required_keys:
            self.assertIn(k, kb, f"Missing {k} in disease knowledge base!")
            entry = kb[k]
            self.assertIn("condition_name", entry)
            self.assertIn("summary", entry)
            self.assertIn("immediate_actions", entry)
            self.assertIn("treatment", entry)
            self.assertIn("prevention", entry)
            self.assertIn("symptoms", entry)
            self.assertIn("causes", entry)
            self.assertIsInstance(entry["immediate_actions"], list)
            self.assertTrue(len(entry["immediate_actions"]) >= 3)

if __name__ == "__main__":
    unittest.main()
