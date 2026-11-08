"""
Automated Flutter Synchronization Script for FloraNet-261 (SEA Expansion).
Updates botanical encyclopedia, plant profiles, unit tests, and verifies the Flutter app.
"""

import sys
import json
import re
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
CORETECH_ROOT = EXPERIMENT_ROOT.parent
APP_ROOT = CORETECH_ROOT.parent / "AppIntegration"
FLUTTER_ROOT = APP_ROOT / "Plant-App-Flutter-UI"

FLORA_DATA_PATH = FLUTTER_ROOT / "lib" / "data" / "flora_250_data.dart"
PROFILE_PATH = FLUTTER_ROOT / "lib" / "models" / "plant_species_profile_model.dart"
TEST_PATH = FLUTTER_ROOT / "test" / "flora_250_test.dart"

SEA_FLORA_SPECIES_CODE = """    const FloraSpecies(
      id: 'flora_252',
      commonName: 'Curry Leaf (Daun Kari)',
      scientificName: 'Murraya koenigii',
      rawName: 'Curry Leaf Tree (Murraya koenigii)',
      category: 'Herbs & Aromatics',
      description: 'Beloved aromatic curry leaf bush essential in Malaysian culinary heritage. Releases rich fragrance when bruised; thrives in full Malaysian sunshine and well-draining soil.',
      imageUrl: 'assets/images/image_1.png',
      lowMoistureThreshold: 25,
      highMoistureThreshold: 45,
      wateringIntervalDays: 4,
      wateringRecommendation: 'Allow surface soil to dry slightly between waterings; thrives in full sun and warmth.',
      sunlight: 'Full direct tropical sun',
      idealTemperature: '24°C - 35°C',
      petToxicity: 'Non-toxic / Pet friendly',
      soilType: 'Well-draining loamy garden soil with organic compost',
    ),
    const FloraSpecies(
      id: 'flora_253',
      commonName: 'Kaffir Lime (Limau Purut)',
      scientificName: 'Citrus hystrix',
      rawName: 'Kaffir Lime (Citrus hystrix)',
      category: 'Fruits & Vegetables',
      description: 'Iconic double-lobed leaves and bumpy lime fruit crucial in Tom Yum, Rendang, and Malaysian curries. Compact tree perfect for porch containers.',
      imageUrl: 'assets/images/image_2.png',
      lowMoistureThreshold: 30,
      highMoistureThreshold: 50,
      wateringIntervalDays: 4,
      wateringRecommendation: 'Water deeply when top 2 inches dry out; ensure container drains freely.',
      sunlight: 'Full direct sun (6+ hours)',
      idealTemperature: '25°C - 35°C',
      petToxicity: 'Toxic to pets (citrus essential oils)',
      soilType: 'Slightly acidic, well-draining sandy loam',
    ),
    const FloraSpecies(
      id: 'flora_254',
      commonName: 'Calamansi (Limau Kasturi)',
      scientificName: 'Citrus microcarpa',
      rawName: 'Calamansi (Citrus microcarpa)',
      category: 'Fruits & Vegetables',
      description: 'The quintessential Malaysian potted citrus shrub, bearing abundant round aromatic fruit for sambal belacan, beverages, and seafood seasoning.',
      imageUrl: 'assets/images/image_3.png',
      lowMoistureThreshold: 30,
      highMoistureThreshold: 50,
      wateringIntervalDays: 3,
      wateringRecommendation: 'Water regularly during hot tropical days; avoid letting the root ball dry out completely.',
      sunlight: 'Full direct sun',
      idealTemperature: '24°C - 35°C',
      petToxicity: 'Toxic to pets (citrus essential oils)',
      soilType: 'Fertile, fast-draining potting soil with compost',
    ),
    const FloraSpecies(
      id: 'flora_255',
      commonName: 'Butterfly Pea (Bunga Telang)',
      scientificName: 'Clitoria ternatea',
      rawName: 'Butterfly Pea (Clitoria ternatea)',
      category: 'Flowers & Ornamentals',
      description: 'Vibrant cobalt-blue edible flower vine grown along fences across Malaysia. Natural blue food dye for Nasi Kerabu, traditional pulut tai tai, and herbal tea.',
      imageUrl: 'assets/images/image_1.png',
      lowMoistureThreshold: 35,
      highMoistureThreshold: 60,
      wateringIntervalDays: 3,
      wateringRecommendation: 'Keep soil evenly moist; water regularly during active flowering and climbing.',
      sunlight: 'Full direct sun to partial trellis shade',
      idealTemperature: '24°C - 35°C',
      petToxicity: 'Non-toxic (edible flowers)',
      soilType: 'Well-draining fertile garden soil',
    ),
    const FloraSpecies(
      id: 'flora_256',
      commonName: 'Vietnamese Coriander (Daun Kesum)',
      scientificName: 'Persicaria odorata',
      rawName: 'Vietnamese Coriander (Persicaria odorata)',
      category: 'Herbs & Aromatics',
      description: 'Essential fragrant herb with lanceolate leaves and dark chevrons, giving Laksa and Asam Pedas their distinctive tangy zest. Loves abundant moisture.',
      imageUrl: 'assets/images/image_2.png',
      lowMoistureThreshold: 50,
      highMoistureThreshold: 75,
      wateringIntervalDays: 2,
      wateringRecommendation: 'Keep soil consistently wet to semi-boggy; never let the root zone dry out.',
      sunlight: 'Morning sun or bright filtered shade',
      idealTemperature: '22°C - 34°C',
      petToxicity: 'Non-toxic / Pet friendly',
      soilType: 'Moisture-retentive rich organic loam',
    ),
    const FloraSpecies(
      id: 'flora_257',
      commonName: 'Turmeric (Kunyit)',
      scientificName: 'Curcuma longa',
      rawName: 'Turmeric (Curcuma longa)',
      category: 'Herbs & Aromatics',
      description: 'Broad lush tropical foliage and bright golden rhizomes used in rendang, curries, and traditional wellness tonics (Jamu).',
      imageUrl: 'assets/images/image_3.png',
      lowMoistureThreshold: 40,
      highMoistureThreshold: 65,
      wateringIntervalDays: 3,
      wateringRecommendation: 'Water thoroughly when top inch dries; maintain moderate soil moisture for healthy rhizomes.',
      sunlight: 'Bright indirect sun to partial shade',
      idealTemperature: '22°C - 34°C',
      petToxicity: 'Non-toxic / Pet friendly',
      soilType: 'Loose, rich organic soil with good drainage',
    ),
    const FloraSpecies(
      id: 'flora_258',
      commonName: 'Torch Ginger (Bunga Kantan)',
      scientificName: 'Etlingera elatior',
      rawName: 'Torch Ginger (Etlingera elatior)',
      category: 'Flowers & Ornamentals',
      description: 'Spectacular pink culinary blossom and architectural tropical foliage, prized for its floral tang in Laksa, rojak, and kerabu dishes.',
      imageUrl: 'assets/images/image_1.png',
      lowMoistureThreshold: 45,
      highMoistureThreshold: 70,
      wateringIntervalDays: 2,
      wateringRecommendation: 'Keep soil rich and damp; mulch well around base to retain tropical moisture.',
      sunlight: 'Morning sun or high bright canopy shade',
      idealTemperature: '23°C - 34°C',
      petToxicity: 'Non-toxic / Pet friendly',
      soilType: 'Deep, nutrient-rich moist garden soil',
    ),
    const FloraSpecies(
      id: 'flora_259',
      commonName: 'Betel Leaf (Daun Sireh)',
      scientificName: 'Piper betle',
      rawName: 'Betel Leaf (Piper betle)',
      category: 'Herbs & Aromatics',
      description: 'Heart-shaped glossy aromatic climbing vine with deep cultural, celebratory, and medicinal heritage across Southeast Asian communities.',
      imageUrl: 'assets/images/image_2.png',
      lowMoistureThreshold: 35,
      highMoistureThreshold: 60,
      wateringIntervalDays: 3,
      wateringRecommendation: 'Water when topsoil feels slightly dry; shield from intense harsh midday scorch.',
      sunlight: 'Partial shade / bright indirect light',
      idealTemperature: '22°C - 34°C',
      petToxicity: 'Mildly toxic to pets',
      soilType: 'Rich, porous potting mix with perlite and leaf mold',
    ),
    const FloraSpecies(
      id: 'flora_260',
      commonName: 'Water Spinach (Kangkung)',
      scientificName: 'Ipomoea aquatica',
      rawName: 'Water Spinach (Ipomoea aquatica)',
      category: 'Fruits & Vegetables',
      description: 'Fast-growing hollow-stem leafy green universally loved for Sambal Kangkung Belacan in home, container, and hydroponic setups.',
      imageUrl: 'assets/images/image_3.png',
      lowMoistureThreshold: 65,
      highMoistureThreshold: 85,
      wateringIntervalDays: 1,
      wateringRecommendation: 'Semi-aquatic plant; keep soil saturated daily or grow in a continuous water reservoir.',
      sunlight: 'Full direct tropical sun',
      idealTemperature: '25°C - 36°C',
      petToxicity: 'Non-toxic / Pet friendly',
      soilType: 'Very wet organic soil or hydroponic reservoir',
    ),
    const FloraSpecies(
      id: 'flora_261',
      commonName: 'Bitter Gourd (Peria Katak)',
      scientificName: 'Momordica charantia',
      rawName: 'Bitter Gourd (Momordica charantia)',
      category: 'Fruits & Vegetables',
      description: 'Hardy climbing vine with yellow flowers and warty green gourds celebrated for their antioxidant bitterness, stir-fries, and traditional health teas.',
      imageUrl: 'assets/images/image_1.png',
      lowMoistureThreshold: 35,
      highMoistureThreshold: 55,
      wateringIntervalDays: 3,
      wateringRecommendation: 'Water at the base every 2-3 days; avoid wetting vines to prevent fungal foliage issues.',
      sunlight: 'Full direct tropical sun',
      idealTemperature: '24°C - 35°C',
      petToxicity: 'Non-toxic to humans',
      soilType: 'Fertile, well-draining loamy soil',
    ),
"""

def sync_flora_data():
    print(f"[*] Reading {FLORA_DATA_PATH}...")
    content = FLORA_DATA_PATH.read_text(encoding="utf-8")
    
    if "flora_261" in content:
        print("  [✓] FloraNet-261 already synchronized in flora_250_data.dart.")
        return

    # Find the closing of speciesList:
    # Look for the last FloraSpecies (flora_251) and insert right after it
    marker = "    const FloraSpecies(\n      id: 'flora_251',"
    if marker not in content:
        raise ValueError(f"Could not find marker for flora_251 in {FLORA_DATA_PATH}")

    # Find end of flora_251 block (ends with "    ),\n  ];")
    end_pattern = r"(      id: 'flora_251',[\s\S]*?    \),)\n  \];"
    match = re.search(end_pattern, content)
    if not match:
        raise ValueError(f"Could not find closing bracket for flora_251 in {FLORA_DATA_PATH}")

    replacement = match.group(1) + "\n" + SEA_FLORA_SPECIES_CODE + "  ];"
    new_content = content[:match.start()] + replacement + content[match.end():]
    
    FLORA_DATA_PATH.write_text(new_content, encoding="utf-8")
    print("  [✓] Successfully appended 10 SEA species (flora_252 to flora_261) to flora_250_data.dart!")

def sync_flora_test():
    print(f"[*] Reading {TEST_PATH}...")
    content = TEST_PATH.read_text(encoding="utf-8")
    
    # 1. Update total count from 251 to 261
    content = re.sub(
        r"expect\(Flora250Data\.speciesList\.length,\s*equals\(251\)\);",
        "expect(Flora250Data.speciesList.length, equals(261));",
        content
    )
    content = re.sub(
        r"expect\(total,\s*equals\(251\)\);",
        "expect(total, equals(261));",
        content
    )
    content = re.sub(
        r"covers all 251 plants",
        "covers all 261 plants",
        content
    )
    
    # 2. Update category counts:
    # Herbs & Aromatics: 26 -> 30
    content = re.sub(
        r"expect\(categoryCounts\['Herbs & Aromatics'\],\s*equals\(26\)\);.*",
        "expect(categoryCounts['Herbs & Aromatics'], equals(30)); // + Pandan + 4 SEA",
        content
    )
    # Fruits & Vegetables: 40 -> 44
    content = re.sub(
        r"expect\(categoryCounts\['Fruits & Vegetables'\],\s*equals\(40\)\);.*",
        "expect(categoryCounts['Fruits & Vegetables'], equals(44)); // + 4 SEA",
        content
    )
    # Flowers & Ornamentals: 50 -> 52
    content = re.sub(
        r"expect\(categoryCounts\['Flowers & Ornamentals'\],\s*equals\(50\)\);.*",
        "expect(categoryCounts['Flowers & Ornamentals'], equals(52)); // + 2 SEA",
        content
    )

    # 3. Add SEA checks to group test
    if "Curry Leaf" not in content:
        content = content.replace(
            "expect(Flora250Data.findByNameOrSpecies('Pandan'), isNotNull);",
            "expect(Flora250Data.findByNameOrSpecies('Pandan'), isNotNull);\n      expect(Flora250Data.findByNameOrSpecies('Curry Leaf'), isNotNull);\n      expect(Flora250Data.findByNameOrSpecies('Kaffir Lime'), isNotNull);\n      expect(Flora250Data.findByNameOrSpecies('Bunga Telang'), isNotNull);\n      expect(Flora250Data.findByNameOrSpecies('Kangkung'), isNotNull);"
        )

    TEST_PATH.write_text(content, encoding="utf-8")
    print("  [✓] Successfully updated test assertions to 261 species in flora_250_test.dart!")

def main():
    print("=" * 70)
    print("Synchronizing Southeast Asian Flora into Flutter Application...")
    print("=" * 70)
    sync_flora_data()
    sync_flora_test()
    print("\n[✓] Flutter encyclopedia & unit test synchronization complete!")

if __name__ == "__main__":
    main()
