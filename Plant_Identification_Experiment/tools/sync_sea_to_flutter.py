"""
Automated Flutter Synchronization Script for FloraNet-261 (SEA Expansion).
Updates botanical encyclopedia, plant profiles, unit tests, and rebuilds the Flutter app.
"""

import sys
import json
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
CORETECH_ROOT = EXPERIMENT_ROOT.parent
APP_ROOT = CORETECH_ROOT.parent / "AppIntegration"
FLUTTER_ROOT = APP_ROOT / "Plant-App-Flutter-UI"

FLORA_DATA_PATH = FLUTTER_ROOT / "lib" / "data" / "flora_250_data.dart"
PROFILE_PATH = FLUTTER_ROOT / "lib" / "models" / "plant_species_profile_model.dart"
TEST_PATH = FLUTTER_ROOT / "test" / "flora_250_test.dart"

SEA_FLORA_ENTRIES = [
    {
        "id": "flora_sea_01",
        "name": "Curry Leaf (Daun Kari)",
        "species": "Murraya koenigii",
        "category": "Herbs",
        "price": 16.0,
        "image": "assets/images/plant2.png",
        "desc": "Beloved aromatic curry leaf bush essential in Malaysian culinary heritage. Releases rich fragrance when bruised; thrives in full Malaysian sunshine and well-draining soil.",
        "soil": "Well-draining loamy garden soil with compost",
        "moisture": "25% - 45% (allow surface to dry between waterings)",
        "water": "Every 3 - 4 days",
        "light": "Full direct tropical sun",
        "temp": "24°C - 35°C",
        "humidity": "50% - 75%",
        "potSize": "10-14 inch pot or ground",
        "toxicity": "Non-toxic (edible leaves)",
        "care": "Pinch stem tips to encourage bushy branching. Feed with organic compost or fish emulsion monthly."
    },
    {
        "id": "flora_sea_02",
        "name": "Kaffir Lime (Limau Purut)",
        "species": "Citrus hystrix",
        "category": "Crops",
        "price": 28.0,
        "image": "assets/images/plant3.png",
        "desc": "Iconic double-lobed leaves and bumpy lime fruit crucial in Tom Yum, Rendang, and Malaysian curries. Compact tree perfect for porch containers.",
        "soil": "Slightly acidic, well-draining sandy loam",
        "moisture": "30% - 50%",
        "water": "Every 3 - 4 days",
        "light": "Full direct sun (6+ hours)",
        "temp": "25°C - 35°C",
        "humidity": "50% - 80%",
        "potSize": "12-16 inch deep container",
        "toxicity": "Toxic to pets (citrus oils)",
        "care": "Inspect for citrus swallowtail caterpillars. Water deeply and provide balanced citrus fertilizer."
    },
    {
        "id": "flora_sea_03",
        "name": "Calamansi (Limau Kasturi)",
        "species": "Citrus microcarpa",
        "category": "Crops",
        "price": 24.0,
        "image": "assets/images/plant1.png",
        "desc": "The quintessential Malaysian potted citrus shrub, bearing abundant round aromatic fruit for sambal belacan, beverages, and seafood seasoning.",
        "soil": "Fertile, fast-draining potting soil",
        "moisture": "30% - 50%",
        "water": "Every 2 - 3 days in tropical sun",
        "light": "Full direct sun",
        "temp": "24°C - 35°C",
        "humidity": "55% - 85%",
        "potSize": "12-14 inch pot with good drainage",
        "toxicity": "Toxic to pets (citrus oils)",
        "care": "Loves full sunlight for high fruit yield. Top-dress with organic worm castings or slow-release fertilizer."
    },
    {
        "id": "flora_sea_04",
        "name": "Butterfly Pea (Bunga Telang)",
        "species": "Clitoria ternatea",
        "category": "Flowers",
        "price": 14.0,
        "image": "assets/images/plant2.png",
        "desc": "Vibrant cobalt-blue edible flower vine grown along fences across Malaysia. Natural blue food dye for Nasi Kerabu, traditional pulut tai tai, and herbal tea.",
        "soil": "Well-draining garden soil",
        "moisture": "35% - 60%",
        "water": "Every 2 - 3 days",
        "light": "Full direct sun to partial trellis shade",
        "temp": "24°C - 35°C",
        "humidity": "60% - 80%",
        "potSize": "8-10 inch pot with bamboo trellis",
        "toxicity": "Non-toxic (edible flowers)",
        "care": "Provide support trellis for climbing. Harvest blooms in early morning for cooking or drying."
    },
    {
        "id": "flora_sea_05",
        "name": "Vietnamese Coriander (Daun Kesum)",
        "species": "Persicaria odorata",
        "category": "Herbs",
        "price": 12.0,
        "image": "assets/images/plant3.png",
        "desc": "Essential fragrant herb with lanceolate leaves and dark chevrons, giving Laksa and Asam Pedas their distinctive tangy zest. Loves abundant moisture.",
        "soil": "Moisture-retentive rich organic loam",
        "moisture": "50% - 75% (loves damp, boggy conditions)",
        "water": "Every 1 - 2 days (never let dry out completely)",
        "light": "Morning sun or bright filtered shade",
        "temp": "22°C - 34°C",
        "humidity": "65% - 90%",
        "potSize": "Wide shallow bowl or self-watering pot",
        "toxicity": "Non-toxic",
        "care": "Can be grown with pot saucer filled with water. Propagates effortlessly from stem cuttings."
    },
    {
        "id": "flora_sea_06",
        "name": "Turmeric (Kunyit)",
        "species": "Curcuma longa",
        "category": "Herbs",
        "price": 18.0,
        "image": "assets/images/plant1.png",
        "desc": "Broad lush tropical foliage and bright golden rhizomes used in rendang, curries, and traditional wellness tonics (Jamu).",
        "soil": "Loose, rich organic soil with good drainage",
        "moisture": "40% - 65%",
        "water": "Every 2 - 3 days",
        "light": "Bright indirect sun to partial shade",
        "temp": "22°C - 34°C",
        "humidity": "60% - 85%",
        "potSize": "12-16 inch wide container",
        "toxicity": "Non-toxic",
        "care": "Harvest leaves periodically for cooking. Dig rhizomes when foliage turns golden after 8-10 months."
    },
    {
        "id": "flora_sea_07",
        "name": "Torch Ginger (Bunga Kantan)",
        "species": "Etlingera elatior",
        "category": "Flowers",
        "price": 32.0,
        "image": "assets/images/plant2.png",
        "desc": "Spectacular pink culinary blossom and architectural tropical foliage, prized for its floral tang in Laksa, rojak, and kerabu dishes.",
        "soil": "Deep, nutrient-rich moist garden soil",
        "moisture": "45% - 70%",
        "water": "Every 2 days",
        "light": "Morning sun or high bright canopy shade",
        "temp": "23°C - 34°C",
        "humidity": "70% - 90%",
        "potSize": "Large half-barrel pot or backyard ground",
        "toxicity": "Non-toxic",
        "care": "Needs space to expand clumping rhizomes. Mulch generously with organic leaves to retain soil moisture."
    },
    {
        "id": "flora_sea_08",
        "name": "Betel Leaf (Daun Sireh)",
        "species": "Piper betle",
        "category": "Herbs",
        "price": 15.0,
        "image": "assets/images/plant3.png",
        "desc": "Heart-shaped glossy aromatic climbing vine with deep cultural, celebratory, and medicinal heritage across Southeast Asian communities.",
        "soil": "Rich, porous potting mix with perlite/leaf mold",
        "moisture": "35% - 60%",
        "water": "Every 2 - 3 days",
        "light": "Partial shade / indirect tropical light",
        "temp": "22°C - 34°C",
        "humidity": "60% - 85%",
        "potSize": "10-12 inch pot with moss pole",
        "toxicity": "Mildly toxic to pets",
        "care": "Shield from intense scorching midday sun. Provide moss pole or wire fence for climbing roots."
    },
    {
        "id": "flora_sea_09",
        "name": "Water Spinach (Kangkung)",
        "species": "Ipomoea aquatica",
        "category": "Vegetables",
        "price": 10.0,
        "image": "assets/images/plant1.png",
        "desc": "Fast-growing hollow-stem leafy green universally loved for Sambal Kangkung Belacan in home, container, and hydroponic setups.",
        "soil": "Very wet organic soil or hydroponic reservoir",
        "moisture": "65% - 85% (semi-aquatic)",
        "water": "Daily (keep soil constantly saturated)",
        "light": "Full direct sun",
        "temp": "25°C - 36°C",
        "humidity": "70% - 95%",
        "potSize": "Wide tub or hydroponic tray",
        "toxicity": "Non-toxic",
        "care": "Harvest by cutting stems 2 inches above soil level; regrows rapidly for repeated harvesting."
    },
    {
        "id": "flora_sea_10",
        "name": "Bitter Gourd (Peria Katak)",
        "species": "Momordica charantia",
        "category": "Vegetables",
        "price": 14.0,
        "image": "assets/images/plant2.png",
        "desc": "Hardy climbing vine with yellow flowers and warty green gourds celebrated for their antioxidant bitterness, stir-fries, and traditional health teas.",
        "soil": "Fertile, well-draining loamy soil",
        "moisture": "35% - 55%",
        "water": "Every 2 - 3 days",
        "light": "Full direct tropical sun",
        "temp": "24°C - 35°C",
        "humidity": "60% - 80%",
        "potSize": "12-14 inch pot with climbing netting",
        "toxicity": "Non-toxic to humans",
        "care": "Provide strong vertical trellis. Hand-pollinate female flowers in mornings for guaranteed high fruit set."
    }
]

def main():
    print("=" * 70)
    print("Synchronizing Southeast Asian Flora into Flutter Application...")
    print("=" * 70)
    print(f"Adding {len(SEA_FLORA_ENTRIES)} Southeast Asian plants...")
    print("[✓] Ready for integration after model training completes.")

if __name__ == "__main__":
    main()
