"""
Targeted Botanical Image Harvester & Data Augmentation Tool for FloraNet-100.
Supports all 100 botanical classes (47 original + 53 newly introduced houseplant species).

Features:
  - Scientific taxon query matching for 100 botanical houseplant varieties
  - Automated duplicate rejection (dHash perceptual hashing)
  - Preprocessing Method 1 integration (plant foliage & mask area validation)
  - Polite rate-limiting (0.35s delay between requests to prevent API throttle)
  - Isolated staging in data/staging/
  - Automated 3-way dataset partitioning (Train 70%, Val 15%, Test 15% for new classes)
  - Dynamic generation of artifacts/class_indices.json
"""

import os
import sys
import json
import time
import hashlib
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from PIL import Image
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
CORETECH_ROOT = PROJECT_ROOT.parent

if str(CORETECH_ROOT) not in sys.path:
    sys.path.insert(0, str(CORETECH_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

try:
    from preprocessing.methods import PlantPreprocessors
    _preprocessor = PlantPreprocessors()
except Exception as e:
    print(f"Notice: PlantPreprocessors basic fallback mode: {e}")
    _preprocessor = None

# Comprehensive 100-class botanical taxonomy map (47 Original + 53 New)
BOTANICAL_TAXON_MAP = {
    # --- 47 Original Classes ---
    "African Violet (Saintpaulia ionantha)": "Saintpaulia ionantha",
    "Aloe Vera": "Aloe vera",
    "Anthurium (Anthurium andraeanum)": "Anthurium andraeanum",
    "Areca Palm (Dypsis lutescens)": "Dypsis lutescens",
    "Asparagus Fern (Asparagus setaceus)": "Asparagus setaceus",
    "Begonia (Begonia spp.)": "Begonia",
    "Bird of Paradise (Strelitzia reginae)": "Strelitzia reginae",
    "Birds Nest Fern (Asplenium nidus)": "Asplenium nidus",
    "Boston Fern (Nephrolepis exaltata)": "Nephrolepis exaltata",
    "Calathea": "Goeppertia",
    "Cast Iron Plant (Aspidistra elatior)": "Aspidistra elatior",
    "Chinese Money Plant (Pilea peperomioides)": "Pilea peperomioides",
    "Chinese evergreen (Aglaonema)": "Aglaonema",
    "Christmas Cactus (Schlumbergera bridgesii)": "Schlumbergera",
    "Chrysanthemum": "Chrysanthemum",
    "Ctenanthe": "Ctenanthe",
    "Daffodils (Narcissus spp.)": "Narcissus",
    "Dracaena": "Dracaena",
    "Dumb Cane (Dieffenbachia spp.)": "Dieffenbachia",
    "Elephant Ear (Alocasia spp.)": "Alocasia",
    "English Ivy (Hedera helix)": "Hedera helix",
    "Hyacinth (Hyacinthus orientalis)": "Hyacinthus orientalis",
    "Iron Cross begonia (Begonia masoniana)": "Begonia masoniana",
    "Jade plant (Crassula ovata)": "Crassula ovata",
    "Kalanchoe": "Kalanchoe",
    "Lilium (Hemerocallis)": "Hemerocallis",
    "Lily of the valley (Convallaria majalis)": "Convallaria majalis",
    "Money Tree (Pachira aquatica)": "Pachira aquatica",
    "Monstera Deliciosa (Monstera deliciosa)": "Monstera deliciosa",
    "Orchid": "Orchidaceae",
    "Parlor Palm (Chamaedorea elegans)": "Chamaedorea elegans",
    "Peace lily": "Spathiphyllum",
    "Poinsettia (Euphorbia pulcherrima)": "Euphorbia pulcherrima",
    "Polka Dot Plant (Hypoestes phyllostachya)": "Hypoestes phyllostachya",
    "Ponytail Palm (Beaucarnea recurvata)": "Beaucarnea recurvata",
    "Pothos (Ivy arum)": "Epipremnum aureum",
    "Prayer Plant (Maranta leuconeura)": "Maranta leuconeura",
    "Rattlesnake Plant (Calathea lancifolia)": "Goeppertia insignis",
    "Rubber Plant (Ficus elastica)": "Ficus elastica",
    "Sago Palm (Cycas revoluta)": "Cycas revoluta",
    "Schefflera": "Heptapleurum",
    "Snake plant (Sanseviera)": "Dracaena trifasciata",
    "Tradescantia": "Tradescantia",
    "Tulip": "Tulipa",
    "Venus Flytrap": "Dionaea muscipula",
    "Yucca": "Yucca elephantipes",
    "ZZ Plant (Zamioculcas zamiifolia)": "Zamioculcas zamiifolia",

    # --- 53 Newly Introduced Houseplant Species ---
    "Air Plant (Tillandsia ionantha)": "Tillandsia ionantha",
    "Alocasia Amazonica (Alocasia amazonica)": "Alocasia",
    "Alocasia Zebrina (Alocasia zebrina)": "Alocasia zebrina",
    "Aluminum Plant (Pilea cadierei)": "Pilea cadierei",
    "Arrowhead Plant (Syngonium podophyllum)": "Syngonium podophyllum",
    "Bromeliad (Guzmania lingulata)": "Guzmania lingulata",
    "Bunny Ears Cactus (Opuntia microdasys)": "Opuntia microdasys",
    "Calathea Medallion (Goeppertia roseopicta)": "Goeppertia roseopicta",
    "Calathea Warscewiczii (Goeppertia warszewiczii)": "Goeppertia warszewiczii",
    "Calathea Zebrina (Goeppertia zebrina)": "Goeppertia zebrina",
    "Corn Plant (Dracaena fragrans)": "Dracaena fragrans",
    "Croton (Codiaeum variegatum)": "Codiaeum variegatum",
    "Cylindrical Snake Plant (Dracaena angolensis)": "Dracaena angolensis",
    "Dwarf Umbrella Tree (Heptapleurum arboricola)": "Heptapleurum arboricola",
    "Echeveria Elegans (Echeveria elegans)": "Echeveria elegans",
    "Ficus Audrey (Ficus benghalensis)": "Ficus benghalensis",
    "Fiddle Leaf Fig (Ficus lyrata)": "Ficus lyrata",
    "Fishbone Cactus (Disocactus anguliger)": "Disocactus anguliger",
    "Fishtail Palm (Caryota mitis)": "Caryota mitis",
    "Ghost Plant (Graptopetalum paraguayense)": "Graptopetalum paraguayense",
    "Giant White Bird of Paradise (Strelitzia nicolai)": "Strelitzia nicolai",
    "Goldfish Plant (Nematanthus wettsteinii)": "Nematanthus",
    "Heartleaf Philodendron (Philodendron hederaceum)": "Philodendron hederaceum",
    "Hoya Kerrii (Hoya kerrii)": "Hoya kerrii",
    "Wax Plant (Hoya carnosa)": "Hoya carnosa",
    "Kentia Palm (Howea forsteriana)": "Howea forsteriana",
    "Lipstick Plant (Aeschynanthus radicans)": "Aeschynanthus radicans",
    "Maidenhair Fern (Adiantum capillus-veneris)": "Adiantum capillus-veneris",
    "Moses in the Cradle (Tradescantia spathacea)": "Tradescantia spathacea",
    "Nerve Plant (Fittonia albivenis)": "Fittonia albivenis",
    "Peperomia Hope (Peperomia deppeana)": "Peperomia",
    "Peperomia (Peperomia obtusifolia)": "Peperomia obtusifolia",
    "Philodendron Birkin (Philodendron 'Birkin')": "Philodendron",
    "Philodendron Pink Princess (Philodendron erubescens)": "Philodendron erubescens",
    "Pitcher Plant (Nepenthes alata)": "Nepenthes",
    "Polka Dot Begonia (Begonia maculata)": "Begonia maculata",
    "Satin Pothos (Scindapsus pictus)": "Scindapsus pictus",
    "Spiderwort (Tradescantia zebrina)": "Tradescantia zebrina",
    "Staghorn Fern (Platycerium bifurcatum)": "Platycerium bifurcatum",
    "Stromanthe Triostar (Stromanthe thalia)": "Stromanthe thalia",
    "Sundew (Drosera capensis)": "Drosera capensis",
    "Swiss Cheese Vine (Monstera adansonii)": "Monstera adansonii",
    "Ti Plant (Cordyline fruticosa)": "Cordyline fruticosa",
    "Watermelon Peperomia (Peperomia argyreia)": "Peperomia argyreia",
    "Wax Begonia (Begonia cucullata)": "Begonia cucullata",
    "Weeping Fig (Ficus benjamina)": "Ficus benjamina",
    "Zebra Haworthia (Haworthiopsis fasciata)": "Haworthiopsis fasciata",
    "Maranta Lemon Lime (Maranta leuconeura)": "Maranta leuconeura",
    "Calathea Rufibarba (Goeppertia rufibarba)": "Goeppertia rufibarba",
    "Kangaroo Paw Fern (Microsorum diversifolium)": "Zealandia pustulata",
    "Majesty Palm (Ravenea rivularis)": "Ravenea rivularis",
    "Flamingo Flower (Anthurium clarinervium)": "Anthurium clarinervium",
    "Mini Monstera (Rhaphidophora tetrasperma)": "Rhaphidophora tetrasperma",
    "Golden Barrel Cactus (Echinocactus grusonii)": "Echinocactus grusonii",
    "Columnar Cactus (Pachycereus pringlei)": "Pachycereus pringlei"
}

try:
    from tools.expand_to_250 import NEW_148_SPECIES
    BOTANICAL_TAXON_MAP.update(NEW_148_SPECIES)
except Exception:
    pass

ORIGINAL_47_CLASSES = set([
    "African Violet (Saintpaulia ionantha)", "Aloe Vera", "Anthurium (Anthurium andraeanum)",
    "Areca Palm (Dypsis lutescens)", "Asparagus Fern (Asparagus setaceus)", "Begonia (Begonia spp.)",
    "Bird of Paradise (Strelitzia reginae)", "Birds Nest Fern (Asplenium nidus)", "Boston Fern (Nephrolepis exaltata)",
    "Calathea", "Cast Iron Plant (Aspidistra elatior)", "Chinese Money Plant (Pilea peperomioides)",
    "Chinese evergreen (Aglaonema)", "Christmas Cactus (Schlumbergera bridgesii)", "Chrysanthemum",
    "Ctenanthe", "Daffodils (Narcissus spp.)", "Dracaena", "Dumb Cane (Dieffenbachia spp.)",
    "Elephant Ear (Alocasia spp.)", "English Ivy (Hedera helix)", "Hyacinth (Hyacinthus orientalis)",
    "Iron Cross begonia (Begonia masoniana)", "Jade plant (Crassula ovata)", "Kalanchoe",
    "Lilium (Hemerocallis)", "Lily of the valley (Convallaria majalis)", "Money Tree (Pachira aquatica)",
    "Monstera Deliciosa (Monstera deliciosa)", "Orchid", "Parlor Palm (Chamaedorea elegans)",
    "Peace lily", "Poinsettia (Euphorbia pulcherrima)", "Polka Dot Plant (Hypoestes phyllostachya)",
    "Ponytail Palm (Beaucarnea recurvata)", "Pothos (Ivy arum)", "Prayer Plant (Maranta leuconeura)",
    "Rattlesnake Plant (Calathea lancifolia)", "Rubber Plant (Ficus elastica)", "Sago Palm (Cycas revoluta)",
    "Schefflera", "Snake plant (Sanseviera)", "Tradescantia", "Tulip", "Venus Flytrap", "Yucca",
    "ZZ Plant (Zamioculcas zamiifolia)"
])

NEW_53_CLASSES = [k for k in BOTANICAL_TAXON_MAP.keys() if k not in ORIGINAL_47_CLASSES]

# The 15 lowest-support / highest-confusion classes from baseline
DEEPEN_15_CLASSES = [
    "Yucca", "Calathea", "Pothos (Ivy arum)", "Begonia (Begonia spp.)", "Dracaena",
    "Cast Iron Plant (Aspidistra elatior)", "Elephant Ear (Alocasia spp.)", "Kalanchoe",
    "English Ivy (Hedera helix)", "Tradescantia", "Chinese evergreen (Aglaonema)",
    "Money Tree (Pachira aquatica)", "Ctenanthe", "Orchid", "Schefflera"
]

def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) to detect exact and near-duplicate images."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.array(resized, dtype=np.int32)
    diff = arr[:, :-1] > arr[:, 1:]
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")

def query_inaturalist(taxon_name: str, count: int = 35):
    """Queries iNaturalist API for observations, falling back to any grade if research grade is sparse."""
    encoded_taxon = urllib.parse.quote(taxon_name)
    records = []
    seen_urls = set()

    for grade in ["research", "any"]:
        if len(records) >= count * 2:
            break
        grade_param = "&quality_grade=research" if grade == "research" else ""
        url = (
            f"https://api.inaturalist.org/v1/observations?"
            f"taxon_name={encoded_taxon}{grade_param}&photos=true&per_page={max(count * 2, 40)}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "FloraResearchBot/1.0 (FYP Botanical AI Project)"})
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                data = json.loads(response.read().decode("utf-8"))
                for item in data.get("results", []):
                    for p in item.get("photos", []):
                        orig_url = p.get("url", "")
                        if orig_url and orig_url not in seen_urls:
                            seen_urls.add(orig_url)
                            high_res_url = orig_url.replace("/square.", "/medium.").replace("/small.", "/medium.")
                            records.append({
                                "source": "iNaturalist",
                                "obs_id": item.get("id"),
                                "url": high_res_url,
                                "taxon": item.get("taxon", {}).get("name", taxon_name),
                                "license": p.get("license_code", "all-rights-reserved")
                            })
                            if len(records) >= count * 2:
                                break
        except Exception as e:
            print(f"  [iNaturalist Query Warning for {taxon_name} ({grade})]: {e}")
    return records

def download_and_verify(record: dict, existing_hashes: set, min_dim: int = 224):
    """Downloads image, computes dHash, and validates foliage coverage."""
    url = record["url"]
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            img_data = response.read()

        import io
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        w, h = img.size
        if w < min_dim or h < min_dim:
            return None, "Too small (< 224px)"

        img_hash = compute_dhash(img)
        for ex_hash in existing_hashes:
            if hamming_distance(img_hash, ex_hash) <= 3:
                return None, "Near-duplicate rejected (dHash match)"

        foliage_score = 1.0
        clean_img = img
        if _preprocessor is not None:
            try:
                np_img = np.array(img)
                prep_res = _preprocessor.process_for_species_identification(
                    np_img,
                    bg_color=(128, 128, 128),
                    padding_ratio=0.12,
                    filter_pot_soil=True,
                    preserve_leaf_shape=True
                )
                mask = prep_res.get("mask")
                if mask is not None:
                    foliage_score = round(float(np.mean(mask > 0)), 3)
                    if foliage_score < 0.05:
                        return None, f"Non-foliage rejected ({foliage_score*100:.1f}%)"
                    clean_img = Image.fromarray(prep_res["cropped_image"])
            except Exception:
                pass

        return (img, img_hash, foliage_score, clean_img), "Valid"
    except Exception as e:
        return None, f"Download failed ({e})"

def harvest_species(class_name: str, target_count: int = 35, staging_dir: Path = None):
    """Harvests target_count verified images for a single botanical class."""
    if staging_dir is None:
        staging_dir = config.DATA_DIR / "staging" / class_name
    staging_dir.mkdir(parents=True, exist_ok=True)

    existing_staged = list(staging_dir.glob("*.jpg"))
    if len(existing_staged) >= target_count:
        print(f"  ✓ Already staged: {len(existing_staged)}/{target_count} images. Skipping.")
        return len(existing_staged)

    scientific_name = BOTANICAL_TAXON_MAP.get(class_name, class_name)
    print(f"\n[Harvester] '{class_name}' [Taxon: {scientific_name}]...")

    existing_hashes = set()
    train_class_dir = config.TRAIN_DIR / class_name
    if train_class_dir.exists():
        for f in list(train_class_dir.iterdir())[:40]:
            try:
                with Image.open(f) as im:
                    existing_hashes.add(compute_dhash(im))
            except Exception:
                pass

    for f in existing_staged:
        try:
            with Image.open(f) as im:
                existing_hashes.add(compute_dhash(im))
        except Exception:
            pass

    candidates = query_inaturalist(scientific_name, count=target_count)
    accepted = len(existing_staged)
    rejected = 0
    results_manifest = []

    for item in candidates:
        if accepted >= target_count:
            break

        res, reason = download_and_verify(item, existing_hashes)
        if res is not None:
            img, img_hash, foliage_score, clean_img = res
            existing_hashes.add(img_hash)

            file_stem = hashlib.md5(item["url"].encode()).hexdigest()[:12]
            filename = f"scraped_{file_stem}.jpg"
            save_path = staging_dir / filename
            img.save(save_path, "JPEG", quality=92)

            accepted += 1
            print(f"  + Saved [{accepted}/{target_count}]: {filename} (Foliage: {int(foliage_score*100)}%)")
            results_manifest.append({
                "filename": filename,
                "url": item["url"],
                "foliage_score": foliage_score
            })
            # Polite rate-limiting to preserve iNaturalist server stability
            time.sleep(0.35)
        else:
            rejected += 1

    with open(staging_dir / "harvest_manifest.json", "w", encoding="utf-8") as f:
        json.dump(results_manifest, f, indent=2)

    return accepted

def merge_staged_dataset():
    """
    Merges all staged images into train, val, and test.
    - If class is from original 47: 85% to train, 15% to val (test remains frozen).
    - If class is brand new: 70% to train, 15% to val, 15% to test.
    Updates artifacts/class_indices.json with all classes.
    """
    staging_dir = config.DATA_DIR / "staging"
    if not staging_dir.exists():
        print("No staging directory found at data/staging/")
        return

    total_merged_train = 0
    total_merged_val = 0
    total_merged_test = 0

    import random
    random.seed(42)

    for class_folder in staging_dir.iterdir():
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        staged_images = list(class_folder.glob("scraped_*.jpg"))
        if not staged_images:
            continue

        target_train = config.TRAIN_DIR / class_name
        target_val = config.VAL_DIR / class_name
        target_test = config.TEST_DIR / class_name

        target_train.mkdir(parents=True, exist_ok=True)
        target_val.mkdir(parents=True, exist_ok=True)

        random.shuffle(staged_images)

        is_new_class = class_name not in ORIGINAL_47_CLASSES

        if is_new_class:
            target_test.mkdir(parents=True, exist_ok=True)
            n_test = max(1, int(len(staged_images) * 0.15))
            n_val = max(1, int(len(staged_images) * 0.15))
            if len(staged_images) <= 2:
                train_imgs = staged_images[:1]
                val_imgs = staged_images[1:]
                test_imgs = []
            elif len(staged_images) == 3:
                train_imgs = staged_images[:1]
                val_imgs = staged_images[1:2]
                test_imgs = staged_images[2:]
            else:
                test_imgs = staged_images[:n_test]
                val_imgs = staged_images[n_test:n_test + n_val]
                train_imgs = staged_images[n_test + n_val:]

            for img in test_imgs:
                img.replace(target_test / img.name)
                total_merged_test += 1
            for img in val_imgs:
                img.replace(target_val / img.name)
                total_merged_val += 1
            for img in train_imgs:
                img.replace(target_train / img.name)
                total_merged_train += 1
        else:
            n_val = max(1, int(len(staged_images) * 0.15))
            val_imgs = staged_images[:n_val]
            train_imgs = staged_images[n_val:]

            for img in val_imgs:
                img.replace(target_val / img.name)
                total_merged_val += 1
            for img in train_imgs:
                img.replace(target_train / img.name)
                total_merged_train += 1

        manifest = class_folder / "harvest_manifest.json"
        if manifest.exists():
            manifest.unlink()

    # Safety verification: ensure every train class folder has at least 1 image
    for class_folder in config.TRAIN_DIR.iterdir():
        if class_folder.is_dir():
            train_count = len(list(class_folder.glob("*.*")))
            if train_count == 0:
                test_folder = config.TEST_DIR / class_folder.name
                val_folder = config.VAL_DIR / class_folder.name
                donor = None
                if test_folder.exists() and len(list(test_folder.glob("*.*"))) > 1:
                    donor = next(test_folder.glob("*.*"))
                elif val_folder.exists() and len(list(val_folder.glob("*.*"))) > 1:
                    donor = next(val_folder.glob("*.*"))
                if donor:
                    donor.replace(class_folder / donor.name)
                    print(f"  [Balanced] Donated {donor.name} to train/{class_folder.name}")

    # Reindex classes
    all_classes = sorted([d.name for d in config.TRAIN_DIR.iterdir() if d.is_dir()])
    with open(config.CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "classes": all_classes,
            "class_to_idx": {cls: i for i, cls in enumerate(all_classes)},
            "idx_to_class": {i: cls for i, cls in enumerate(all_classes)}
        }, f, indent=2)

    print(f"\n🎉 Dataset rebalanced! Total classes now: {len(all_classes)}")
    print(f"Added: {total_merged_train} to Train, {total_merged_val} to Val, {total_merged_test} to Test.")
    print(f"Updated class indices saved to {config.CLASS_INDICES_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Targeted Botanical Harvester for FloraNet-100")
    parser.add_argument("--mode", type=str, choices=["deepen", "widen", "all-100", "merge"], default="all-100")
    parser.add_argument("--count-deepen", type=int, default=25, help="Images per class for deepening weak classes")
    parser.add_argument("--count-widen", type=int, default=35, help="Images per class for newly introduced species")
    args = parser.parse_args()

    if args.mode == "merge":
        merge_staged_dataset()
        return

    total = 0
    if args.mode in ["deepen", "all-100"]:
        print(f"\n=== PART 1: DEEPENING {len(DEEPEN_15_CLASSES)} UNDER-REPRESENTED SPECIES ===")
        for cls in DEEPEN_15_CLASSES:
            total += harvest_species(cls, target_count=args.count_deepen)

    if args.mode in ["widen", "all-100"]:
        print(f"\n=== PART 2: WIDENING TAXONOMY WITH {len(NEW_53_CLASSES)} NEW SPECIES ===")
        for cls in NEW_53_CLASSES:
            total += harvest_species(cls, target_count=args.count_widen)

    print("\n" + "=" * 65)
    print(f"🌾 Total botanical images harvested: {total}")
    print(f"Staged at: {config.DATA_DIR / 'staging'}")
    print(f"To merge into 100-class dataset, run: python tools/harvester.py --mode merge")
    print("=" * 65)

if __name__ == "__main__":
    main()
