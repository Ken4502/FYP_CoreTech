"""Builds canonical CoreTech v2 botanical taxonomy JSON (268+ species) directly from verified project sources + critical gaps."""
import json
import sys
from pathlib import Path

ROOT = Path(r"c:\Users\User\Desktop\FYP\CoreTech")
sys.path.insert(0, str(ROOT / "Plant_Identification_Experiment"))

from tools.harvester import BOTANICAL_TAXON_MAP
from tools.overnight_sea_expansion import SEA_10_SPECIES

V2_CFG_DIR = ROOT / "CoreTech_v2" / "configs"
V2_CFG_DIR.mkdir(parents=True, exist_ok=True)

taxonomy = {}
# 1. Start with the 250 core taxa from FloraNet
for common_sci, sci_name in BOTANICAL_TAXON_MAP.items():
    taxonomy[common_sci] = {
        "scientific_name": sci_name,
        "source_group": "floranet_250",
        "target_images": 500,
    }

# Verify against actual 250 train folders
p250_dirs = {d.name for d in (ROOT / "Plant_Identification_Experiment" / "data" / "train").iterdir() if d.is_dir()}
for dname in sorted(p250_dirs):
    if dname not in taxonomy:
        # fallback parse parenthetical
        if "(" in dname and dname.endswith(")"):
            sci = dname[dname.rfind("(")+1:-1]
        else:
            sci = dname
        taxonomy[dname] = {"scientific_name": sci, "source_group": "floranet_250", "target_images": 500}

# Prune any keys in BOTANICAL_TAXON_MAP that aren't in p250_dirs (keep strict alignment)
taxonomy = {k: v for k, v in taxonomy.items() if k in p250_dirs}

# 2. Add Pandan (exists in Indonesian Herb dataset + iNaturalist)
taxonomy["Pandan (Pandanus amaryllifolius)"] = {
    "scientific_name": "Pandanus amaryllifolius",
    "source_group": "sea_herb",
    "target_images": 550,
}

# 3. Add 10 Southeast Asian / Malaysian species (already 400 each in specimens/)
for k, info in SEA_10_SPECIES.items():
    taxonomy[k] = {
        "scientific_name": info["taxon"],
        "source_group": "sea_10",
        "target_images": 550,
    }

# 4. Critical Missing Core Host & Expanded Indoor Cactaceae Diversity (user-reported failure cases!)
EXPANDED_CORE = {
    "Spider Plant (Chlorophytum comosum)": {
        "scientific_name": "Chlorophytum comosum",
        "source_group": "core_houseplant_gap",
        "target_images": 650,
    },
    "Pincushion Cactus (Mammillaria)": {
        "scientific_name": "Mammillaria",
        "source_group": "expanded_cactaceae",
        "target_images": 650,
    },
    "Moon Cactus (Gymnocalycium)": {
        "scientific_name": "Gymnocalycium",
        "source_group": "expanded_cactaceae",
        "target_images": 650,
    },
    "Prickly Pear Cactus (Opuntia ficus-indica)": {
        "scientific_name": "Opuntia ficus-indica",
        "source_group": "expanded_cactaceae",
        "target_images": 650,
    },
    "Hedgehog Cactus (Echinopsis)": {
        "scientific_name": "Echinopsis",
        "source_group": "expanded_cactaceae",
        "target_images": 650,
    },
    "Peruvian Apple Cactus (Cereus repandus)": {
        "scientific_name": "Cereus repandus",
        "source_group": "expanded_cactaceae",
        "target_images": 600,
    },
    "Barrel Cactus (Ferocactus)": {
        "scientific_name": "Ferocactus",
        "source_group": "expanded_cactaceae",
        "target_images": 600,
    },
    "Star Cactus (Astrophytum)": {
        "scientific_name": "Astrophytum",
        "source_group": "expanded_cactaceae",
        "target_images": 600,
    },
}
taxonomy.update(EXPANDED_CORE)

# Boost existing cacti & supported hosts to 650 target images
for k, v in taxonomy.items():
    kl = k.lower()
    if any(x in kl for x in ["cactus", "aloe vera", "pothos", "snake plant", "haworthia", "jade plant", "echeveria"]):
        v["target_images"] = max(v["target_images"], 650)

# Attach botanical family tag for hierarchical genus/family uncertainty fallback
def infer_family(name: str, sci: str) -> str:
    s = f"{name} {sci}".lower()
    if any(k in s for k in ["cactus", "opuntia", "pachycereus", "disocactus", "echinocactus", "schlumbergera", "mammillaria", "gymnocalycium", "echinopsis", "cereus", "ferocactus", "astrophytum"]):
        return "Cactaceae (Cactus Family)"
    if any(k in s for k in ["aloe", "haworthia", "haworthiopsis"]):
        return "Asphodelaceae (Aloe & Haworthia)"
    if any(k in s for k in ["sansevieria", "sanseviera", "dracaena", "yucca", "aspidistra", "cordyline", "asparagus", "chlorophytum", "spider plant"]):
        return "Asparagaceae (Snake, Spider & Dracaena)"
    if any(k in s for k in ["pothos", "epipremnum", "scindapsus", "monstera", "philodendron", "alocasia", "anthurium", "syngonium", "dieffenbachia", "aglaonema", "zamioculcas", "zantedeschia", "spathiphyllum", "peace lily", "rhaphidophora"]):
        return "Araceae (Aroid & Pothos Family)"
    if any(k in s for k in ["crassula", "echeveria", "graptopetalum", "kalanchoe"]):
        return "Crassulaceae (Succulent Family)"
    if any(k in s for k in ["ficus", "fig "]):
        return "Moraceae (Fig Family)"
    if any(k in s for k in ["palm", "dypsis", "chamaedorea", "howea", "ravenea", "caryota", "cycas", "beaucarnea"]):
        return "Arecaceae / Cycadales (Palms)"
    if any(k in s for k in ["calathea", "maranta", "goeppertia", "stromanthe", "ctenanthe"]):
        return "Marantaceae (Prayer Plant Family)"
    if any(k in s for k in ["citrus", "murraya", "lemon", "lime", "orange", "calamansi"]):
        return "Rutaceae (Citrus Family)"
    if any(k in s for k in ["tomato", "potato", "eggplant", "pepper", "capsicum", "solanum"]):
        return "Solanaceae (Nightshade Family)"
    if any(k in s for k in ["basil", "mint", "mentha", "salvia", "rosemary", "thyme", "oregano", "lavender", "nepeta", "coleus", "melissa", "majorana", "glechoma", "lamium", "ocimum"]):
        return "Lamiaceae (Mint & Herb Family)"
    if any(k in s for k in ["fern", "nephrolepis", "asplenium", "platycerium", "adiantum", "microsorum", "zealandia"]):
        return "Polypodiopsida (Ferns)"
    if any(k in s for k in ["begonia"]):
        return "Begoniaceae (Begonia Family)"
    if any(k in s for k in ["peperomia", "piper"]):
        return "Piperaceae (Peperomia & Pepper Family)"
    if any(k in s for k in ["tradescantia"]):
        return "Commelinaceae (Tradescantia Family)"
    if any(k in s for k in ["curcuma", "etlingera", "zingiber", "turmeric", "ginger"]):
        return "Zingiberaceae (Ginger & Turmeric)"
    if any(k in s for k in ["cucumber", "pumpkin", "zucchini", "watermelon", "cantaloupe", "momordica", "cucurbita", "cucumis", "citrullus", "bitter gourd"]):
        return "Cucurbitaceae (Gourd & Squash)"
    if any(k in s for k in ["brassica", "raphanus", "capsella", "cabbage", "broccoli", "kale", "radish"]):
        return "Brassicaceae (Mustard & Brassicas)"
    if any(k in s for k in ["rosa", "fragaria", "malus", "rubus", "prunus", "spiraea", "rose", "strawberry", "apple"]):
        return "Rosaceae (Rose & Fruit Family)"
    if any(k in s for k in ["pisum", "phaseolus", "trifolium", "medicago", "clitoria", "lathyrus", "wisteria", "pea", "clover"]):
        return "Fabaceae (Legume Family)"
    if any(k in s for k in ["orchid"]):
        return "Orchidaceae (Orchid Family)"
    return "Other Botanical Taxa"

for k, v in taxonomy.items():
    v["family"] = infer_family(k, v["scientific_name"])

out_path = V2_CFG_DIR / "botanical_taxonomy_v2.json"
out_path.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(taxonomy)} species to {out_path}")
from collections import Counter
print("Family breakdown (top 15):", Counter(v["family"] for v in taxonomy.values()).most_common(15))
