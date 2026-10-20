"""
PhytoCare Ultra 31-Class Flagship Pathology & Universal Pest Suite
Harvests 9 additional high-impact universal houseplant pests & pathogens from iNaturalist:
  1. Universal_Thrips (Thripidae, ID: 83443)
  2. Universal_Whiteflies (Aleyrodidae, ID: 154784)
  3. Universal_Fungus_Gnats (Sciaridae, ID: 85259)
  4. Universal_Leaf_Miners (Agromyzidae, ID: 48086)
  5. Universal_Armored_Scales (Diaspididae, ID: 48093)
  6. Universal_Botrytis_Gray_Mold (Botrytis cinerea, ID: 324263)
  7. Universal_Alternaria_Leaf_Spot (Alternaria, ID: 429119)
  8. Universal_Downy_Mildew (Peronosporaceae, ID: 52598)
  9. Universal_Rust_Fungi (Pucciniales, ID: 69968)

Combines with existing 6 universal taxa + 16 host-specific classes for a 31-class flagship model.
Streams real-time telemetry to reports/mega_pipeline.log (visible at http://127.0.0.1:8001/log).
"""

import sys
import os
import time
import json
import random
import io
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from src import config

DATA_DIR = config.DATA_DIR
TRAIN_DIR = config.TRAIN_DIR
VAL_DIR = config.VAL_DIR
TEST_DIR = config.TEST_DIR
REPORTS_DIR = config.REPORTS_DIR
ARTIFACTS_DIR = config.ARTIFACTS_DIR
LOG_FILE = REPORTS_DIR / "mega_pipeline.log"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "PhytoCare-Research-Harvester/3.0 (academic FYP)"}

UNIVERSAL_TAXA = {
    # Existing 6 Universal Taxa
    "Universal_Mealybugs": {
        "taxon_id": 52788,
        "common_name": "Mealybugs (Pseudococcidae)",
        "scientific_name": "Pseudococcidae",
        "category": "Insect Infestation",
        "severity": "Moderate to High",
        "summary": "Sap-sucking soft-bodied scale insects covered in white cottony wax that cluster in leaf axils and undersides, causing leaf yellowing and honeydew secretion.",
        "symptoms": ["White cottony or powdery wax masses in leaf joints", "Yellowing, wilting, or distorted foliage growth", "Sticky honeydew accumulation", "Stunted plant development"],
        "causes": ["Contaminated nursery soil or unquarantined new plants", "Warm indoor conditions with stagnant, dry airflow", "Over-fertilization with high nitrogen"],
        "immediate_actions": ["Isolate plant immediately.", "Dab white cottony clusters with 70% isopropyl alcohol on a cotton swab.", "Rinse leaves with lukewarm water."],
        "treatment": {
            "organic": ["Spray with insecticidal soap every 5 days for 3 cycles.", "Apply cold-pressed neem oil spray."],
            "chemical": ["Systemic root drench containing acetamiprid or imidacloprid.", "Pyrethrin contact spray."]
        },
        "prevention": ["Quarantine new plants for 14 days.", "Inspect leaf axils regularly during watering."]
    },
    "Universal_Spider_Mites": {
        "taxon_id": 68318,
        "common_name": "Spider Mites (Tetranychidae)",
        "scientific_name": "Tetranychidae",
        "category": "Mite Infestation",
        "severity": "High",
        "summary": "Microscopic arachnid pests that pierce plant cell walls to feed on chlorophyll, producing fine silky webbing under leaves and stippled yellow foliage.",
        "symptoms": ["Fine silky webbing between leaf petioles and tips", "Pale yellow or silvery speckling (stippling) on foliage", "Dull, bronzed, or brittle foliage", "Leaf drop in severe infestations"],
        "causes": ["Hot, dry indoor environments with low humidity (<40%)", "Lack of foliar moisture and poor ventilation"],
        "immediate_actions": ["Isolate specimen immediately away from drafts.", "Take plant to the shower and blast undersides with lukewarm water.", "Mist foliage to raise local humidity."],
        "treatment": {
            "organic": ["Apply neem oil or rosemary-based miticide spray.", "Introduce predatory mites (Phytoseiulus persimilis)."],
            "chemical": ["Apply abamectin or bifenazate miticide spray targeting egg and adult cycles."]
        },
        "prevention": ["Maintain indoor relative humidity between 50-65%.", "Wipe down broad leaves with a damp cloth biweekly."]
    },
    "Universal_Powdery_Mildew": {
        "taxon_id": 55525,
        "common_name": "Powdery Mildew (Erysiphaceae)",
        "scientific_name": "Erysiphaceae",
        "category": "Fungal Infection",
        "severity": "Moderate",
        "summary": "Superficial fungal parasite creating white talcum powder-like circular patches across upper leaf surfaces, impeding photosynthesis.",
        "symptoms": ["White to light-gray powdery fungal patches on upper leaf surfaces", "Curling, twisting, or puckering of young leaves", "Yellowing under the fungal spots"],
        "causes": ["High relative humidity combined with low light and stagnant air", "Warm days and cool nights promoting spore germination"],
        "immediate_actions": ["Quarantine specimen.", "Prune heavily infected leaves and seal in trash.", "Increase room airflow using a gentle oscillating fan."],
        "treatment": {
            "organic": ["Spray potassium bicarbonate solution (1 tbsp per gallon water).", "Diluted milk spray (1 part milk : 9 parts water).", "Neem oil spray."],
            "chemical": ["Myclobutanil or sulfur-based fungicide spray."]
        },
        "prevention": ["Space plants adequately for airflow.", "Avoid overhead watering."]
    },
    "Universal_Scale_Insects": {
        "taxon_id": 49456,
        "common_name": "Soft Scale Insects (Coccidae)",
        "scientific_name": "Coccidae",
        "category": "Insect Infestation",
        "severity": "High",
        "summary": "Immobile waxy oval domes adhering firmly to stems and leaf ribs, sucking phloem sap and excreting copious honeydew.",
        "symptoms": ["Smooth, waxy, convex brown or yellow bumps on stems and leaf midribs", "Sticky honeydew residue on lower foliage and furniture", "Secondary black sooty mold growth"],
        "causes": ["Warm, dry indoor microclimates", "Absence of natural predators"],
        "immediate_actions": ["Isolate plant.", "Scrape off visible scale bumps using a soft toothbrush dipped in 70% rubbing alcohol.", "Wipe honeydew off leaves."],
        "treatment": {
            "organic": ["Thorough foliar spray with horticultural dormant oil or neem oil to suffocate scales.", "Beneficial lacewing larvae."],
            "chemical": ["Systemic dinotefuran or imidacloprid drench."]
        },
        "prevention": ["Inspect undersides of leaves and leaf stems monthly.", "Avoid stressed, underwatered plants."]
    },
    "Universal_Sooty_Mold": {
        "taxon_id": 154563,
        "common_name": "Sooty Mold (Capnodiaceae)",
        "scientific_name": "Capnodiaceae",
        "category": "Secondary Fungal Pathology",
        "severity": "Moderate",
        "summary": "Non-parasitic black fungal crust that colonizes sticky honeydew secreted by aphids, scales, or whiteflies, suffocating leaf respiration.",
        "symptoms": ["Black, charcoal-like powdery soot coating foliage", "Underlying sticky sugary sap layer", "Yellowing and poor growth due to light deprivation"],
        "causes": ["Underlying infestation of sap-sucking insects (aphids, scales, mealybugs, whiteflies)"],
        "immediate_actions": ["Eradicate the primary insect population.", "Gently wash foliage with lukewarm soapy water (mild castile soap).", "Gently wipe leaves with a microfiber towel."],
        "treatment": {
            "organic": ["Spray with 1% horticultural oil to loosen fungal crust.", "Wash with warm water and insecticidal soap."],
            "chemical": ["Treat underlying pests with systemic insecticide; sooty mold will starve and flake off."]
        },
        "prevention": ["Prevent honeydew-secreting insects.", "Clean leaf surfaces periodically."]
    },
    "Universal_Aphids": {
        "taxon_id": 52381,
        "common_name": "Aphids (Aphididae)",
        "scientific_name": "Aphididae",
        "category": "Insect Infestation",
        "severity": "Moderate to High",
        "summary": "Prolific soft-bodied pear-shaped insects clustering on tender new shoot tips and bud clusters, causing distorted curling growth and honeydew.",
        "symptoms": ["Dense clusters of tiny green, black, or yellow insects on new tips", "Curled, twisted, or stunted new leaf growth", "Shed white exoskeletons (casts) on foliage", "Sticky honeydew drops"],
        "causes": ["Tender over-fertilized nitrogen-rich flush", "Open windows in spring/summer allowing winged adults entry"],
        "immediate_actions": ["Isolate plant.", "Wash tips under a forceful stream of tepid tap water to blast aphids off.", "Squish remaining colonies by hand."],
        "treatment": {
            "organic": ["Spray with insecticidal soap or diluted castile soap (1 tbsp/L).", "Apply neem oil emulsion."],
            "chemical": ["Pyrethrin contact spray or systemic acetamiprid."]
        },
        "prevention": ["Check tender tips weekly.", "Do not over-apply synthetic high-nitrogen fertilizers."]
    },

    # 9 NEW UNIVERSAL TAXA (Expanding to 15 Universal / 31 Total Classes)
    "Universal_Thrips": {
        "taxon_id": 83443,
        "common_name": "Thrips (Thripidae)",
        "scientific_name": "Thripidae",
        "category": "Insect Infestation",
        "severity": "Very High",
        "summary": "Slender microscopic fringed-wing insects that rasp plant cells and suck pigments, leaving silvery mottled leaf surfaces and black tar-like fecal dots.",
        "symptoms": [
            "Silvery, bleached, or translucent patches on leaf surfaces",
            "Tiny black pepper-like fecal drops (frass) scattered over damaged areas",
            "Curled, deformed, or brown crinkled leaf margins",
            "Slender, fast-crawling yellowish or black insects visible under leaves"
        ],
        "causes": [
            "Introduction from outdoor garden flowers or uninspected fresh produce",
            "Warm indoor air (20-28°C) speeding up rapid multi-stage reproduction"
        ],
        "immediate_actions": [
            "Isolate the plant immediately in a separate closed room.",
            "Hang blue or yellow sticky cards at canopy height to capture adult flyers.",
            "Wipe down all leaves with a damp cloth to remove surface nymphs and frass."
        ],
        "treatment": {
            "organic": [
                "Apply Spinosad spray every 4-5 days covering top and bottom leaf surfaces thoroughly.",
                "Spray with insecticidal soap combined with cold-pressed neem oil.",
                "Introduce predatory mites (Amblyseius swirskii or cucumeris)."
            ],
            "chemical": [
                "Apply systemic insecticide containing acetamiprid or imidacloprid to kill larvae inside leaf tissue.",
                "Rotate with pyrethroid contact spray to prevent pesticide resistance."
            ]
        },
        "prevention": [
            "Inspect underside of new growth weekly using a 10x magnifying loupe.",
            "Avoid placing houseplants near open unscreened windows."
        ]
    },
    "Universal_Whiteflies": {
        "taxon_id": 154784,
        "common_name": "Whiteflies (Aleyrodidae)",
        "scientific_name": "Aleyrodidae",
        "category": "Insect Infestation",
        "severity": "High",
        "summary": "Tiny moth-like white insects clustering on lower leaf surfaces that flutter in a white cloud when the plant is disturbed, causing chlorosis and honeydew.",
        "symptoms": [
            "Cloud of tiny white winged insects fluttering when foliage is shaken",
            "Translucent scale-like flat nymphs fixed to the underside of leaves",
            "Yellowing, wilting foliage with general decline in plant vigor",
            "Copious sticky honeydew leading to secondary black sooty mold"
        ],
        "causes": [
            "Warm, humid indoor environments with poor air movement",
            "Introduction from greenhouse-grown nursery plants"
        ],
        "immediate_actions": [
            "Isolate specimen immediately.",
            "Use a handheld vacuum cleaner with low suction around foliage to capture adult flyers gently.",
            "Place bright yellow sticky traps directly beside the foliage."
        ],
        "treatment": {
            "organic": [
                "Spray undersides of leaves with insecticidal soap or horticultural oil every 3-5 days.",
                "Apply cold-pressed neem oil emulsion to suffocate nymph scales.",
                "Introduce Encarsia formosa parasitic micro-wasps for large collections."
            ],
            "chemical": [
                "Systemic root drench containing imidacloprid or dinotefuran.",
                "Contact bifenthrin or pyrethrin spray."
            ]
        },
        "prevention": [
            "Quarantine new plants for 2 weeks.",
            "Keep yellow sticky monitoring cards near collections year-round."
        ]
    },
    "Universal_Fungus_Gnats": {
        "taxon_id": 85259,
        "common_name": "Fungus Gnats (Sciaridae)",
        "scientific_name": "Sciaridae",
        "category": "Soil/Root Pest Infestation",
        "severity": "Moderate to High",
        "summary": "Tiny dark mosquito-like gnats walking across soil and fluttering near pots; subterranean larvae feed on feeder roots and soil fungi, triggering sudden wilting.",
        "symptoms": [
            "Small dark gnats emerging from soil surface and flying erratically around room",
            "Sudden unexplained leaf yellowing, wilting, and poor moisture uptake",
            "Loss of fine root hairs upon inspection, leading to secondary Pythium root rot",
            "Tiny translucent larvae with black head capsules visible in topsoil layer"
        ],
        "causes": [
            "Chronic overwatering and poorly draining potting soil holding excess moisture",
            "Organic potting mixes rich in uncomposted bark or peat breakdown"
        ],
        "immediate_actions": [
            "Allow the top 2-3 inches of potting soil to dry out completely before watering.",
            "Insert horizontal yellow sticky cards directly on top of soil to trap egg-laying adults.",
            "Remove stagnant runoff water from saucers."
        ],
        "treatment": {
            "organic": [
                "Drench soil with Bacillus thuringiensis israelensis (BTI / Mosquito Dunks) to destroy larvae biologically.",
                "Water soil with a 1:4 dilution of 3% hydrogen peroxide and water to kill larvae on contact.",
                "Apply beneficial parasitic nematodes (Steinernema feltiae) to potting soil."
            ],
            "chemical": [
                "Apply pyrethrin soil surface spray or systemic imidacloprid granules."
            ]
        },
        "prevention": [
            "Top dress potting soil with a 1/2-inch layer of coarse sand or pumice to prevent egg laying.",
            "Bottom-water plants and never let pots sit in standing water."
        ]
    },
    "Universal_Leaf_Miners": {
        "taxon_id": 48086,
        "common_name": "Leaf Miners (Agromyzidae)",
        "scientific_name": "Agromyzidae",
        "category": "Insect Pathology",
        "severity": "Moderate",
        "summary": "Fly larvae that bore inside the mesophyll layers of leaves, carving distinct white winding serpentine tunnels and translucent blotches.",
        "symptoms": [
            "Distinct serpentine white or translucent squiggly trails winding across leaves",
            "Blotchy translucent patches where inner leaf tissue has been excavated",
            "Tiny black dots of frass inside the translucent tunnels",
            "Premature browning and drop of heavily mined leaves"
        ],
        "causes": [
            "Adult female flies laying eggs directly into leaf epidermal slits",
            "Open windows or moving plants outdoors during warm seasons"
        ],
        "immediate_actions": [
            "Inspect leaf tunnels; gently squeeze the wide end between your fingers to crush the active larva inside.",
            "Prune off heavily mined leaves and dispose in sealed trash.",
            "Isolate the plant to prevent adult emergence."
        ],
        "treatment": {
            "organic": [
                "Spray foliage with cold-pressed neem oil to disrupt egg laying and larval feeding.",
                "Apply Spinosad spray which penetrates the leaf cuticle.",
                "Introduce parasitic Diglyphus isaea wasps."
            ],
            "chemical": [
                "Apply systemic translaminar insecticide (abamectin or acetamiprid)."
            ]
        },
        "prevention": [
            "Hang yellow sticky traps to intercept adult flies.",
            "Screen all windows near indoor plant displays."
        ]
    },
    "Universal_Armored_Scales": {
        "taxon_id": 48093,
        "common_name": "Armored Scales (Diaspididae)",
        "scientific_name": "Diaspididae",
        "category": "Insect Infestation",
        "severity": "High to Severe",
        "summary": "Hard-shelled scale insects with rigid waxy armor detached from their bodies; they suck mesophyll fluids without producing honeydew, causing yellow foliar halo rings.",
        "symptoms": [
            "Hard, flattened or oyster-shell shaped brown, gray, or white encrustations on stems and leaves",
            "Yellow chlorotic halos or spotting directly under and around each scale",
            "Lack of sticky honeydew (distinguishing armored scales from soft scales)",
            "Leaf yellowing, branch dieback, and bark cracking in severe cases"
        ],
        "causes": [
            "Introduction from commercial greenhouses or infected cuttings",
            "Dense foliage with low light and poor ventilation"
        ],
        "immediate_actions": [
            "Isolate specimen immediately.",
            "Use a soft toothbrush or plastic card to scrape off visible armored shells mechanically.",
            "Prune severely encrusted stems and discard."
        ],
        "treatment": {
            "organic": [
                "Apply horticultural mineral oil or neem oil spray to smother adult armor and nymphs.",
                "Repeat oil sprays every 7-10 days for 4 treatments to intercept crawler hatch cycles."
            ],
            "chemical": [
                "Apply systemic insecticide containing dinotefuran or pyriproxyfen (insect growth regulator).",
                "Contact pyrethroid spray timed during crawler migration."
            ]
        },
        "prevention": [
            "Scrutinize woody stems and leaf undersides of all new acquisitions.",
            "Clean pruning shears with alcohol between plants."
        ]
    },
    "Universal_Botrytis_Gray_Mold": {
        "taxon_id": 324263,
        "common_name": "Botrytis Gray Mold (Botrytis cinerea)",
        "scientific_name": "Botrytis cinerea",
        "category": "Fungal Infection",
        "severity": "High",
        "summary": "Aggressive necrotrophic fungus producing fuzzy grayish-brown velvety spore cushions on decaying leaves, flower petals, and soft stem bases in humid stagnant rooms.",
        "symptoms": [
            "Fuzzy, grayish-brown velvety mold cushions growing over foliage or flowers",
            "Water-soaked brown spots on leaves that rapidly enlarge and turn soft",
            "Brown stem lesions causing sudden collapse (damping off) of stems",
            "Puff of powdery gray spores released when touched"
        ],
        "causes": [
            "High relative humidity (>85%) combined with cool/moderate temperatures (15-20°C)",
            "Poor air circulation and water droplets sitting on leaves or flower petals overnight"
        ],
        "immediate_actions": [
            "Isolate the plant immediately away from all airflow that could blow spores.",
            "Prune off all infected tissue with sterile shears and immediately bag in plastic.",
            "Cease all misting and foliar watering."
        ],
        "treatment": {
            "organic": [
                "Improve air circulation immediately with a low-speed oscillating fan.",
                "Spray copper-based fungicide or bio-fungicide containing Bacillus subtilis.",
                "Apply potassium bicarbonate spray."
            ],
            "chemical": [
                "Apply fludioxonil, iprodione, or chlorothalonil fungicide spray."
            ]
        },
        "prevention": [
            "Water strictly at soil level early in the morning.",
            "Maintain indoor relative humidity below 65% and keep foliage dry."
        ]
    },
    "Universal_Alternaria_Leaf_Spot": {
        "taxon_id": 429119,
        "common_name": "Alternaria Leaf Spot (Alternaria)",
        "scientific_name": "Alternaria",
        "category": "Fungal Infection",
        "severity": "Moderate to High",
        "summary": "Fungal leaf pathogen creating dark brown to black circular lesions with distinct concentric target rings and surrounding yellow chlorotic halos.",
        "symptoms": [
            "Circular brown or black spots on leaves featuring concentric rings ('target spot' pattern)",
            "Prominent yellow chlorotic halo surrounding each circular necrotic lesion",
            "Lesions coalescing into large necrotic blights causing leaf withering",
            "Premature defoliation starting from lower foliage"
        ],
        "causes": [
            "Foliage remaining wet for 8+ hours from overhead watering or misting",
            "Warm indoor temperatures (20-27°C) with humid stagnant air"
        ],
        "immediate_actions": [
            "Quarantine plant.",
            "Prune spotted leaves using clean alcohol-sterilized scissors.",
            "Immediately stop all overhead misting."
        ],
        "treatment": {
            "organic": [
                "Apply liquid copper octanoate or sulfur-based organic fungicide.",
                "Spray with Bacillus amyloliquefaciens biofungicide.",
                "Ensure maximum light and rapid foliage drying."
            ],
            "chemical": [
                "Apply broad-spectrum systemic fungicide containing difenoconazole, azoxystrobin, or chlorothalonil."
            ]
        },
        "prevention": [
            "Always bottom-water or water at soil base; never wet leaves.",
            "Space pots so leaves do not touch."
        ]
    },
    "Universal_Downy_Mildew": {
        "taxon_id": 52598,
        "common_name": "Downy Mildew (Peronosporaceae)",
        "scientific_name": "Peronosporaceae",
        "category": "Oomycete Pathogen",
        "severity": "High",
        "summary": "Water mold organism causing angular yellow chlorotic patches constrained by leaf veins on upper leaf surfaces, with purplish-gray downy fuzz underneath.",
        "symptoms": [
            "Angular, vein-bounded yellow or pale green patches on upper leaf surfaces",
            "Delicate purplish-gray downy/felt-like fungal growth on the underside of yellow patches",
            "Leaves turning brown, dry, and brittle, curling downwards",
            "Rapid defoliation under cool, humid conditions"
        ],
        "causes": [
            "High relative humidity (>85%) and free water on leaves during cool nights",
            "Overcrowded plants with zero air circulation"
        ],
        "immediate_actions": [
            "Quarantine specimen immediately.",
            "Remove all symptomatic leaves and burn or discard in sealed trash.",
            "Drastically reduce humidity and increase airflow."
        ],
        "treatment": {
            "organic": [
                "Spray copper-based fungicide to protect unaffected foliage.",
                "Apply potassium phosphite (phosphonate) systemic plant defense activator."
            ],
            "chemical": [
                "Apply dimethomorph, mefenoxam, or fosetyl-aluminum systemic fungicide."
            ]
        },
        "prevention": [
            "Avoid cool night temperatures combined with high humidity.",
            "Provide ample spacing between indoor plants."
        ]
    },
    "Universal_Rust_Fungi": {
        "taxon_id": 69968,
        "common_name": "Rust Fungi (Pucciniales)",
        "scientific_name": "Pucciniales",
        "category": "Fungal Infection",
        "severity": "Moderate to High",
        "summary": "Obligate fungal parasite producing raised reddish-orange, yellow, or cinnamon-brown blister-like spore pustules (uredinia) that rupture on leaf undersides.",
        "symptoms": [
            "Raised orange, bright yellow, or rusty-brown powdery blisters on lower leaf surfaces",
            "Corresponding pale yellow or chlorotic flecks on the upper leaf surface directly above blisters",
            "Rusty powder rubs off easily onto fingers when touched",
            "Heavy leaf distortion, yellowing, and premature leaf drop"
        ],
        "causes": [
            "Extended moisture on leaves (4-6 hours of free water required for spore infection)",
            "Moderate temperatures (18-24°C) with still air"
        ],
        "immediate_actions": [
            "Isolate the plant immediately in a separate dry room.",
            "Carefully snip off infected leaves over a damp towel to prevent rust spores from wafting.",
            "Cease all foliar spraying and keep leaf surfaces bone-dry."
        ],
        "treatment": {
            "organic": [
                "Dust or spray with wettable micronized sulfur.",
                "Apply copper octanoate fungicide every 7-10 days."
            ],
            "chemical": [
                "Apply systemic DMI fungicide such as myclobutanil, tebuconazole, or propiconazole to eradicate systemic mycelium."
            ]
        },
        "prevention": [
            "Water early in the day at soil line only.",
            "Provide strong ventilation and maintain dry foliage."
        ]
    }
}

def log_event(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def fetch_inat_taxa_urls(taxon_id: int, target_count: int = 500) -> list:
    urls = []
    page = 1
    per_page = 100
    max_pages = 10
    
    while len(urls) < target_count and page <= max_pages:
        endpoint = (
            f"https://api.inaturalist.org/v1/observations?"
            f"taxon_id={taxon_id}&has[]=photos&quality_grade=research&"
            f"per_page={per_page}&page={page}&order=desc&order_by=votes"
        )
        try:
            req = urllib.request.Request(endpoint, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if not results:
                break
            for obs in results:
                photos = obs.get("photos", [])
                for p in photos:
                    u = p.get("url")
                    if u:
                        large_u = u.replace("square.", "medium.").replace("small.", "medium.")
                        if large_u not in urls:
                            urls.append(large_u)
                            if len(urls) >= target_count:
                                break
                if len(urls) >= target_count:
                    break
            page += 1
            time.sleep(0.3)
        except Exception as e:
            log_event(f"  [Taxon {taxon_id}] Page {page} fetch error: {e}")
            break
            
    return urls[:target_count]

def download_and_save_image(url: str, dest_path: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            content = resp.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        if img.width < 150 or img.height < 150:
            return False
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_path, "JPEG", quality=90)
        return True
    except Exception:
        return False

def harvest_all_universal_taxa():
    log_event("\n" + "=" * 70)
    log_event("PHASE 1: HARVESTING COMPLETE UNIVERSAL PEST & PATHOGEN SUITE")
    log_event(f"• Total Universal Classes: {len(UNIVERSAL_TAXA)}")
    log_event(f"• Target Images per Taxon: 500 Research-Grade Specimens")
    log_event("=" * 70)

    for cls_name, info in UNIVERSAL_TAXA.items():
        train_dir = TRAIN_DIR / cls_name
        val_dir = VAL_DIR / cls_name
        test_dir = TEST_DIR / cls_name

        existing_train = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
        if existing_train >= 300:
            log_event(f"✓ [{cls_name}] Already populated with {existing_train} train images. Skipping harvest.")
            continue

        log_event(f"\nHarvesting research-grade observations for: {cls_name} (Taxon ID: {info['taxon_id']})...")
        urls = fetch_inat_taxa_urls(info["taxon_id"], target_count=500)
        log_event(f"  Found {len(urls)} observation URLs for {cls_name}")

        if not urls:
            log_event(f"  [Warning] No URLs retrieved for {cls_name}.")
            continue

        random.seed(42)
        random.shuffle(urls)

        # 70% Train, 15% Val, 15% Test
        n = len(urls)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_urls = urls[:n_train]
        val_urls = urls[n_train:n_train + n_val]
        test_urls = urls[n_train + n_val:]

        tasks = []
        for i, u in enumerate(train_urls):
            tasks.append((u, train_dir / f"inat_{cls_name}_{i+1:04d}.jpg"))
        for i, u in enumerate(val_urls):
            tasks.append((u, val_dir / f"inat_{cls_name}_{i+1:04d}.jpg"))
        for i, u in enumerate(test_urls):
            tasks.append((u, test_dir / f"inat_{cls_name}_{i+1:04d}.jpg"))

        saved_count = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(download_and_save_image, u, p): p for (u, p) in tasks}
            for fut in as_completed(future_to_url):
                if fut.result():
                    saved_count += 1
                if saved_count % 100 == 0 or saved_count == len(tasks):
                    log_event(f"  [{cls_name}] Downloaded and verified {saved_count}/{len(tasks)} images...")

        log_event(f"  ✓ {cls_name} ready: {len(list(train_dir.glob('*.jpg')))} train, {len(list(val_dir.glob('*.jpg')))} val, {len(list(test_dir.glob('*.jpg')))} test.")

def update_knowledge_base_and_indices():
    log_event("\n" + "=" * 70)
    log_event("PHASE 2: UPDATING CLINICAL KNOWLEDGE BASE & 31-CLASS INDICES")
    log_event("=" * 70)

    kb_path = ARTIFACTS_DIR / "disease_knowledge_base.json"
    if kb_path.exists():
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
    else:
        kb = {}

    for cls_name, info in UNIVERSAL_TAXA.items():
        kb[cls_name] = {
            "class_id": cls_name,
            "plant_name": "Universal Houseplants",
            "scientific_name": info["scientific_name"],
            "condition_name": info["common_name"],
            "is_healthy": False,
            "severity": info["severity"],
            "category": info["category"],
            "summary": info["summary"],
            "symptoms": info["symptoms"],
            "causes": info["causes"],
            "immediate_actions": info["immediate_actions"],
            "treatment": info["treatment"],
            "prevention": info["prevention"]
        }

    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2)
    log_event(f"✓ Knowledge Base updated at {kb_path} (Total Entries: {len(kb)})")

    all_classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    indices_path = ARTIFACTS_DIR / "class_indices.json"
    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump({cls: i for i, cls in enumerate(all_classes)}, f, indent=2)
    log_event(f"✓ 31-Class Indices updated at {indices_path} ({len(all_classes)} Total Taxa):")
    for i, c in enumerate(all_classes):
        log_event(f"    [{i:02d}] {c}")

    return all_classes

def train_31class_flagship(all_classes: list):
    log_event("\n" + "=" * 70)
    log_event("PHASE 3: TRAINING 31-CLASS CONVNEXT-SMALL BOTANICAL FLAGSHIP")
    log_event(f"• Target Device:    {config.DEVICE} (RTX 3060 12GB VRAM Max Performance)")
    log_event(f"• Architecture:     ConvNeXt-Small (50.2M Parameters, 384x384 Native)")
    log_event(f"• Loss Function:    Class-Balanced Focal Loss (gamma=2.0, label_smoothing=0.05)")
    log_event("=" * 70)

    num_classes = len(all_classes)

    train_tf = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.RandomResizedCrop(384, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
    val_ds = datasets.ImageFolder(str(VAL_DIR), transform=val_tf)

    log_event(f"[Corpus] Loaded {len(train_ds)} train images, {len(val_ds)} validation images across {num_classes} classes.")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    model = models.convnext_small(weights=None)
    in_features = model.classifier[2].in_features

    # Backbone transfer from best existing 22-class model
    ckpt_path = ARTIFACTS_DIR / "best_model.pth"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", {})
            filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("classifier.2")}
            model.load_state_dict(filtered_state, strict=False)
            log_event(f"✓ Transferred backbone weights from {ckpt_path}")
        except Exception as e:
            log_event(f"[Warning] Backbone transfer error: {e}, falling back to ImageNet prior.")
            model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
    else:
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)

    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(config.DEVICE)

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, label_smoothing=0.05):
            super().__init__()
            self.gamma = gamma
            self.smoothing = label_smoothing

        def forward(self, inputs, targets):
            ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.smoothing)
            pt = torch.exp(-ce_loss)
            focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=3.5e-5, weight_decay=1e-4)
    epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_acc = 0.0
    best_weights = None

    log_event("\n" + "=" * 70)
    log_event(f"COMMENCING 31-CLASS PROGRESSIVE HIGH-ACUITY TRAINING (10 EPOCHS @ 384x384)")
    log_event("=" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct_train += preds.eq(labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                for images, labels in val_loader:
                    images = images.to(config.DEVICE, non_blocking=True)
                    labels = labels.to(config.DEVICE, non_blocking=True)
                    outputs = model(images)
                    _, preds = outputs.max(1)
                    correct_val += preds.eq(labels).sum().item()
                    total_val += labels.size(0)

        val_acc = (correct_val / total_val) * 100.0
        elapsed = round(time.time() - t0, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        log_event(f"  Epoch [{epoch:02d}/{epochs:02d}] ({elapsed}s) | Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {lr_now:.7f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()
            log_event(f"    ★ Peak Validation Accuracy Reached: {best_val_acc:.2f}% (Saved Checkpoint)")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Temperature Scaling
    log_event("\n[Calibration] Tuning Temperature Scaling on Validation Set...")
    model.eval()
    val_logits = []
    val_targets = []
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            for images, labels in val_loader:
                images = images.to(config.DEVICE)
                val_logits.append(model(images).cpu())
                val_targets.append(labels)

    val_logits = torch.cat(val_logits, dim=0).float()
    val_targets = torch.cat(val_targets, dim=0)

    temp_param = nn.Parameter(torch.ones(1) * 1.0)
    optimizer_temp = optim.LBFGS([temp_param], lr=0.01, max_iter=50)

    def temp_eval():
        optimizer_temp.zero_grad()
        loss = nn.functional.cross_entropy(val_logits / temp_param, val_targets)
        loss.backward()
        return loss

    optimizer_temp.step(temp_eval)
    calibrated_temp = float(torch.clamp(temp_param, min=0.5, max=2.5).item())
    log_event(f"✓ Calibrated Temperature: T = {calibrated_temp:.2f}")

    # Save 31-class weights
    best_path = ARTIFACTS_DIR / "best_model.pth"
    best_31_path = ARTIFACTS_DIR / "best_model_31class.pth"

    ckpt_save = {
        "model_state_dict": model.state_dict(),
        "classes": all_classes,
        "temperature": calibrated_temp,
        "val_acc": best_val_acc,
        "model_architecture": "PhytoCare Ultra ConvNeXt-Small 384px (31 Classes)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    torch.save(ckpt_save, best_path)
    torch.save(ckpt_save, best_31_path)
    log_event(f"✓ Saved 31-class flagship model to {best_path}")

    return model, calibrated_temp

def evaluate_31class_test_set(model, calibrated_temp: float, all_classes: list):
    log_event("\n" + "=" * 70)
    log_event("PHASE 4: EVALUATING 31-CLASS MODEL ON COMPLETE TEST SET (5-CROP TTA)")
    log_event("=" * 70)

    test_tf = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = datasets.ImageFolder(str(TEST_DIR), transform=test_tf)
    log_event(f"[Dataset] Evaluating {len(test_ds)} unseen test images across {len(all_classes)} classes...")

    model.eval()
    all_preds = []
    all_targets = []

    def get_5crop_batch(img_tensor):
        c, h, w = img_tensor.shape
        crop_size = 384
        crops = [
            img_tensor[:, :crop_size, :crop_size],
            img_tensor[:, :crop_size, w - crop_size:],
            img_tensor[:, h - crop_size:, :crop_size],
            img_tensor[:, h - crop_size:, w - crop_size:],
            img_tensor[:, (h - crop_size)//2:(h - crop_size)//2 + crop_size, (w - crop_size)//2:(w - crop_size)//2 + crop_size]
        ]
        return torch.stack(crops, dim=0)

    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            for idx, (img_t, target) in enumerate(test_ds):
                crops = get_5crop_batch(img_t).to(config.DEVICE)
                logits = model(crops)
                probs = torch.softmax(logits / calibrated_temp, dim=1).mean(dim=0)
                pred = probs.argmax().item()

                all_preds.append(pred)
                all_targets.append(target)

                if (idx + 1) % 250 == 0 or (idx + 1) == len(test_ds):
                    running_acc = (sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_preds)) * 100
                    log_event(f"  [{idx + 1:4d}/{len(test_ds)}] Running Test Accuracy: {running_acc:.2f}%")

    eval_duration = round(time.time() - t0, 1)
    report_text = classification_report(
        all_targets,
        all_preds,
        target_names=all_classes,
        digits=4,
        zero_division=0
    )

    log_event("\n" + "=" * 70)
    log_event("31-CLASS COMPREHENSIVE CLASSIFICATION REPORT:")
    log_event("=" * 70)
    log_event(f"Duration: {eval_duration}s\n" + report_text)

    # Save report
    rep_file = REPORTS_DIR / "classification_report.txt"
    with open(rep_file, "w", encoding="utf-8") as f:
        f.write(f"PhytoCare Ultra 31-Class Benchmark Performance (Duration: {eval_duration}s)\n\n" + report_text)
    log_event(f"✓ Saved updated classification report to {rep_file}")

    # Plot Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=all_classes,
        yticklabels=all_classes,
        title="PhytoCare Ultra 31-Class Confusion Matrix (ConvNeXt-Small 384px)",
        ylabel="True Pathology / Pest Taxon",
        xlabel="Predicted Diagnosis"
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                ax.text(j, i, format(val, 'd'),
                        ha="center", va="center",
                        color="white" if val > cm.max() / 2. else "black",
                        fontsize=6)

    fig.tight_layout()
    cm_plot_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    log_event(f"✓ Saved 31-class confusion matrix plot to {cm_plot_path}")

    # Reload port 8001
    try:
        reload_req = urllib.request.Request("http://127.0.0.1:8001/api/model/reload", data=b"", headers={"User-Agent": "PhytoCare"})
        with urllib.request.urlopen(reload_req, timeout=5) as r:
            pass
        log_event("✓ Triggered live hot-reload on clinical web service (Port 8001).")
    except Exception:
        log_event("[Notice] Web server on port 8001 will load new weights upon restart.")

def main():
    t_start = time.time()
    log_event("======================================================================")
    log_event("STARTING PHYTONET ULTRA 31-CLASS HARVEST & FLAGSHIP TRAINING PIPELINE")
    log_event("======================================================================")

    # 1. Harvest
    harvest_all_universal_taxa()

    # 2. Update Knowledge Base & Indices
    all_classes = update_knowledge_base_and_indices()

    # 3. Train
    model, calibrated_temp = train_31class_flagship(all_classes)

    # 4. Evaluate Test Set
    evaluate_31class_test_set(model, calibrated_temp, all_classes)

    total_time = round((time.time() - t_start) / 60, 2)
    log_event("\n" + "=" * 70)
    log_event(f"31-CLASS UNIVERSAL PATHOLOGY PIPELINE COMPLETED IN {total_time} MINUTES!")
    log_event("======================================================================")

if __name__ == "__main__":
    main()

