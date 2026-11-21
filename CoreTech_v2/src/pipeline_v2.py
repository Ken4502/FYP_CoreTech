"""
CoreTech v2 Unified End-to-End Diagnostic Pipeline (100% Online Server Architecture).
Implements Stages 0 -> 1A -> 1B -> 2 -> 3 -> 4 with graceful user recommendations:
  - Stage 0: Objective Image Quality Gate (Focus / Exposure / Resolution)
  - Stage 1A: Online Gemini Vision Non-Plant Semantic Gate
  - Stage 1B: YOLO11-Seg Multi-Plant Dominant Coverage Isolation (focuses ONLY on highest coverage plant)
  - Stage 2: Foundation Botanical Ensemble (BioCLIP-2 + EVA-02 + ConvNeXt-Base) + Genus/OOD Gate
  - Stage 3: Dual-Scale Surface Pathology Ensemble + YOLO11 Lesion Detector + Masked Hydration Stress
  - Stage 4: Actionable Recommendations & Backward-Compatible Flutter Payload
"""
import io
import sys
import time
import base64
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image

V2_ROOT = Path(__file__).resolve().parent.parent
CORETECH_ROOT = V2_ROOT.parent
if str(CORETECH_ROOT) not in sys.path:
    sys.path.append(str(CORETECH_ROOT))
if str(V2_ROOT) not in sys.path:
    sys.path.append(str(V2_ROOT))

try:
    from configs.v2_config import QUALITY_THRESHOLDS
except (ImportError, ValueError):
    from CoreTech_v2.configs.v2_config import QUALITY_THRESHOLDS

try:
    from .quality_gate import evaluate_image_quality, normalize_illumination_lab
    from .plant_verifier import verify_plant_online, rerank_species_with_gemini
    from .dominant_isolator import isolate_dominant_plant
    from .botanical_ensemble import identify_dominant_plant, get_species_visual_priors, extract_genus_from_species
    from .surface_pathology import diagnose_surface_condition
except (ImportError, ValueError):
    try:
        from CoreTech_v2.src.quality_gate import evaluate_image_quality, normalize_illumination_lab
        from CoreTech_v2.src.plant_verifier import verify_plant_online, rerank_species_with_gemini
        from CoreTech_v2.src.dominant_isolator import isolate_dominant_plant
        from CoreTech_v2.src.botanical_ensemble import identify_dominant_plant, get_species_visual_priors, extract_genus_from_species
        from CoreTech_v2.src.surface_pathology import diagnose_surface_condition
    except (ImportError, ValueError):
        from src.quality_gate import evaluate_image_quality, normalize_illumination_lab
        from src.plant_verifier import verify_plant_online, rerank_species_with_gemini
        from src.dominant_isolator import isolate_dominant_plant
        from src.botanical_ensemble import identify_dominant_plant, get_species_visual_priors, extract_genus_from_species
        from src.surface_pathology import diagnose_surface_condition


def _img_to_b64(pil_img: Image.Image, max_size: int = 640) -> str:
    w, h = pil_img.size
    if max(w, h) > max_size:
        sc = max_size / float(max(w, h))
        pil_img = pil_img.resize((max(32, int(w * sc)), max(32, int(h * sc))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


SURFACE_CARE_MAP = {
    "Healthy_Foliage": {
        "condition_name": "Healthy Foliage & Optimal Surface Condition",
        "category": "Healthy",
        "severity": "None",
        "summary": "No active fungal lesions, pest colonies, or severe surface stress detected on the primary plant.",
        "immediate_actions": ["No treatment needed. Continue current light and watering schedule."],
    },
    "Fungal_Leaf_Spot_Anthracnose": {
        "condition_name": "Fungal Leaf Spot / Anthracnose",
        "category": "Fungal Infection",
        "severity": "Moderate",
        "summary": "Necrotic spotting, target rings, or sunken lesions detected on plant surface foliage.",
        "immediate_actions": [
            "Isolate plant away from neighboring foliage to prevent spore splash.",
            "Prune heavily spotted leaves with 70% alcohol-sterilized scissors.",
            "Stop overhead misting; apply copper fungicide or neem oil spray.",
        ],
    },
    "Bacterial_Wilt_Soft_Rot": {
        "condition_name": "Bacterial Spot / Soft Rot",
        "category": "Bacterial Infection",
        "severity": "High",
        "summary": "Water-soaked spreading lesions or bacterial leaf blight patterns detected.",
        "immediate_actions": [
            "Quarantine immediately and sterilize tools between every cut.",
            "Remove infected tissue below the margin and improve air circulation.",
            "Avoid wetting leaves and reduce excess nitrogen fertilization.",
        ],
    },
    "Botrytis_Gray_Mold": {
        "condition_name": "Botrytis Gray Mold",
        "category": "Fungal Infection",
        "severity": "High",
        "summary": "Fuzzy gray/brown decay cushions typical of Botrytis cinerea detected.",
        "immediate_actions": [
            "Remove decaying flowers/leaves carefully into a sealed bag.",
            "Increase airflow with a gentle fan and lower ambient humidity.",
        ],
    },
    "Powdery_Mildew": {
        "condition_name": "Powdery Mildew",
        "category": "Fungal Infection",
        "severity": "Moderate",
        "summary": "White talcum-like superficial fungal mycelium patches detected on leaf surface.",
        "immediate_actions": [
            "Wipe visible spores gently with a damp cloth, then spray potassium bicarbonate or neem oil.",
            "Move to brighter indirect light with stronger airflow.",
        ],
    },
    "Downy_Mildew": {
        "condition_name": "Downy Mildew",
        "category": "Oomycete Infection",
        "severity": "Moderate to High",
        "summary": "Angular chlorotic patches and down-surface sporulation detected.",
        "immediate_actions": [
            "Water strictly at soil level in the morning so foliage stays dry.",
            "Remove infected leaves and treat with copper-based fungicide.",
        ],
    },
    "Rust_Fungi": {
        "condition_name": "Rust Fungi (Pucciniales)",
        "category": "Fungal Infection",
        "severity": "Moderate",
        "summary": "Raised orange, cinnamon, or rust-brown spore pustules detected on surface tissue.",
        "immediate_actions": [
            "Prune rusted leaves without shaking spores onto clean foliage.",
            "Apply sulfur or neem fungicide and eliminate leaf-surface condensation.",
        ],
    },
    "Rust_Fungi_With_Bug_Bites": {
        "condition_name": "Rust Fungi (Pucciniales) & Insect Chew Holes",
        "category": "Co-Occurring Biotic Infection & Pest Damage",
        "severity": "High",
        "summary": "Both raised orange fungal rust spore pustules (Pucciniales) and insect chew holes / bite perforations are actively detected on foliage.",
        "immediate_actions": [
            "Prune heavily rusted and chewed leaves cleanly into a sealed disposal bag to stop spore spread.",
            "Apply dual treatment: spray neem oil emulsion or Bacillus thuringiensis (Bt) for chewing pests, combined with sulfur or copper fungicide for rust.",
            "Inspect leaf undersides at night for nocturnal chewing caterpillars, beetles, or slugs.",
            "Eliminate overhead leaf misting and provide gentle airflow to prevent new fungal spore germination.",
        ],
    },
    "Sooty_Mold": {
        "condition_name": "Sooty Mold",
        "category": "Secondary Fungal Crust",
        "severity": "Moderate",
        "summary": "Black soot-like crust coating leaf surface, typically following honeydew from sap-suckers.",
        "immediate_actions": [
            "Wash off black crust gently with mild lukewarm insecticidal soap.",
            "Inspect leaf undersides for underlying aphids, scales, or whiteflies.",
        ],
    },
    "Leaf_Miners": {
        "condition_name": "Leaf Miners",
        "category": "Insect Infestation",
        "severity": "Mild to Moderate",
        "summary": "Pale winding serpentine tunnels burrowed inside leaf mesophyll.",
        "immediate_actions": [
            "Prune heavily mined leaves or gently crush larvae visible at tunnel ends.",
            "Apply Spinosad or neem oil if new trails continue appearing.",
        ],
    },
    "Chewing_Pests_Bug_Bites_Holes": {
        "condition_name": "Insect Feeding Damage (Bug Bites & Chewed Holes)",
        "category": "Insect Feeding Damage",
        "severity": "Moderate to High",
        "summary": "Foliar chew holes, shot-hole perforations, or ragged bitten leaf margins detected from chewing insect pests (caterpillars, beetles, grasshoppers, leaf-cutter bees, or slugs/snails).",
        "immediate_actions": [
            "Inspect the plant carefully, especially leaf undersides and stem bases at night or dusk, for active caterpillars, beetles, slugs, or snails.",
            "Hand-pick and remove any visible caterpillars, beetles, or worms immediately.",
            "Apply Bacillus thuringiensis (Bt) spray for caterpillars, or food-grade diatomaceous earth around the soil base for crawling pests/slugs.",
            "Spray cold-pressed neem oil or organic insecticidal soap across foliage to deter further chewing feeding.",
            "Prune severely tattered or perforated leaves that have lost photosynthetic capacity.",
        ],
    },
    "Spider_Mites": {
        "condition_name": "Spider Mites",
        "category": "Mite Infestation",
        "severity": "High",
        "summary": "Fine stippling, bronzed texture, or silky micro-webbing detected.",
        "immediate_actions": [
            "Isolate plant immediately and rinse undersides thoroughly with tepid water.",
            "Treat every 4-5 days with insecticidal soap, neem oil, or miticide.",
        ],
    },
    "Mealybugs_Cochineal": {
        "condition_name": "Mealybugs / Cochineal Wax Scale",
        "category": "Insect Infestation",
        "severity": "High",
        "summary": "White cottony, powdery, or waxy masses/clusters detected along stems, leaf joints, or leaf undersides. Mealybugs suck vital plant sap and excrete sticky honeydew.",
        "immediate_actions": [
            "Isolate specimen immediately to prevent mealybugs spreading to neighboring plants.",
            "Dip a cotton swab in 70% isopropyl alcohol and dab directly onto all white cottony clusters to dissolve their protective wax and kill pests on contact.",
            "Spray the entire plant thoroughly (especially leaf undersides, stem joints, and petiole crotches) with neem oil emulsion or insecticidal soap.",
            "Inspect soil surface; drench root zone with systemic pesticide if subterranean root mealybugs are present.",
            "Repeat weekly checks and treatments until all white waxy fluff and insects are eradicated.",
        ],
    },
    "Scale_Insects": {
        "condition_name": "Scale Insects (Armored / Soft Shell)",
        "category": "Insect Infestation",
        "severity": "High",
        "summary": "Waxy brown, black, or tan oval shell bumps firmly adhered along stems, petioles, and leaf midribs sucking sap.",
        "immediate_actions": [
            "Isolate plant immediately to halt nymph crawler migration.",
            "Scrape off visible adult scale shells gently using a soft toothbrush dipped in 70% isopropyl alcohol.",
            "Suffocate crawler nymphs by spraying thoroughly with horticultural dormant oil or insecticidal soap.",
            "Check leaf undersides and stem crotches weekly for new scale clusters.",
        ],
    },
    "Aphids": {
        "condition_name": "Aphids (Black Aphids / Plant Lice)",
        "category": "Insect Infestation",
        "severity": "Moderate to High",
        "summary": "Dense colonies of small soft-bodied sap-sucking insects (black aphids, brown, or green plant lice) clustered on tender shoots, new leaf buds, or stem nodes.",
        "immediate_actions": [
            "Quarantine the plant away from clean foliage immediately.",
            "Blast visible aphid clusters off shoots and foliage using a firm stream of lukewarm water in the sink or shower.",
            "Spray foliage thoroughly with insecticidal soap, neem oil emulsion, or pyrethrin spray, ensuring complete coverage under leaves and along stems.",
            "Wipe away sticky honeydew residues to prevent secondary black sooty mold crust from developing.",
            "Reapply spray every 3-4 days for two weeks to eliminate newly hatched nymphs.",
        ],
    },
    "Thrips_Damage": {
        "condition_name": "Thrips Damage",
        "category": "Insect Infestation",
        "severity": "High",
        "summary": "Silvery rasped patches, mottled stippling, and fine dark frass speckling detected on surface.",
        "immediate_actions": [
            "Quarantine immediately and place blue sticky traps beside canopy.",
            "Spray Spinosad across upper and lower leaf surfaces every 5 days.",
        ],
    },
    "Whiteflies": {
        "condition_name": "Whiteflies",
        "category": "Insect Infestation",
        "severity": "Moderate to High",
        "summary": "Tiny white moth-like insects or translucent nymphs detected on foliage.",
        "immediate_actions": [
            "Place yellow sticky traps at leaf height and wash undersides with soapy water.",
            "Apply neem oil or insecticidal soap targeting nymphs under leaves.",
        ],
    },
    "Abiotic_Dehydration_Underwatering": {
        "condition_name": "Underwatering Dehydration Stress (Needs Water)",
        "category": "Abiotic Water Stress",
        "severity": "Moderate",
        "summary": "Foliage exhibits acute moisture deficit, loss of cellular turgor, leaf blade wrinkling, wild curled margins, and dry papery tips. These symptoms are physiological moisture deficit stress, NOT infectious fungal rust or insect bite damage.",
        "immediate_actions": [
            "Check soil moisture: Insert a finger or moisture meter 2 inches into soil. If dry and crumbly, water thoroughly immediately.",
            "Perform deep soaking: Water from above until water flows freely out of bottom drainage holes, or bottom-soak the pot in a basin of lukewarm water for 25-35 minutes.",
            "Do NOT apply chemical fungicides or pesticides; the plant is suffering from water deficit, not fungal rust or chewing insects.",
            "Prune only completely papery brown crispy margins using clean shears, leaving a thin dry buffer to protect healthy live tissue.",
            "Empty drainage saucer: Allow excess water to drain out completely so roots do not sit in stagnant pools.",
            "Avoid direct harsh sun: Keep the recovering plant in bright indirect light until foliage regains full turgor firmness (usually within 3-6 hours).",
            "Maintain regular watering schedule: Water whenever top 1-2 inches of soil feel dry to touch.",
            "Increase ambient humidity using a pebble tray or humidifier to slow down foliar transpiration."
        ],
        "treatment": {
            "organic": ["Bottom-watering deep soak and room humidity elevation with a pebble tray."],
            "chemical": ["No chemical pesticides or fungicides needed; chemical treatments can scorch dehydrated leaves."]
        },
        "prevention": [
            "Water thoroughly until water drains from the base whenever the top 2 inches of soil feel dry.",
            "Avoid placing plant directly in hot dry air conditioning drafts or heat vents."
        ],
        "symptoms": ["Leaf shriveling, wrinkled surface texture, curled wild margins, and crispy brown tip necrosis."],
        "causes": ["Extended dry soil period, low ambient humidity, or irregular watering intervals."]
    },
    "Abiotic_Overwatering_Edema": {
        "condition_name": "Overwatering / Edema Stress",
        "category": "Abiotic Water Stress",
        "severity": "Moderate",
        "summary": "Foliage exhibits symptoms of overwatering and root waterlogging (translucent water-soaked tissue, soggy chlorosis, or corky edema blisters). Roots are experiencing hypoxia from saturated potting substrate.",
        "immediate_actions": [
            "Cease watering immediately and allow the top 50% of the potting mix to dry out thoroughly.",
            "Empty any standing water pooled in the saucer and ensure bottom drainage holes are completely unblocked.",
            "Aerate potting soil by gently poking 4-5 holes around root perimeter with a chopstick to introduce root oxygen.",
            "Do NOT apply chemical fungicides or pesticides; the condition is physiological root suffocation, not an infection.",
            "Inspect roots for root rot (black/mushy tissue); trim dead roots and repot in fresh, well-draining mix if necessary."
        ],
        "treatment": {
            "organic": ["Soil aeration, drying root ball, and repotting in perlite-rich fast-draining substrate."],
            "chemical": ["No chemical fungicides needed; ensure soil oxygenation and moisture control."]
        },
        "prevention": [
            "Always check soil dryness 2 inches deep before watering.",
            "Use pots with functional drainage holes and well-aerated potting soil."
        ],
        "symptoms": ["Translucent water-soaked patches, diffuse leaf yellowing, soggy stems, and raised corky edema bumps."],
        "causes": ["Excessive watering frequency, poorly draining soil mix, or standing saucer water."]
    },
    "Abiotic_Sunburn_Scald": {
        "condition_name": "Solar Scorch / Sunburn",
        "category": "Abiotic Light Stress",
        "severity": "Mild to Moderate",
        "summary": "Bleached tan, white, or papery scorched patches detected on sun-facing foliage surfaces caused by excessive direct ultraviolet radiation and solar heat.",
        "immediate_actions": [
            "Move the plant immediately back from direct midday sun into bright filtered/indirect light.",
            "Do NOT prune mildly scorched leaves that still retain healthy green photosynthetic tissue.",
            "Maintain consistent root hydration to help plant foliage endure ambient heat."
        ],
        "treatment": {
            "organic": ["Relocate to filtered canopy light or install sheer curtains."],
            "chemical": ["No chemical intervention needed; sunburn is non-infectious radiation damage."]
        },
        "prevention": [
            "Acclimate plants gradually to increased light levels over 2-3 weeks.",
            "Avoid placing delicate foliage directly behind hot window glass in summer."
        ],
        "symptoms": ["Bleached white or tan dry surface patches with crisp dry borders."],
        "causes": ["Sudden exposure to intense direct sunlight, magnifying water droplets, or unshaded outdoor placement."]
    },
    "Abiotic_Nutrient_Chlorosis": {
        "condition_name": "Nutrient Chlorosis / Mineral Deficiency",
        "category": "Abiotic Nutritional Stress",
        "severity": "Mild to Moderate",
        "summary": "Interveinal yellowing (green veins with pale yellow leaf blade tissue) or chlorotic mottling indicating micronutrient (Iron/Magnesium/Nitrogen) deficiency or pH uptake lockout.",
        "immediate_actions": [
            "Check soil pH; nutrient lockout occurs if pH is too alkaline (> 7.0) or too acidic (< 5.5).",
            "Feed with a balanced half-strength liquid fertilizer containing chelated micronutrients (Fe, Mn, Mg).",
            "Flush potting mix with distilled water if mineral salts have accumulated in the soil."
        ],
        "treatment": {
            "organic": ["Chelated iron foliar spray and balanced organic liquid kelp/seaweed fertilizer."],
            "chemical": ["Apply balanced NPK fertilizer with complete micronutrient package at half strength."]
        },
        "prevention": [
            "Maintain seasonal feeding schedule during active spring/summer vegetative growth.",
            "Repot every 1-2 years to replenish degraded potting soil nutrients."
        ],
        "symptoms": ["Interveinal chlorosis (yellow tissue between dark green veins) and stunted shoot growth."],
        "causes": ["Soil nutrient depletion, high soil alkalinity locking out iron, or root root-bound root systems."]
    },
    "Abiotic_Severe_Desiccation_Dried_Plant": {
        "condition_name": "Severe Desiccation / Dried Plant Specimen",
        "category": "Abiotic Water Deficit",
        "severity": "High",
        "summary": "Specimen foliage and stems are severely dried, crispy, or withered from extreme moisture depletion.",
        "immediate_actions": [
            "Perform a stem scratch test: scrape a tiny section of the main stem near the base with a fingernail or clean blade. If you see moist green cambium tissue beneath the bark, the root crown is still alive and can recover.",
            "Emergency bottom-watering soak: place the pot into a basin with 2-3 inches of lukewarm water for 30-45 minutes so dry compacted soil can rehydrate from the bottom up, then drain completely.",
            "Prune fully dead crispy leaves: prune completely brittle, brown leaves that snap easily so the plant conserves water and energy for new viable growth.",
            "Provide recovery humidity: move the plant into bright indirect light away from hot direct sun, and place a clear bag or humidity tray nearby to reduce transpiration stress.",
            "Withhold all fertilizer: never fertilize a dehydrated or stressed root ball; wait until healthy new green shoots emerge.",
        ],
    },
    "Abiotic_Terminal_Dead_Plant": {
        "condition_name": "Irreversible Tissue Mortality / Non-Viable Dead Specimen",
        "category": "Terminal Tissue Necrosis",
        "severity": "Terminal (Beyond Saving)",
        "summary": "Specimen foliage and vascular tissue exhibit complete cellular collapse and irreversible necrosis. The sample is beyond recovery.",
        "immediate_actions": [
            "Perform a stem snap test: gently bend the lower stem. If it snaps cleanly like a brittle dry twig with no green interior cambium layer, the plant's vascular core is dead and cannot be revived.",
            "Single Leaf Assessment: If this photograph is of an individual fallen or trimmed leaf from an otherwise living plant, simply prune and discard the dead leaf to prevent saprophytic mold, and inspect the parent plant's main stem and root ball.",
            "Whole Plant Assessment: If the entire plant is brittle, dry, and lacks any green tissue or firm roots, safely discard the plant matter into compost or waste.",
            "Sanitize pot and soil: Do not reuse unsterilized soil from a dead plant. Wash containers thoroughly with a 10% bleach solution before potting new plants.",
            "Check neighboring plants: If plant death occurred suddenly, inspect surrounding plants for hidden root rot, soil pests, or systemic pathogen spread.",
        ],
    },
}


def run_v2_diagnosis(
    pil_img: Image.Image,
    secondary_img: Optional[Image.Image] = None,
    gemini_api_key: Optional[str] = None,
    soil_moisture_status: Optional[str] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    try:
        from PIL import ImageOps
        transposed = ImageOps.exif_transpose(pil_img)
        if transposed is not None:
            pil_img = transposed
    except Exception:
        pass
    prep_b64 = _img_to_b64(pil_img, max_size=512)

    # 1. Stage 0: Objective Image Quality Gate (global pre-check)
    q_ok, q_code, q_info = evaluate_image_quality(
        pil_img,
        min_short_side=QUALITY_THRESHOLDS["min_short_side_px"],
        min_laplacian_var=QUALITY_THRESHOLDS["min_laplacian_var"],
        min_tenengrad=QUALITY_THRESHOLDS["min_tenengrad_mean"],
        max_dark_pct=QUALITY_THRESHOLDS["max_dark_clipped_pct"],
        max_bright_pct=QUALITY_THRESHOLDS["max_bright_clipped_pct"],
    )
    if not q_ok:
        return {
            "image_quality_status": q_code,
            "species": {
                "name": "Image Quality Too Low",
                "confidence": 0.0,
                "visual_priors": {
                    "is_variegated": False,
                    "is_fenestrated": False,
                    "is_glossy": False,
                    "is_succulent": False,
                },
                "detected_mode": f"Stage 0 Quality Gate ({q_code})",
                "top_candidates": [],
            },
            "pdi_grade": 0,
            "overall_health_tier": "None",
            "affected_area_percent": 0.0,
            "needs_user_soil_check": False,
            "detections": [],
            "status": "retake_recommended",
            "grace_code": q_code,
            "is_plant": False,
            "disease_available": False,
            "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
            "status_message": q_info["reason"],
            "recommendations": q_info["recommendations"],
            "quality_metrics": q_info,
            "diagnosis": {"condition_name": "Retake Photo Recommended", "category": "Image Quality Issue", "is_healthy": True, "confidence": 0.0, "severity": "None"},
            "spots": {"has_spots": False, "spots_count": 0, "infected_area_pct": 0.0, "status_message": q_info["reason"], "annotated_image_base64": prep_b64},
            "care_guide": {"summary": q_info["reason"], "immediate_actions": q_info["recommendations"], "treatment": {"organic": [], "chemical": []}, "prevention": [], "symptoms": [], "causes": []},
            "predictions": [],
            "preprocessed_image_base64": prep_b64,
        }

    # Normalize illumination via CIELAB L* CLAHE to neutralize warm indoor LEDs and backlighting
    norm_img = normalize_illumination_lab(pil_img)

    # 2. Stage 1A: Multi-Tier Botanical Verification Gate (Online + Robust Local Semantic Verifier)
    is_plant, ver_msg, ver_meta = verify_plant_online(pil_img, explicit_api_key=gemini_api_key)
    if not is_plant:
        recs = [
            "Point your camera directly at a living plant, leaf cluster, cactus, or flower.",
            "Ensure the plant fills at least 35-60% of the photo.",
            "Avoid framing furniture, beds, walls, clothing, or electronics.",
        ]
        return {
            "image_quality_status": "PASS",
            "species": {
                "name": "No Plant Detected",
                "confidence": 0.0,
                "visual_priors": {
                    "is_variegated": False,
                    "is_fenestrated": False,
                    "is_glossy": False,
                    "is_succulent": False,
                },
                "family": "Non-Plant",
                "detected_mode": "Stage 1A Botanical Verification Gate",
                "top_candidates": [],
            },
            "pdi_grade": 0,
            "overall_health_tier": "None",
            "affected_area_percent": 0.0,
            "needs_user_soil_check": False,
            "detections": [],
            "status": "no_plant",
            "grace_code": "NO_PLANT_DETECTED",
            "is_plant": False,
            "disease_available": False,
            "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
            "status_message": ver_msg,
            "recommendations": recs,
            "verification_meta": ver_meta,
            "diagnosis": {"condition_name": "No Plant Detected", "category": "Non-Plant Subject", "is_healthy": True, "confidence": 0.0, "severity": "None"},
            "spots": {"has_spots": False, "spots_count": 0, "infected_area_pct": 0.0, "status_message": ver_msg, "annotated_image_base64": prep_b64},
            "care_guide": {
                "plant_name": "No Plant Detected",
                "condition_name": "No Plant Detected",
                "is_healthy": True,
                "severity": "None",
                "category": "Non-Plant Subject",
                "summary": ver_msg,
                "immediate_actions": recs,
                "treatment": {"organic": [], "chemical": []},
                "prevention": [],
                "symptoms": [ver_msg],
                "causes": []
            },
            "predictions": [],
            "preprocessed_image_base64": prep_b64,
        }

    # 3. Stage 1B: Dominant Multi-Plant Instance Segmentation (Highest Coverage Focus)
    iso = isolate_dominant_plant(norm_img, min_coverage_pct=QUALITY_THRESHOLDS["min_plant_coverage_pct"], allow_fallback=True)
    verified_pest = ver_meta.get("verified_pest_class")
    verified_pathology = ver_meta.get("verified_pathology_class")
    is_infestation = bool(ver_meta.get("is_infestation", False) or verified_pest is not None)
    is_verified_disease = bool(ver_meta.get("is_verified_disease", False) or verified_pathology is not None)
    is_underwatered_specimen = bool(ver_meta.get("is_underwatered", False))
    is_dried_specimen = bool(ver_meta.get("is_dried", False)) and not is_infestation and not is_verified_disease
    is_dead_specimen = bool(ver_meta.get("is_dead_beyond_saving", False)) and not is_infestation and not is_verified_disease

    if not iso["detected"] or iso["dominant_coverage_pct"] < QUALITY_THRESHOLDS["min_plant_coverage_pct"]:
        if is_plant or is_dried_specimen or is_dead_specimen or is_infestation or is_underwatered_specimen or is_verified_disease:
            # Fall back to using the full photo for dried, dead, infested, or irregular botanical subjects
            iso["crop_img"] = pil_img
            iso["masked_crop_img"] = pil_img
            iso["unmasked_crop_img"] = pil_img
            iso["mask_crop"] = np.ones((pil_img.height, pil_img.width), dtype=bool)
            iso["full_mask"] = iso["mask_crop"]
            iso["dominant_coverage_pct"] = 100.0
            iso["dominant_conf"] = 0.50
            iso["detected"] = True
            iso["instances_count"] = 1
            iso["mode"] = "verified_plant_full_frame"
        else:
            cov = iso["dominant_coverage_pct"]
            msg = f"Plant subject is too small or unclear in the frame ({cov:.1f}% coverage)."
            recs = [
                "Move the camera closer so the target plant fills at least 35-60% of the photo.",
                "Keep pots, desks, and room walls minimized around the edges.",
            ]
            return {
                "status": "retake_recommended",
                "grace_code": "LOW_COVERAGE",
                "is_plant": True,
                "disease_available": False,
                "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
                "status_message": msg,
                "recommendations": recs,
                "isolation_meta": {"instances_count": iso["instances_count"], "dominant_coverage_pct": cov},
                "species": {"name": "Move Closer to Plant", "confidence": 0.0, "detected_mode": "Stage 1B Coverage Gate", "top_candidates": []},
                "diagnosis": {"condition_name": "Subject Too Far", "category": "Framing Issue", "is_healthy": True, "confidence": 0.0, "severity": "None"},
                "spots": {"has_spots": False, "spots_count": 0, "infected_area_pct": 0.0, "status_message": msg, "annotated_image_base64": _img_to_b64(iso["overlay_img"])},
                "care_guide": {"summary": msg, "immediate_actions": recs, "treatment": {"organic": [], "chemical": []}, "prevention": [], "symptoms": [], "causes": []},
                "predictions": [],
                "preprocessed_image_base64": prep_b64,
            }

    # Re-run sharp focus verification strictly on the isolated dominant plant pixels
    fg_ok, fg_code, fg_q = evaluate_image_quality(iso["crop_img"], foreground_mask=iso["mask_crop"], min_short_side=120)
    if not fg_ok and not is_dried_specimen and not is_dead_specimen and not is_infestation and not is_underwatered_specimen and not is_verified_disease:
        return {
            "status": "retake_recommended",
            "grace_code": f"SUBJECT_{fg_code}",
            "is_plant": True,
            "disease_available": False,
            "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
            "status_message": fg_q["reason"],
            "recommendations": fg_q["recommendations"],
            "species": {"name": "Retake Clearer Photo", "confidence": 0.0, "detected_mode": f"Foreground Quality Check ({fg_code})", "top_candidates": []},
            "diagnosis": {"condition_name": "Plant Surface Unclear", "category": "Image Quality Issue", "is_healthy": True, "confidence": 0.0, "severity": "None"},
            "spots": {"has_spots": False, "spots_count": 0, "infected_area_pct": 0.0, "status_message": fg_q["reason"], "annotated_image_base64": _img_to_b64(iso["overlay_img"])},
            "care_guide": {"summary": fg_q["reason"], "immediate_actions": fg_q["recommendations"], "treatment": {"organic": [], "chemical": []}, "prevention": [], "symptoms": [], "causes": []},
            "predictions": [],
            "preprocessed_image_base64": _img_to_b64(iso["crop_img"]),
        }

    # 4. Stage 2: Foundation Botanical Identification (Loose Unmasked Crop + Multi-Photo Averaging)
    loose_crop = iso.get("loose_plant_crop", iso.get("unmasked_crop_img", iso["crop_img"]))
    stage2_crops = [loose_crop]

    if secondary_img is not None:
        try:
            from PIL import ImageOps
            sec_trans = ImageOps.exif_transpose(secondary_img)
            if sec_trans is not None:
                secondary_img = sec_trans
            norm_sec = normalize_illumination_lab(secondary_img)
            sec_iso = isolate_dominant_plant(norm_sec, allow_fallback=True)
            sec_loose = sec_iso.get("loose_plant_crop", sec_iso.get("unmasked_crop_img", secondary_img))
            stage2_crops.append(sec_loose)
        except Exception as se:
            print(f"[Warning] Secondary image processing failed: {se}")

    id_res = identify_dominant_plant(stage2_crops if len(stage2_crops) > 1 else loose_crop, top_k=5)
    if id_res.get("species") == "No Plant Detected" or id_res.get("status") == "NON_PLANT_OOD":
        recs = [
            "Point your camera directly at living plant foliage, leaf cluster, or flower.",
            "Ensure the plant fills at least 35-60% of the photo.",
            "Avoid framing furniture, beds, walls, clothing, or electronics.",
        ]
        msg = "No recognizable botanical structures detected. Please photograph a living plant, leaf, or flower."
        return {
            "status": "no_plant",
            "grace_code": "NO_PLANT_DETECTED",
            "is_plant": False,
            "disease_available": False,
            "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
            "status_message": msg,
            "recommendations": recs,
            "verification_meta": {"is_plant": False, "source": "Stage 2 Botanical Out-of-Distribution Gate"},
            "species": {"name": "No Plant Detected", "confidence": 0.0, "family": "Non-Plant", "detected_mode": "Stage 2 Botanical Gate", "top_candidates": []},
            "top_candidates": [],
            "needs_user_confirmation": False,
            "diagnosis": {"condition_name": "No Plant Detected", "category": "Non-Plant Subject", "is_healthy": True, "confidence": 0.0, "severity": "None"},
            "spots": {"has_spots": False, "spots_count": 0, "infected_area_pct": 0.0, "status_message": msg, "annotated_image_base64": prep_b64},
            "care_guide": {
                "plant_name": "No Plant Detected",
                "condition_name": "No Plant Detected",
                "is_healthy": True,
                "severity": "None",
                "category": "Non-Plant Subject",
                "summary": msg,
                "immediate_actions": recs,
                "treatment": {"organic": [], "chemical": []},
                "prevention": [],
                "symptoms": [msg],
                "causes": []
            },
            "predictions": [],
            "preprocessed_image_base64": prep_b64,
        }

    top1_conf = float(id_res.get("species_confidence", id_res.get("confidence", 0.0)))
    top_cands = list(id_res.get("candidates", []))
    reranked_by_vlm = False
    vlm_reason = ""

    # Conditional Gemini Vision Re-Ranker: Only triggers when Stage 2 Top-1 confidence is < 0.80 (< 80.0%)
    if top1_conf < 80.0 and len(top_cands) >= 2 and is_plant:
        try:
            rerank_res = rerank_species_with_gemini(pil_img, top_cands[:5], explicit_api_key=gemini_api_key)
            sel_species = rerank_res.get("selected_species", "UNCERTAIN")
            if sel_species and sel_species != "UNCERTAIN":
                match_idx = None
                for idx, c in enumerate(top_cands):
                    c_name = c.get("species_name", c.get("species", ""))
                    if c_name.lower() == sel_species.lower() or sel_species.lower() in c_name.lower() or c_name.lower() in sel_species.lower():
                        match_idx = idx
                        break
                if match_idx is not None and match_idx > 0:
                    promoted = top_cands.pop(match_idx)
                    top_cands.insert(0, promoted)
                    for r, c in enumerate(top_cands, 1):
                        c["rank"] = r
                    reranked_by_vlm = True
                    vlm_reason = rerank_res.get("reason", "")
                    id_res["raw_top_species"] = promoted["species_name"]
                    id_res["species"] = promoted["species_name"]
                    id_res["genus"] = promoted.get("genus", extract_genus_from_species(promoted["species_name"]))
                    id_res["family"] = promoted.get("family", id_res.get("family"))
                    id_res["candidates"] = top_cands
                    id_res["top_candidates"] = top_cands
                    id_res["status"] = "VLM_RERANKED"
                elif match_idx == 0:
                    reranked_by_vlm = True
                    vlm_reason = rerank_res.get("reason", "")
        except Exception as re_err:
            print(f"[Gemini Re-Ranker Warning] {re_err}")

    # Top-3 candidates formatted for response & UI confirmation
    top_candidates_3 = []
    for c in top_cands[:3]:
        spec_n = c.get("species_name", c.get("species", ""))
        fam_n = c.get("family", "")
        top_candidates_3.append({
            "species_name": spec_n,
            "species": spec_n,
            "genus": c.get("genus", extract_genus_from_species(spec_n)),
            "family": fam_n,
            "display_name": c.get("display_name", spec_n),
            "confidence": c.get("confidence", 0.0),
            "visual_priors": c.get("visual_priors", get_species_visual_priors(spec_n, fam_n)),
        })

    needs_user_confirmation = bool(top1_conf < 85.0)

    # 5. Stage 3: Unified Symptom Detection & Surface Pathology
    cand_str = " ".join(c.get("species", "").lower() for c in id_res.get("candidates", [])[:5])
    fam_tax = f"{id_res.get('species', '')} {id_res.get('family', '')} {cand_str}".strip()
    visual_priors = id_res.get("visual_priors", {})

    species_conf = float(id_res.get("confidence", 0.0))
    if species_conf > 1.0:
        species_conf /= 100.0

    path_res = diagnose_surface_condition(
        iso.get("unmasked_crop_img", iso["crop_img"]),
        iso["mask_crop"],
        plant_family=fam_tax,
        species_name=id_res.get("species", "Botanical Specimen"),
        species_conf=species_conf,
        visual_priors=visual_priors,
        is_externally_dried=is_dried_specimen,
        is_externally_dead=is_dead_specimen,
        is_externally_underwatered=is_underwatered_specimen,
        verified_pest=verified_pest,
        verified_pathology=verified_pathology,
    )
    primary_cond = path_res["primary_class"]
    care = SURFACE_CARE_MAP.get(primary_cond, SURFACE_CARE_MAP["Healthy_Foliage"])
    lesion_info = path_res["lesion_info"]
    detections = path_res.get("detections", [])
    affected_area_percent = float(path_res.get("affected_area_percent", 0.0))

    # 6. Stage 4: Symptom-Based Triage & Decoupled Soil Moisture Engine
    sms = soil_moisture_status.lower().strip() if soil_moisture_status else None

    # Check Stage 1A whole-plant visual verification cues
    is_wilting_or_drooping = bool(ver_meta.get("is_wilting_or_drooping", False))
    has_dried_or_crispy_leaves = bool(ver_meta.get("has_dried_or_crispy_leaves", False))

    # Fix 2: Piggyback Whole-Plant Wilting & Dried-Leaf Check from Stage 1A
    has_crispy = any(d["class"] == "crispy_brown_edges" for d in detections)
    if has_dried_or_crispy_leaves and not has_crispy:
        crop_w, crop_h = iso.get("unmasked_crop_img", iso["crop_img"]).size
        detections.append({
            "class": "crispy_brown_edges",
            "display_label": "Crispy Brown Edges & Dried Foliage",
            "confidence": 0.88,
            "confidence_tier": "high",
            "source": "whole_plant_check",
            "bbox": [0, 0, crop_w, crop_h],
            "probable_causes": [
                "Low indoor humidity (AC / dry air exposure)",
                "Underwatering / Irregular watering intervals",
                "Tap water mineral / fluoride salt sensitivity",
            ],
            "recommended_check": "Touch the top 2 inches of soil: Is it dry, damp, or soggy?",
        })
        has_crispy = True
        if path_res.get("is_healthy", False) or primary_cond == "Healthy_Foliage":
            primary_cond = "Abiotic_Dehydration_Underwatering"
            care = SURFACE_CARE_MAP.get(primary_cond, SURFACE_CARE_MAP["Healthy_Foliage"])
            path_res["is_healthy"] = False
            path_res["super_state"] = "Abiotic_Stress"
        if affected_area_percent <= 0.0:
            affected_area_percent = 8.5

    has_wilting = any(d["class"] in ("foliar_wilting_droop", "wilting_drooping") for d in detections)
    if is_wilting_or_drooping and not has_wilting:
        crop_w, crop_h = iso.get("unmasked_crop_img", iso["crop_img"]).size
        detections.append({
            "class": "foliar_wilting_droop",
            "display_label": "Foliar Wilting & Drooping Foliage",
            "confidence": 0.85,
            "confidence_tier": "high",
            "source": "whole_plant_check",
            "bbox": [0, 0, crop_w, crop_h],
            "probable_causes": [
                "Underwatering (if soil feels dry and crumbly)",
                "Overwatering / Root rot (if soil is saturated/soggy)",
            ],
            "recommended_check": "Touch the top 2 inches of soil: Is it dry, damp, or soggy?",
        })
        has_wilting = True
        if path_res.get("is_healthy", False) or primary_cond == "Healthy_Foliage":
            primary_cond = "Abiotic_Dehydration_Underwatering"
            care = SURFACE_CARE_MAP.get(primary_cond, SURFACE_CARE_MAP["Healthy_Foliage"])
            path_res["is_healthy"] = False
            path_res["super_state"] = "Abiotic_Stress"
        if affected_area_percent <= 0.0:
            affected_area_percent = 6.0

    has_chlorosis = any(d["class"] == "yellowing_chlorosis" for d in detections)

    if (has_chlorosis or has_wilting or has_crispy) and sms == "soggy":
        for d in detections:
            if d["class"] in ("yellowing_chlorosis", "foliar_wilting_droop", "crispy_brown_edges"):
                d["probable_causes"] = [
                    "Overwatering / Root Rot (Substrate waterlogged & soggy)",
                    "Anaerobic root hypoxia and suffocation",
                    "Nutrient lockout from compromised waterlogged roots",
                ]
                d["recommended_check"] = "Pause watering immediately, check pot drainage, and allow potting mix to dry out."
    elif (has_crispy or has_chlorosis or has_wilting) and sms == "dry":
        for d in detections:
            if d["class"] in ("crispy_brown_edges", "yellowing_chlorosis", "foliar_wilting_droop"):
                d["probable_causes"] = [
                    "Underwatering / Low Humidity (Substrate completely dry)",
                    "Root-ball dehydration and acute water deficit",
                    "High ambient transpiration and dry air",
                ]
                d["recommended_check"] = "Water thoroughly until water drains from the bottom; consider a 30-min bottom soak."
    elif (has_crispy or has_chlorosis or has_wilting) and sms == "damp":
        for d in detections:
            if d["class"] in ("crispy_brown_edges", "yellowing_chlorosis", "foliar_wilting_droop"):
                d["probable_causes"] = [
                    "Soil moisture optimal (Check indoor humidity, light levels, or natural leaf aging)",
                    "Tap water mineral / fluoride salt sensitivity",
                    "Recent environmental acclimation",
                ]
                d["recommended_check"] = "Maintain balanced watering cadence; monitor humidity with a hygrometer."

    needs_user_soil_check = bool(
        sms is None
        and not (primary_cond in ("Scale_Insects", "Mealybugs_Cochineal") or path_res.get("surface_stress", {}).get("has_white_colonies") or verified_pest is not None)
        and (has_chlorosis or has_crispy or has_wilting or is_wilting_or_drooping or has_dried_or_crispy_leaves)
    )

    # PDI Grade calculation
    if affected_area_percent == 0.0 or path_res.get("is_healthy", False):
        pdi_grade = 0
        overall_health_tier = "Optimal"
    elif affected_area_percent < 2.0:
        pdi_grade = 1
        overall_health_tier = "Trace"
    elif affected_area_percent <= 10.0:
        pdi_grade = 2
        overall_health_tier = "Mild"
    elif affected_area_percent <= 25.0:
        pdi_grade = 3
        overall_health_tier = "Moderate"
    else:
        pdi_grade = 4
        overall_health_tier = "Severe"

    is_pest = bool(
        verified_pest
        or primary_cond in (
            "Mealybugs_Cochineal",
            "Aphids",
            "Scale_Insects",
            "Thrips_Damage",
            "Spider_Mites",
            "Whiteflies",
            "Leaf_Miners",
            "Chewing_Pests_Bug_Bites_Holes",
            "Rust_Fungi_With_Bug_Bites",
        )
    )
    is_dead = bool((is_dead_specimen or path_res.get("is_dead", False) or primary_cond == "Abiotic_Terminal_Dead_Plant") and not is_pest and not is_verified_disease)
    is_dried = bool((is_dead or is_dried_specimen or primary_cond == "Abiotic_Severe_Desiccation_Dried_Plant") and not is_pest and not is_verified_disease)
    is_underwatered = bool((is_underwatered_specimen or primary_cond == "Abiotic_Dehydration_Underwatering") and not is_pest and not is_dried and not is_dead and not is_verified_disease)

    is_healthy_leaf = primary_cond == "Healthy_Foliage"

    has_bug_bites = bool(
        ver_meta.get("has_bug_bites_holes", False)
        or primary_cond in ("Chewing_Pests_Bug_Bites_Holes", "Rust_Fungi_With_Bug_Bites")
        or path_res["surface_stress"].get("has_optical_bite_holes", False)
        or any(d["class"] == "chewed_holes_stippling" for d in detections)
    )
    if has_bug_bites and (primary_cond == "Healthy_Foliage" or path_res.get("is_healthy", False)):
        primary_cond = "Chewing_Pests_Bug_Bites_Holes"
        care = SURFACE_CARE_MAP.get(primary_cond, SURFACE_CARE_MAP["Healthy_Foliage"])
        path_res["is_healthy"] = False
        path_res["super_state"] = "Biotic_Pathology"
        is_healthy_leaf = False

    has_rust = bool(
        ver_meta.get("has_rusting", False)
        or primary_cond in ("Rust_Fungi", "Rust_Fungi_With_Bug_Bites")
        or path_res["surface_stress"].get("has_rust_pustules", False)
    )

    target_species = id_res.get("species") or id_res.get("raw_top_species", "Botanical Specimen")
    species_name = target_species

    multi_note = (
        f"Detected {iso['instances_count']} plants; focused exclusively on largest foreground plant ({iso['dominant_coverage_pct']}% frame coverage)."
        if iso["instances_count"] > 1
        else f"Primary plant isolated cleanly ({iso['dominant_coverage_pct']}% frame coverage)."
    )
    if primary_cond == "Rust_Fungi_With_Bug_Bites":
        multi_note = f"{multi_note} Co-occurring foliar rust pustules and insect chew holes actively verified."
    elif primary_cond == "Chewing_Pests_Bug_Bites_Holes":
        multi_note = f"{multi_note} Active insect chewing damage and bug bite holes verified on foliage."
    elif is_pest and verified_pest:
        multi_note = f"{multi_note} Active pest infestation verified: {care['condition_name']}."
    elif primary_cond == "Rust_Fungi":
        multi_note = f"{multi_note} Fungal rust pustules verified on leaf surface (Pucciniales)."
    elif is_dead:
        multi_note = f"{multi_note} Specimen exhibits irreversible cellular mortality (beyond saving)."
    elif is_dried:
        multi_note = f"{multi_note} Specimen exhibits severe desiccation / dried foliage."
    elif is_underwatered:
        multi_note = f"{multi_note} Specimen exhibits foliar dehydration / loss of turgor (needs water)."

    all_recs = list(care["immediate_actions"]) + [r for r in id_res.get("recommendations", []) if r not in care["immediate_actions"]]

    if primary_cond == "Rust_Fungi_With_Bug_Bites":
        pustule_pct = path_res["surface_stress"].get("rust_pustule_pct", 0.0)
        holes_cnt = max(1, path_res["surface_stress"].get("bite_holes_count", 1))
        spots_status_msg = f"Dual damage: {holes_cnt} insect bite hole(s) and {pustule_pct}% fungal rust pustule coverage detected on foliage."
    elif primary_cond == "Chewing_Pests_Bug_Bites_Holes":
        holes_cnt = max(1, path_res["surface_stress"].get("bite_holes_count", 1))
        holes_pct = path_res["surface_stress"].get("bite_holes_area_pct", 0.0)
        spots_status_msg = f"{holes_cnt} insect chew hole(s) / bitten perforation zone(s) detected on foliage ({holes_pct}% area loss)."
    elif is_pest and verified_pest:
        spots_status_msg = f"Active biotic pest infestation detected: {care['condition_name']}. Inspect leaf nodes, axils, and leaf undersides immediately."
    elif primary_cond == "Rust_Fungi":
        pustule_pct = path_res["surface_stress"].get("rust_pustule_pct", 0.0)
        spots_status_msg = f"Raised orange/rusty-cinnamon fungal pustules (uredinia) detected on foliage ({pustule_pct}% spore pustule coverage). Rust Fungi (Pucciniales)."
    elif primary_cond == "Abiotic_Terminal_Dead_Plant":
        spots_status_msg = f"Specimen exhibits irreversible cellular mortality ({path_res['surface_stress'].get('desiccation_grade', 'Dead')}). Zero living green tissue."
    elif primary_cond == "Abiotic_Severe_Desiccation_Dried_Plant":
        spots_status_msg = f"Specimen exhibits severe foliar desiccation ({path_res['surface_stress'].get('desiccation_grade', 'Dried')}). Foliage is dried and moisture-depleted."
    elif primary_cond == "Abiotic_Dehydration_Underwatering":
        spots_status_msg = "Specimen exhibits acute foliar dehydration and wilting (Needs Water). Living leaves are drooping."
    elif lesion_info["spots_count"] > 0:
        spots_status_msg = f"{lesion_info['spots_count']} surface lesion/symptom zone(s) detected ({lesion_info['infected_area_pct']}% of plant mask)."
    else:
        spots_status_msg = "No distinct localized lesion boxes detected."

    co_alerts = []
    if has_bug_bites and has_rust:
        co_alerts.append({
            "type": "Co-Occurring Foliar Alert",
            "badge": "Bug Bites & Rusting",
            "message": "Foliage exhibits BOTH insect chewing bite holes and fungal rust spore pustules. Apply dual protocol: spray Bt/neem for chewing pests and copper/sulfur fungicide for rust.",
        })
    elif has_bug_bites:
        co_alerts.append({
            "type": "Insect Feeding Alert",
            "badge": "Bug Bites & Chewed Holes",
            "message": "Chewed holes, shot-hole punctures, or missing leaf blade sections detected. Inspect leaf undersides at night for caterpillars, beetles, or slugs and apply Bt or neem oil.",
        })
    elif has_rust:
        pustule_pct = path_res["surface_stress"].get("rust_pustule_pct", 0.0)
        co_alerts.append({
            "type": "Fungal Spore Alert",
            "badge": "Rust Fungi (Pucciniales)",
            "message": f"Raised orange/rusty-cinnamon fungal pustules (uredinia) detected on foliage ({pustule_pct}% spore pustule coverage). Prune infected leaves and apply sulfur or copper fungicide.",
        })

    foliar_damage_indicators = {
        "has_bug_bite_holes": has_bug_bites,
        "has_rusting": has_rust,
        "bug_bite_holes_severity": "Severe" if (path_res["surface_stress"].get("bite_holes_count", 0) > 20) else (ver_meta.get("bug_bite_holes_severity") if ver_meta.get("bug_bite_holes_severity") not in (None, "None") else ("Moderate" if has_bug_bites else "None")),
        "rusting_severity": ver_meta.get("rust_severity", "Moderate" if has_rust else "None"),
        "spore_pustule_pct": path_res["surface_stress"].get("rust_pustule_pct", 0.0),
        "bite_holes_count": path_res["surface_stress"].get("bite_holes_count", 1 if has_bug_bites else 0),
        "status_summary": (
            "Dual condition: Foliar bug bite holes AND fungal rust pustules detected on foliage."
            if (has_bug_bites and has_rust)
            else (
                "Foliar bug bite holes / insect chewing damage detected on foliage."
                if has_bug_bites
                else (
                    "Fungal rust pustules (Pucciniales) detected on foliage."
                    if has_rust
                    else "No bug bite holes or fungal rust pustules detected."
                )
            )
        ),
    }

    return {
        "image_quality_status": "PASS",
        "pdi_grade": pdi_grade,
        "overall_health_tier": overall_health_tier,
        "affected_area_percent": affected_area_percent,
        "needs_user_soil_check": needs_user_soil_check,
        "detections": detections,
        "status": "success",
        "grace_code": id_res["status"],
        "is_plant": True,
        "disease_available": True,
        "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
        "status_message": f"{multi_note} Surface assessment: {care['condition_name']}.",
        "recommendations": all_recs,
        "isolation_meta": {
            "instances_count": iso["instances_count"],
            "dominant_coverage_pct": iso["dominant_coverage_pct"],
            "dominant_box": iso["dominant_box"],
            "mode": iso["mode"],
        },
        "species": {
            "name": species_name,
            "confidence": round(float(id_res["confidence"]) / 100.0 if float(id_res["confidence"]) > 1.0 else float(id_res["confidence"]), 2),
            "visual_priors": visual_priors,
            "raw_top_species": id_res.get("raw_top_species"),
            "genus": id_res.get("genus"),
            "family": id_res.get("family"),
            "display_name": id_res.get("species", species_name),
            "detected_mode": f"CoreTech v2 Ensemble ({id_res['status']})",
            "top_candidates": top_candidates_3,
            "reranked_by_vlm": reranked_by_vlm,
            "vlm_reason": vlm_reason,
        },
        "top_candidates": top_candidates_3,
        "needs_user_confirmation": needs_user_confirmation,
        "diagnosis": {
            "condition_name": care["condition_name"],
            "class_id": primary_cond,
            "super_state": path_res["super_state"],
            "super_probs": path_res.get("super_probs"),
            "category": care["category"],
            "is_healthy": path_res["is_healthy"],
            "confidence": path_res["confidence"],
            "severity": care["severity"],
        },
        "spots": {
            "has_spots": (
                (not path_res["is_healthy"])
                and (
                    lesion_info["spots_count"] > 0
                    or primary_cond in ("Rust_Fungi", "Rust_Fungi_With_Bug_Bites", "Chewing_Pests_Bug_Bites_Holes")
                    or has_bug_bites
                    or has_rust
                )
                and not is_underwatered
                and not is_dead
                and not is_dried
            ),
            "spots_count": max(
                lesion_info["spots_count"],
                1 if (primary_cond in ("Rust_Fungi", "Rust_Fungi_With_Bug_Bites", "Chewing_Pests_Bug_Bites_Holes") or has_bug_bites or has_rust) else 0
            ) if (not is_underwatered and not is_dead and not is_dried) else 0,
            "infected_area_pct": max(
                lesion_info["infected_area_pct"],
                path_res["surface_stress"].get("rust_pustule_pct", 0.0) if has_rust else 0.0,
                path_res["surface_stress"].get("bite_holes_area_pct", 0.0) if has_bug_bites else 0.0,
            ) if (not is_underwatered and not is_dead and not is_dried) else 0.0,
            "status_message": spots_status_msg,
            "annotated_image_base64": _img_to_b64(lesion_info["annotated_img"]),
        },
        "co_alerts": co_alerts,
        "foliar_damage_indicators": foliar_damage_indicators,
        "surface_stress": path_res["surface_stress"],
        "xai_heatmap_base64": path_res.get("xai_heatmap_base64", ""),
        "tile_attention_pct": path_res.get("tile_attention_pct", []),
        "host_compatibility": path_res.get("host_compatibility", {}),
        "tta_meta": path_res.get("tta_meta", {}),
        "care_guide": {
            "summary": care["summary"],
            "immediate_actions": all_recs,
            "treatment": {"organic": [], "chemical": []},
            "prevention": [],
            "symptoms": [care["summary"]],
            "causes": (
                ["Complete vascular collapse", "Severe irreversible dehydration", "Total chlorophyll loss", "Terminal root desiccation or rot"]
                if (primary_cond == "Abiotic_Terminal_Dead_Plant")
                else (
                    ["Severe moisture depletion", "Prolonged underwatering", "Root-ball dehydration", "Excess heat desiccation"]
                    if (primary_cond == "Abiotic_Severe_Desiccation_Dried_Plant")
                    else (
                        ["Inadequate watering volume or frequency", "Root zone moisture deficit", "Hydrophobic dry soil", "High ambient heat / transpiration"]
                        if (primary_cond == "Abiotic_Dehydration_Underwatering")
                        else (
                            ["Infestation by Pseudococcidae sap-sucking insects", "Warm stagnant indoor environment", "Contaminated nursery stock or soil"]
                            if (primary_cond == "Mealybugs_Cochineal")
                            else (
                                ["Aphidoidea sap-sucker colonies (black aphids / plant lice)", "Tender new vegetative shoots attracting pests", "Lack of natural predatory insects"]
                                if (primary_cond == "Aphids")
                                else (
                                    ["Chewing insect pests (caterpillars, beetles, grasshoppers, leaf-cutter bees, or snails/slugs)", "Outdoor exposure or open greenhouse vents", "Tender foliage attracting defoliating insects"]
                                    if (primary_cond in ("Chewing_Pests_Bug_Bites_Holes", "Rust_Fungi_With_Bug_Bites"))
                                    else (
                                        ["Pucciniales airborne fungal spore infection", "Persistent water droplets or high relative humidity on foliage", "Airborne spore transmission from infected neighboring plants"]
                                        if (primary_cond == "Rust_Fungi")
                                        else []
                                    )
                                )
                            )
                        )
                    )
                )
            ),
        },
        "predictions": path_res["predictions"],
        "preprocessed_image_base64": _img_to_b64(iso["crop_img"]),
    }
