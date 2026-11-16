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
sys.path.insert(0, str(V2_ROOT))

from configs.v2_config import QUALITY_THRESHOLDS
try:
    from .quality_gate import evaluate_image_quality
    from .plant_verifier import verify_plant_online
    from .dominant_isolator import isolate_dominant_plant
    from .botanical_ensemble import identify_dominant_plant
    from .surface_pathology import diagnose_surface_condition
except (ImportError, ValueError):
    from src.quality_gate import evaluate_image_quality
    from src.plant_verifier import verify_plant_online
    from src.dominant_isolator import isolate_dominant_plant
    from src.botanical_ensemble import identify_dominant_plant
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
        "summary": "Foliage exhibits acute moisture deficit and loss of cellular turgor (limp, drooping, wilting, or inward-curling green leaves). The root zone is dry and the plant urgently needs water.",
        "immediate_actions": [
            "Check soil moisture: Insert a finger or moisture meter 2 inches into soil. If dry and crumbly, water thoroughly immediately.",
            "Perform deep soaking: Water from above until water flows freely out of bottom drainage holes, or bottom-soak the pot in a basin of lukewarm water for 25-35 minutes.",
            "Empty drainage saucer: Allow excess water to drain out completely so roots do not sit in stagnant pools.",
            "Avoid direct harsh sun: Keep the recovering plant in bright indirect light until foliage regains full turgor firmness (usually within 3-6 hours).",
            "Maintain regular watering schedule: Water whenever top 1-2 inches of soil feel dry to touch.",
        ],
    },
    "Abiotic_Overwatering_Edema": {
        "condition_name": "Overwatering / Edema Stress",
        "category": "Abiotic Water Stress",
        "severity": "Moderate",
        "summary": "Surface shows translucent water-soaked swelling/blisters or soggy yellowing from excess root moisture.",
        "immediate_actions": [
            "Pause watering until the top half of the potting mix dries out.",
            "Empty standing saucer water and verify pot drainage holes are unblocked.",
        ],
    },
    "Abiotic_Sunburn_Scald": {
        "condition_name": "Solar Scorch / Sunburn",
        "category": "Abiotic Light Stress",
        "severity": "Mild to Moderate",
        "summary": "Bleached white, tan, or scorched surface patches from excessive direct UV/light exposure.",
        "immediate_actions": [
            "Move the plant slightly back from intense midday direct sun into bright filtered light.",
            "Leave mildly scorched leaves attached while green tissue remains photosynthetically active.",
        ],
    },
    "Abiotic_Nutrient_Chlorosis": {
        "condition_name": "Nutrient Chlorosis / Mineral Deficiency",
        "category": "Abiotic Nutritional Stress",
        "severity": "Mild to Moderate",
        "summary": "Interveinal yellowing or chlorotic surface mottling indicating mineral uptake imbalance.",
        "immediate_actions": [
            "Check soil pH and feed with a balanced half-strength liquid fertilizer containing chelated micronutrients (Fe, Mn, Mg).",
        ],
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
    pil_img: Image.Image, gemini_api_key: Optional[str] = None
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
            "status": "retake_recommended",
            "grace_code": q_code,
            "is_plant": False,
            "disease_available": False,
            "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
            "status_message": q_info["reason"],
            "recommendations": q_info["recommendations"],
            "quality_metrics": q_info,
            "species": {"name": "Image Quality Too Low", "confidence": 0.0, "detected_mode": f"Stage 0 Quality Gate ({q_code})", "top_candidates": []},
            "diagnosis": {"condition_name": "Retake Photo Recommended", "category": "Image Quality Issue", "is_healthy": True, "confidence": 0.0, "severity": "None"},
            "spots": {"has_spots": False, "spots_count": 0, "infected_area_pct": 0.0, "status_message": q_info["reason"], "annotated_image_base64": prep_b64},
            "care_guide": {"summary": q_info["reason"], "immediate_actions": q_info["recommendations"], "treatment": {"organic": [], "chemical": []}, "prevention": [], "symptoms": [], "causes": []},
            "predictions": [],
            "preprocessed_image_base64": prep_b64,
        }

    # 2. Stage 1A: 100% Online Semantic Non-Plant Gate
    is_plant, ver_msg, ver_meta = verify_plant_online(pil_img, explicit_api_key=gemini_api_key)
    if is_plant is False:
        recs = [
            "Point your camera directly at a living plant, leaf cluster, cactus, or flower.",
            "Ensure no artificial/plastic plant or furniture fills the main frame.",
        ]
        return {
            "status": "no_plant",
            "grace_code": "NO_PLANT_DETECTED",
            "is_plant": False,
            "disease_available": False,
            "inference_time_ms": round((time.time() - t0) * 1000.0, 1),
            "status_message": ver_msg,
            "recommendations": recs,
            "verification_meta": ver_meta,
            "species": {"name": "No Plant Detected", "confidence": 0.0, "detected_mode": "Stage 1A Online Semantic Gate", "top_candidates": []},
            "diagnosis": {"condition_name": "No Plant Detected", "category": "Non-Plant Subject", "is_healthy": True, "confidence": 0.0, "severity": "None"},
            "spots": {"has_spots": False, "spots_count": 0, "infected_area_pct": 0.0, "status_message": ver_msg, "annotated_image_base64": prep_b64},
            "care_guide": {"summary": ver_msg, "immediate_actions": recs, "treatment": {"organic": [], "chemical": []}, "prevention": [], "symptoms": [ver_msg], "causes": []},
            "predictions": [],
            "preprocessed_image_base64": prep_b64,
        }

    # 3. Stage 1B: Dominant Multi-Plant Instance Segmentation (Highest Coverage Focus)
    iso = isolate_dominant_plant(pil_img, min_coverage_pct=QUALITY_THRESHOLDS["min_plant_coverage_pct"], allow_fallback=True)
    verified_pest = ver_meta.get("verified_pest_class")
    is_infestation = bool(ver_meta.get("is_infestation", False) or verified_pest is not None)
    is_underwatered_specimen = bool(ver_meta.get("is_underwatered", False))
    is_dried_specimen = bool(ver_meta.get("is_dried", False)) and not is_infestation
    is_dead_specimen = bool(ver_meta.get("is_dead_beyond_saving", False)) and not is_infestation

    if not iso["detected"] or iso["dominant_coverage_pct"] < QUALITY_THRESHOLDS["min_plant_coverage_pct"]:
        if is_plant or is_dried_specimen or is_dead_specimen or is_infestation or is_underwatered_specimen:
            # Fall back to using the full photo for dried, dead, infested, or irregular botanical subjects
            iso["crop_img"] = pil_img
            iso["masked_crop_img"] = pil_img
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
    if not fg_ok and not is_dried_specimen and not is_dead_specimen and not is_infestation and not is_underwatered_specimen:
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

    # 4. Stage 2: Foundation Botanical Identification (on Isolated Highest-Coverage Plant Crop)
    id_res = identify_dominant_plant(iso["crop_img"], top_k=5)

    # 5. Stage 3: Dual-Scale Surface Pathology, Lesion Localization & Masked Stress
    path_res = diagnose_surface_condition(
        iso["crop_img"],
        iso["mask_crop"],
        plant_family=id_res.get("family", "Unknown"),
        is_externally_dried=is_dried_specimen,
        is_externally_dead=is_dead_specimen,
        is_externally_underwatered=is_underwatered_specimen,
        verified_pest=verified_pest,
    )
    primary_cond = path_res["primary_class"]
    care = SURFACE_CARE_MAP.get(primary_cond, SURFACE_CARE_MAP["Healthy_Foliage"])
    lesion_info = path_res["lesion_info"]

    is_pest = bool(verified_pest or primary_cond in ("Mealybugs_Cochineal", "Aphids", "Scale_Insects", "Thrips_Damage", "Spider_Mites", "Whiteflies", "Leaf_Miners"))
    is_dead = bool((is_dead_specimen or path_res.get("is_dead", False) or primary_cond == "Abiotic_Terminal_Dead_Plant") and not is_pest)
    is_dried = bool((is_dead or is_dried_specimen or primary_cond == "Abiotic_Severe_Desiccation_Dried_Plant") and not is_pest)
    is_underwatered = bool((is_underwatered_specimen or primary_cond == "Abiotic_Dehydration_Underwatering") and not is_pest and not is_dried and not is_dead)

    raw_species = id_res.get("raw_top_species") or id_res.get("species", "Botanical Specimen")
    if is_pest and verified_pest:
        pest_label = care["condition_name"].split("/")[0].split("(")[0].strip()
        species_name = f"{raw_species} ({pest_label} Infestation)"
    elif is_dead:
        species_name = f"{raw_species} (Dead / Non-Viable Specimen)"
    elif is_dried:
        species_name = f"{raw_species} (Dried / Withered Specimen)"
    elif is_underwatered:
        species_name = f"{raw_species} (Underwatered / Needs Water)"
    else:
        species_name = raw_species

    multi_note = (
        f"Detected {iso['instances_count']} plants; focused exclusively on largest foreground plant ({iso['dominant_coverage_pct']}% frame coverage)."
        if iso["instances_count"] > 1
        else f"Primary plant isolated cleanly ({iso['dominant_coverage_pct']}% frame coverage)."
    )
    if is_pest and verified_pest:
        multi_note = f"{multi_note} Active pest infestation verified: {care['condition_name']}."
    elif is_dead:
        multi_note = f"{multi_note} Specimen exhibits irreversible cellular mortality (beyond saving)."
    elif is_dried:
        multi_note = f"{multi_note} Specimen exhibits severe desiccation / dried foliage."
    elif is_underwatered:
        multi_note = f"{multi_note} Specimen exhibits foliar dehydration / loss of turgor (needs water)."

    all_recs = list(care["immediate_actions"]) + [r for r in id_res.get("recommendations", []) if r not in care["immediate_actions"]]

    if is_pest and verified_pest:
        spots_status_msg = f"Active biotic pest infestation detected: {care['condition_name']}. Inspect leaf nodes, axils, and leaf undersides immediately."
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

    return {
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
            "raw_top_species": id_res.get("raw_top_species"),
            "family": id_res.get("family"),
            "confidence": id_res["confidence"],
            "detected_mode": f"CoreTech v2 Ensemble ({id_res['status']})",
            "top_candidates": id_res["candidates"],
        },
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
                and lesion_info["spots_count"] > 0
                and not is_underwatered
                and not is_dead
                and not is_dried
            ),
            "spots_count": lesion_info["spots_count"] if (not is_underwatered and not is_dead and not is_dried) else 0,
            "infected_area_pct": lesion_info["infected_area_pct"] if (not is_underwatered and not is_dead and not is_dried) else 0.0,
            "status_message": spots_status_msg,
            "annotated_image_base64": _img_to_b64(lesion_info["annotated_img"]),
        },
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
                                else []
                            )
                        )
                    )
                )
            ),
        },
        "predictions": path_res["predictions"],
        "preprocessed_image_base64": _img_to_b64(iso["crop_img"]),
    }
