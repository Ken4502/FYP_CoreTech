"""
Stage 1A: 100% Online Semantic Non-Plant & Scene Verification Gate.
Uses Google Gemini Multimodal Vision API to semantically verify that an uploaded
image contains at least one real living plant (no color heuristics, no offline MobileNet fallback).
"""
import io
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image

V2_ROOT = Path(__file__).resolve().parent.parent
CORETECH_ROOT = V2_ROOT.parent
FYP_ROOT = CORETECH_ROOT.parent


def get_gemini_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    for var in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()
    search_paths = [
        CORETECH_ROOT / "gemini_api_key.txt",
        FYP_ROOT / "gemini_api_key.txt",
        CORETECH_ROOT / "Plant_Disease_Experiment" / "gemini_api_key.txt",
        CORETECH_ROOT / ".env",
        FYP_ROOT / ".env",
    ]
    for p in search_paths:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
                    if len(line) >= 20 and " " not in line:
                        return line
            except Exception:
                pass
    return None


def verify_plant_online(
    img: Image.Image, explicit_api_key: Optional[str] = None
) -> Tuple[Optional[bool], str, Dict[str, Any]]:
    """
    Runs 100% online semantic verification via Gemini Vision.
    Returns (is_plant, message, metadata).
    Note: Returns is_plant=None ONLY if API call itself fails/unavailable so caller can handle gracefully.
    """
    api_key = get_gemini_api_key(explicit_api_key)
    if not api_key:
        return None, "Online verification key missing.", {"source": "unavailable"}

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        buf = io.BytesIO()
        # Preserve aspect ratio up to 640px longest side
        w, h = img.size
        scale = min(1.0, 640.0 / max(w, h))
        resized = img.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.Resampling.BILINEAR)
        resized.convert("RGB").save(buf, format="JPEG", quality=88)
        image_bytes = buf.getvalue()

        prompt = (
            "Examine this photo with expert botanical and entomological precision.\n"
            "1. Determine whether genuine botanical plant matter is present.\n"
            "   Includes: living plants, dried leaves, dry/withered stems, desiccated foliage, "
            "   fallen leaves, cuttings, flowers, crops, trees, cacti, or succulents.\n"
            "   Even if completely dried up, brown, withered, severely underwatered, or dead, "
            "   it MUST be marked is_plant=true.\n"
            "   Non-plant subjects (plastic/silk artificial plants, green walls, furniture, "
            "   clothing, electronics, dishes, animals, humans without plants) MUST be marked is_plant=false.\n\n"
            "2. Detect Biotic Pests & Insect Infestations:\n"
            "   Carefully inspect stems, nodes, leaf axils, veins, and leaf undersides for:\n"
            "   - 'mealybugs': White cottony, powdery, or waxy fluff/clusters/cushions.\n"
            "   - 'aphids': Clusters of tiny soft-bodied sap-sucking insects (black aphids / black pests, brown, or green plant lice).\n"
            "   - 'scale_insects': Brown, black, or tan waxy shell-like or armored bumps stuck to stems/midribs.\n"
            "   - 'thrips': Slender tiny dark/yellow insects with silvery stippling/rasping and dark frass spots.\n"
            "   - 'spider_mites': Fine silky webbing, microscopic moving specks, stippling.\n"
            "   - 'whiteflies': Tiny white powdery moth-like insects or clear nymphs under leaves.\n"
            "   - 'leaf_miners': Pale winding serpentine squiggly tunnels burrowed through leaf mesophyll.\n"
            "   - 'chewing_pests_bug_bites_holes': Foliar chew holes, bite marks, shot-hole perforations, missing leaf margins, or chewed foliage eaten by caterpillars, beetles, grasshoppers, leaf-cutter bees, or slugs/snails.\n"
            "   - 'none': No visible insects, mealybugs, scale, aphids, or pests.\n\n"
            "3. Assess Hydration, Viability, and Desiccation State:\n"
            "   Pay special attention to posture, wilting, and leaf turgor firmness:\n"
            "   - 'needs_water': Living foliage that is visibly wilting, drooping downward, limp/flaccid, curling inward, or losing turgidity due to acute water deficit (even if leaves are still green!).\n"
            "   - 'severely_dried': Foliage or cuttings are completely crispy, papery, brown, withered, or desiccated (can snap/crackle, but root crown/stem base may still have living cambium).\n"
            "   - 'dead_beyond_saving': Completely dead, black/dark brown collapsed, rotted, or dry brittle dead kindling with zero living green cambium tissue.\n"
            "   - 'normal_healthy_turgor': Upright, firm, turgid healthy foliage with optimal hydration and normal turgor.\n\n"
            "4. Detect Specific Fungal or Bacterial Pathologies:\n"
            "   Carefully inspect leaf surfaces for distinct foliar lesions:\n"
            "   - 'rust_fungi': Raised bright orange, cinnamon, rust-brown, or yellow-orange powdery spore pustules / blisters (Pucciniales / uredinia) on foliage.\n"
            "   - 'powdery_mildew': Superficial white or light-gray talcum powder-like fungal mycelium patches coating leaf surfaces.\n"
            "   - 'sooty_mold': Black charcoal/soot-like superficial crust coating leaf surfaces (typically following sap-sucking pests).\n"
            "   - 'downy_mildew': Angular chlorotic patches on upper leaf surface with fuzzy grayish-purple downy sporulation underneath.\n"
            "   - 'botrytis_gray_mold': Fuzzy gray-brown mold cushions on decaying flowers, stems, or soft tissues.\n"
            "   - 'fungal_leaf_spot_anthracnose': Dark brown, black, or tan necrotic circular/angular spots with concentric rings, yellow halos, or sunken lesions (WITHOUT powdery orange spore pustules).\n"
            "   - 'bacterial_wilt_soft_rot': Water-soaked, translucent, foul-smelling soft mushy rotting patches or sudden bacterial wilt.\n"
            "   - 'none': No active fungal or bacterial disease.\n\n"
            "5. Determine Primary Surface Pathology:\n"
            "   Choose the single most dominant condition: 'chewing_pests_bug_bites_holes', 'rust_fungi', 'rust_with_bug_bites', 'pest_infestation', 'powdery_mildew', "
            "   'fungal_leaf_spot_anthracnose', 'bacterial_wilt_soft_rot', 'sooty_mold', 'underwatering_needs_water', "
            "   'severe_desiccation', 'dead_plant', or 'healthy'.\n\n"
            "Respond ONLY with valid JSON:\n"
            "{\n"
            '  "is_plant": true or false,\n'
            '  "detected_pest": "chewing_pests_bug_bites_holes" or "mealybugs" or "aphids" or "scale_insects" or "thrips" or "spider_mites" or "whiteflies" or "leaf_miners" or "none",\n'
            '  "detected_disease": "rust_fungi" or "powdery_mildew" or "sooty_mold" or "downy_mildew" or "botrytis_gray_mold" or "fungal_leaf_spot_anthracnose" or "bacterial_wilt_soft_rot" or "none",\n'
            '  "has_bug_bites_holes": true or false,\n'
            '  "has_rusting": true or false,\n'
            '  "bug_bite_holes_severity": "none" or "mild" or "moderate" or "severe",\n'
            '  "rust_severity": "none" or "mild" or "moderate" or "severe",\n'
            '  "water_and_viability": "needs_water" or "severely_dried" or "dead_beyond_saving" or "normal_healthy_turgor",\n'
            '  "primary_pathology": "chewing_pests_bug_bites_holes" or "rust_fungi" or "rust_with_bug_bites" or "pest_infestation" or "powdery_mildew" or "fungal_leaf_spot_anthracnose" or "bacterial_wilt_soft_rot" or "sooty_mold" or "underwatering_needs_water" or "severe_desiccation" or "dead_plant" or "healthy",\n'
            '  "is_dried": true or false,\n'
            '  "is_dead_beyond_saving": true or false,\n'
            '  "desiccation_status": "fresh_living" or "severely_dried_withered" or "dead_beyond_saving",\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "detected_subject": "concise description of main subject (e.g. leaf with caterpillar bite holes, orange rust pustules, or drooping peace lily)",\n'
            '  "multiple_plants": true or false,\n'
            '  "reason": "1 concise sentence explaining the finding"\n'
            "}"
        )

        response = None
        for model_name in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash"]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                if response and response.text:
                    break
            except Exception:
                continue

        if not response or not response.text:
            return None, "Online verification service temporarily unreachable.", {"source": "api_error"}

        data = json.loads(response.text.strip())
        is_plant = bool(data.get("is_plant", False))
        detected_pest = str(data.get("detected_pest", "none")).lower().strip()
        water_and_viability = str(data.get("water_and_viability", "normal_healthy_turgor")).lower().strip()
        primary_pathology = str(data.get("primary_pathology", "healthy")).lower().strip()
        detected_disease = str(data.get("detected_disease", "none")).lower().strip()
        reason_lower = str(data.get("reason", "")).lower()
        subject_lower = str(data.get("detected_subject", "")).lower()

        combined_diagnostic_text = f"{detected_pest} {detected_disease} {primary_pathology} {reason_lower} {subject_lower}"

        # Dedicated bug bite holes & chewing pest indicators
        has_bug_bites_holes = bool(
            data.get("has_bug_bites_holes", False)
            or any(
                w in combined_diagnostic_text
                for w in [
                    "bite",
                    "chew",
                    "hole",
                    "caterpillar",
                    "beetle",
                    "slug",
                    "snail",
                    "grasshopper",
                    "shot hole",
                    "shothole",
                    "perforat",
                    "ragged margin",
                    "eaten leaf",
                    "chewed leaf",
                ]
            )
        )

        # Dedicated rust fungi indicators
        has_rusting = bool(
            data.get("has_rusting", False)
            or any(
                w in combined_diagnostic_text
                for w in ["rust", "puccinial", "uredini", "orange pustule", "rusty", "cinnamon spore"]
            )
        )

        bug_bite_holes_severity = str(data.get("bug_bite_holes_severity", "none")).capitalize()
        if bug_bite_holes_severity == "None" and has_bug_bites_holes:
            bug_bite_holes_severity = "Moderate"

        rust_severity = str(data.get("rust_severity", "none")).capitalize()
        if rust_severity == "None" and has_rusting:
            rust_severity = "Moderate"

        # Map pest to canonical CoreTech v2 surface class
        verified_pest_class = None
        if "mealy" in detected_pest or "cochineal" in detected_pest:
            verified_pest_class = "Mealybugs_Cochineal"
        elif "aphid" in detected_pest or "black" in detected_pest or "lice" in detected_pest:
            verified_pest_class = "Aphids"
        elif "scale" in detected_pest:
            verified_pest_class = "Scale_Insects"
        elif "thrip" in detected_pest:
            verified_pest_class = "Thrips_Damage"
        elif "spider" in detected_pest or "mite" in detected_pest:
            verified_pest_class = "Spider_Mites"
        elif "whitefl" in detected_pest:
            verified_pest_class = "Whiteflies"
        elif "miner" in detected_pest:
            verified_pest_class = "Leaf_Miners"
        elif has_bug_bites_holes or "chew" in detected_pest or "hole" in detected_pest:
            verified_pest_class = "Chewing_Pests_Bug_Bites_Holes"

        # Map disease to canonical CoreTech v2 surface pathology class
        verified_pathology_class = None
        if has_bug_bites_holes and has_rusting:
            verified_pathology_class = "Rust_Fungi_With_Bug_Bites"
        elif has_rusting:
            verified_pathology_class = "Rust_Fungi"
        elif "powdery" in combined_diagnostic_text or ("mildew" in combined_diagnostic_text and "downy" not in combined_diagnostic_text):
            verified_pathology_class = "Powdery_Mildew"
        elif "sooty" in combined_diagnostic_text:
            verified_pathology_class = "Sooty_Mold"
        elif "downy" in combined_diagnostic_text:
            verified_pathology_class = "Downy_Mildew"
        elif "botrytis" in combined_diagnostic_text or "gray mold" in combined_diagnostic_text or "grey mold" in combined_diagnostic_text:
            verified_pathology_class = "Botrytis_Gray_Mold"
        elif "anthracnose" in combined_diagnostic_text or "leaf spot" in combined_diagnostic_text:
            verified_pathology_class = "Fungal_Leaf_Spot_Anthracnose"
        elif "bacterial" in combined_diagnostic_text or "soft rot" in combined_diagnostic_text:
            verified_pathology_class = "Bacterial_Wilt_Soft_Rot"

        is_infestation = bool(verified_pest_class is not None or primary_pathology in ("pest_infestation", "chewing_pests_bug_bites_holes"))
        is_verified_disease = bool(verified_pathology_class is not None)

        # Crucial: Pest infestations & active diseases must NOT be labeled as dried or dead!
        if is_infestation or is_verified_disease or has_bug_bites_holes or has_rusting:
            is_dried = False
            is_dead = False
            is_underwatered = False
        else:
            combined_text = f"{reason_lower} {subject_lower}"

            is_underwatered = bool(
                water_and_viability == "needs_water"
                or primary_pathology == "underwatering_needs_water"
                or any(w in combined_text for w in ["wilt", "droop", "flaccid", "thirst", "underwater"])
            )
            desiccation_status = str(data.get("desiccation_status", "fresh_living")).lower()
            is_dead = bool(
                data.get("is_dead_beyond_saving", False)
                or water_and_viability == "dead_beyond_saving"
                or primary_pathology == "dead_plant"
                or desiccation_status == "dead_beyond_saving"
                or any(w in combined_text for w in ["dead plant", "dead leaf", "dead stem", "completely dead", "beyond saving"])
            )
            is_dried = bool(
                is_dead
                or data.get("is_dried", False)
                or water_and_viability == "severely_dried"
                or primary_pathology == "severe_desiccation"
                or "dried" in desiccation_status
                or "withered" in desiccation_status
                or any(w in combined_text for w in ["crispy leaf", "dried leaf", "dried plant", "desiccated", "dry brittle"])
            )

        subject = str(data.get("detected_subject", "Unknown subject"))
        reason = str(data.get("reason", ""))
        conf = float(data.get("confidence", 1.0))

        if not is_plant:
            msg = f"Non-plant detected: {subject}. {reason}".strip()
        elif verified_pathology_class == "Rust_Fungi_With_Bug_Bites":
            msg = f"Dual foliar condition verified (Rust Fungi & Bug Bite Holes): {subject}."
        elif has_bug_bites_holes and verified_pest_class == "Chewing_Pests_Bug_Bites_Holes":
            msg = f"Insect chewing damage & bug bite holes verified: {subject}."
        elif is_infestation and verified_pest_class:
            msg = f"Pest infestation verified ({verified_pest_class}): {subject}."
        elif verified_pathology_class == "Rust_Fungi":
            msg = f"Fungal rust infection verified (Rust_Fungi): {subject}."
        elif verified_pathology_class:
            msg = f"Botanical disease verified ({verified_pathology_class}): {subject}."
        elif is_dead:
            msg = f"Non-viable dead botanical matter verified (Beyond Saving): {subject}."
        elif is_dried:
            msg = f"Botanical specimen verified (Dried/Desiccated State): {subject}."
        elif is_underwatered:
            msg = f"Botanical specimen verified (Underwatering / Needs Water): {subject}."
        else:
            msg = f"Living botanical specimen verified: {subject}."

        return is_plant, msg, {
            "source": "Google Gemini Vision API",
            "confidence": round(conf * 100.0, 1),
            "detected_subject": subject,
            "multiple_plants": bool(data.get("multiple_plants", False)),
            "detected_pest": detected_pest,
            "verified_pest_class": verified_pest_class,
            "detected_disease": detected_disease,
            "verified_pathology_class": verified_pathology_class,
            "has_bug_bites_holes": has_bug_bites_holes,
            "has_rusting": has_rusting,
            "bug_bite_holes_severity": bug_bite_holes_severity,
            "rust_severity": rust_severity,
            "is_infestation": is_infestation,
            "is_underwatered": is_underwatered,
            "is_dried": is_dried,
            "is_dead_beyond_saving": is_dead,
            "water_and_viability": water_and_viability,
            "primary_pathology": primary_pathology,
            "desiccation_status": data.get("desiccation_status", "fresh_living"),
            "reason": reason,
        }
    except Exception as e:
        return None, f"Online semantic verifier error: {e}", {"source": "exception"}
