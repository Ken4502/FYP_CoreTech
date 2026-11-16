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
            "   - 'none': No visible insects, mealybugs, scale, aphids, or pests.\n\n"
            "3. Assess Hydration, Viability, and Desiccation State:\n"
            "   Pay special attention to posture, wilting, and leaf turgor firmness:\n"
            "   - 'needs_water': Living foliage that is visibly wilting, drooping downward, limp/flaccid, curling inward, or losing turgidity due to acute water deficit (even if leaves are still green!).\n"
            "   - 'severely_dried': Foliage or cuttings are completely crispy, papery, brown, withered, or desiccated (can snap/crackle, but root crown/stem base may still have living cambium).\n"
            "   - 'dead_beyond_saving': Completely dead, black/dark brown collapsed, rotted, or dry brittle dead kindling with zero living green cambium tissue.\n"
            "   - 'normal_healthy_turgor': Upright, firm, turgid healthy foliage with optimal hydration and normal turgor.\n\n"
            "4. Determine Primary Surface Pathology:\n"
            "   Choose the single most dominant condition: 'pest_infestation', 'underwatering_needs_water', "
            "   'severe_desiccation', 'dead_plant', 'fungal_bacterial_disease', or 'healthy'.\n\n"
            "Respond ONLY with valid JSON:\n"
            "{\n"
            '  "is_plant": true or false,\n'
            '  "detected_pest": "mealybugs" or "aphids" or "scale_insects" or "thrips" or "spider_mites" or "whiteflies" or "leaf_miners" or "none",\n'
            '  "water_and_viability": "needs_water" or "severely_dried" or "dead_beyond_saving" or "normal_healthy_turgor",\n'
            '  "primary_pathology": "pest_infestation" or "underwatering_needs_water" or "severe_desiccation" or "dead_plant" or "fungal_bacterial_disease" or "healthy",\n'
            '  "is_dried": true or false,\n'
            '  "is_dead_beyond_saving": true or false,\n'
            '  "desiccation_status": "fresh_living" or "severely_dried_withered" or "dead_beyond_saving",\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "detected_subject": "concise description of main subject (e.g. drooping peace lily needing water, money plant with mealybug clusters, or dried leaf)",\n'
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

        is_infestation = bool(verified_pest_class is not None or primary_pathology == "pest_infestation")

        # Crucial: Pest infestations (e.g. white cottony mealybugs) must NOT be labeled as dried or dead!
        if is_infestation:
            is_dried = False
            is_dead = False
            is_underwatered = False
        else:
            reason_lower = str(data.get("reason", "")).lower()
            subject_lower = str(data.get("detected_subject", "")).lower()
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
        elif is_infestation and verified_pest_class:
            msg = f"Pest infestation verified ({verified_pest_class}): {subject}."
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
