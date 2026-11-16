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
            "Examine this photo carefully.\n"
            "1. Determine whether there is at least one genuine living botanical plant "
            "(houseplant, leaf, stem, flower, tree, crop, herb, cactus, or succulent) present.\n"
            "2. Artificial plastic plants, painted green furniture (like green chairs/sofas), green walls, "
            "printed plant pictures on screens/posters, clothing, electronics, animals, or people WITHOUT "
            "an actual living plant as a clear subject must be marked is_plant=false.\n"
            "3. Also note if multiple separate plants are visible.\n\n"
            "Respond ONLY with valid JSON:\n"
            "{\n"
            '  "is_plant": true or false,\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "detected_subject": "concise description of main subject",\n'
            '  "multiple_plants": true or false,\n'
            '  "reason": "1 concise sentence explaining why"\n'
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
        subject = str(data.get("detected_subject", "Unknown subject"))
        reason = str(data.get("reason", ""))
        conf = float(data.get("confidence", 1.0))

        if not is_plant:
            msg = f"Non-plant detected: {subject}. {reason}".strip()
        else:
            msg = f"Living botanical specimen verified: {subject}."

        return is_plant, msg, {
            "source": "Google Gemini Vision API",
            "detected_subject": subject,
            "confidence": round(conf * 100.0, 1),
            "multiple_plants": bool(data.get("multiple_plants", False)),
            "reason": reason,
        }
    except Exception as e:
        return None, f"Online semantic verifier error: {e}", {"source": "exception"}
