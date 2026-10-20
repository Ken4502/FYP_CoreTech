"""
FastAPI Server for PhytoCare Ultra Disease Diagnostic & Treatment Advisor.
Runs on Port 8001.
"""

import sys
import io
import time
import base64
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config
from src.predict import diagnose_plant, load_diagnostic_system
from src.botanical_identifier import identify_botanical_species, map_species_to_disease_group
from src.abiotic_analyzer import analyze_abiotic_stress
from web.disease_benchmark import disease_benchmark_manager

app = FastAPI(
    title="PhytoCare Ultra - Plant Pathology Diagnosis & Treatment Advisor",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = EXPERIMENT_ROOT / "web" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

LOG_FILE = EXPERIMENT_ROOT / "reports" / "mega_pipeline.log"

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>PhytoCare Ultra</h1><p>Web template loading...</p>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/log", response_class=HTMLResponse)
async def serve_log_page():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PhytoCare Ultra - Live Training & Telemetry Console</title>
        <meta charset="utf-8">
        <style>
            body { background: #0d1117; color: #58a6ff; font-family: 'Consolas', monospace; padding: 20px; }
            h2 { color: #2ea043; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
            #console { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 15px; height: 80vh; overflow-y: auto; white-space: pre-wrap; font-size: 13px; color: #c9d1d9; }
        </style>
    </head>
    <body>
        <h2>🌿 PhytoCare Ultra Pathology Training & Calibration Telemetry</h2>
        <div id="console">Loading live telemetry stream...</div>
        <script>
            async function fetchLog() {
                try {
                    const res = await fetch('/api/pipeline/log');
                    const text = await res.text();
                    const el = document.getElementById('console');
                    el.textContent = text || 'No pipeline logs recorded yet.';
                    el.scrollTop = el.scrollHeight;
                } catch(e) {}
            }
            setInterval(fetchLog, 1500);
            fetchLog();
        </script>
    </body>
    </html>
    """)

@app.get("/api/info")
async def api_info():
    model, classes, spot_detector, kb = load_diagnostic_system()
    device_str = "NVIDIA RTX 3060 (CUDA)" if "cuda" in str(config.DEVICE).lower() else "CPU"
    return {
        "status": "online",
        "model_ready": model is not None,
        "device": device_str,
        "num_classes": len(classes),
        "classes": classes,
        "supported_plants": ["Aloe_Vera", "Cactus", "Money_Plant", "Snake_Plant", "Spider_Plant", "Universal_Pests"],
        "model_architecture": "PhytoCare Ultra ConvNeXt-Small 384px (31 Classes + FloraNet-250 Prior)"
    }

@app.get("/api/encyclopedia")
async def api_encyclopedia():
    kb_path = EXPERIMENT_ROOT / "artifacts" / "disease_knowledge_base.json"
    if kb_path.exists():
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        return {"status": "success", "count": len(kb), "entries": kb}
    return {"status": "error", "message": "Knowledge base not found"}

@app.get("/api/pipeline/log")
async def get_pipeline_log():
    if not LOG_FILE.exists():
        return "Log initialized. Awaiting training telemetry..."
    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-150:])

class BenchmarkStartRequest(BaseModel):
    total_images: Optional[int] = 500
    source: Optional[str] = "hybrid"

@app.get("/benchmark", response_class=HTMLResponse)
async def serve_benchmark_page():
    bench_path = STATIC_DIR / "benchmark.html"
    if not bench_path.exists():
        return HTMLResponse("<h1>PhytoCare Ultra 500 Benchmark Arena</h1><p>Template loading...</p>")
    with open(bench_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/benchmark/start")
async def api_benchmark_start(req: BenchmarkStartRequest):
    total = req.total_images if req.total_images else 500
    source = req.source if req.source else "hybrid"
    return disease_benchmark_manager.start(total_images=total, source=source)

@app.get("/api/benchmark/status")
async def api_benchmark_status(limit: int = 50):
    return disease_benchmark_manager.get_status(limit=limit)

@app.post("/api/benchmark/stop")
async def api_benchmark_stop():
    return disease_benchmark_manager.stop()

@app.get("/api/benchmark/export")
async def api_benchmark_export():
    csv_str = disease_benchmark_manager.export_csv()
    return StreamingResponse(
        io.StringIO(csv_str),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=phytonet_500_internet_benchmark.csv"}
    )

@app.get("/api/benchmark/image")
async def api_benchmark_image(idx: int):
    img_bytes = disease_benchmark_manager.get_specimen_image(idx)
    if img_bytes is None:
        raise HTTPException(status_code=404, detail="Specimen image not found in cache")
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")

@app.post("/api/predict")
@app.post("/api/diagnose")
async def api_predict(file: UploadFile = File(...), plant: str = Form("auto")):
    t0 = time.time()
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    try:
        # Preprocessed preview thumbnail
        prep_buf = io.BytesIO()
        pil_img.resize((384, 384), Image.BILINEAR).save(prep_buf, format="JPEG", quality=85)
        prep_b64 = "data:image/jpeg;base64," + base64.b64encode(prep_buf.getvalue()).decode("utf-8")

        # 1. LAYER 1: BOTANICAL PLANT IDENTIFICATION (FloraNet-250)
        id_res = identify_botanical_species(pil_img, top_k=5)
        botanical_species = id_res["top_species"]
        botanical_conf = id_res["confidence"]
        auto_group = id_res["mapped_disease_group"]
        candidates = id_res["candidates"]

        # Pathology Model baseline group texture voting
        model, classes, spot_detector, kb = load_diagnostic_system()
        with torch.no_grad():
            from src.predict import get_5crop_tta_batch, _cached_temp
            b_tensor = get_5crop_tta_batch(pil_img, img_size=384).to(config.DEVICE)
            raw_out = model(b_tensor)
            raw_probs = torch.softmax(raw_out / _cached_temp, dim=1).mean(dim=0)
            groups = ["Aloe", "Cactus", "Money_Plant", "Snake_Plant", "Spider_Plant"]
            g_sums = {g: sum(raw_probs[i].item() for i, c in enumerate(classes) if c.startswith(g + "_")) for g in groups}
            top_path_grp = max(g_sums, key=g_sums.get)
            top_path_cf = g_sums[top_path_grp] * 100

        # Determine effective target plant group
        if plant and plant != "auto" and plant != "None":
            target_group = plant
            display_plant = plant.replace("_", " ")
            detected_mode = f"User Selected ({plant.replace('_', ' ')})"
            is_supported = True
        elif auto_group is not None and botanical_conf >= 45.0:
            # High-confidence botanical match from FloraNet-250
            target_group = auto_group
            display_plant = botanical_species
            detected_mode = "FloraNet-250 Botanical Classifier"
            is_supported = True
        elif top_path_cf >= 50.0:
            # Close-up lesion crop where microscopic pathology features dominate
            target_group = top_path_grp
            display_plant = top_path_grp.replace("_", " ")
            detected_mode = f"Clinical Pathology Feature Match ({top_path_grp.replace('_', ' ')})"
            is_supported = True
        elif auto_group is not None:
            target_group = auto_group
            display_plant = botanical_species
            detected_mode = "FloraNet-250 Consensus"
            is_supported = True
        else:
            # Universal pest screening for all other 245 FloraNet plant species!
            target_group = "Universal"
            display_plant = botanical_species
            detected_mode = "FloraNet-250 Botanical Prior (Universal Pest Screening)"
            is_supported = True

        # 2. Branch: Outside Supported Pathology Taxa
        if not is_supported:
            inference_time_ms = round((time.time() - t0) * 1000, 1)
            return {
                "status": "success",
                "inference_time_ms": inference_time_ms,
                "disease_available": False,
                "status_message": f"Identified as {botanical_species} ({botanical_conf}% confidence). The pathology diagnostic model currently specializes in Aloe Vera, Cacti, Money Plant (Pothos), Snake Plant, and Spider Plant.",
                "species": {
                    "name": botanical_species,
                    "confidence": botanical_conf,
                    "detected_mode": detected_mode,
                    "top_candidates": candidates
                },
                "diagnosis": {
                    "condition_name": "Disease Model Unavailable",
                    "category": "Unsupported Botanical Species",
                    "is_healthy": True,
                    "confidence": botanical_conf,
                    "severity": "None"
                },
                "spots": {
                    "has_spots": False,
                    "spots_count": 0,
                    "infected_area_pct": 0.0,
                    "status_message": "Spot localization is disabled because no disease model exists for this botanical species.",
                    "annotated_image_base64": prep_b64
                },
                "care_guide": {
                    "summary": f"Specimen identified as {botanical_species} ({botanical_conf}% match).",
                    "immediate_actions": ["No disease baseline available for this taxon in current model."],
                    "treatment": {"organic": [], "chemical": []},
                    "prevention": ["Maintain standard botanical care guidelines for this species."],
                    "symptoms": [],
                    "causes": []
                },
                "predictions": [],
                "abiotic": analyze_abiotic_stress(pil_img, spots_count=0),
                "preprocessed_image_base64": prep_b64
            }

        # 3. LAYER 2: INTRA-GROUP PATHOLOGY DIAGNOSIS & SPOT DETECTION
        model, classes, spot_detector, kb = load_diagnostic_system()
        diagnosis = diagnose_plant(
            pil_img,
            target_group=target_group,
            top_k=5,
            use_tta=True,
            detect_spots=True
        )

        # Annotated lesion visualization
        annotated_b64 = None
        vis_img = diagnosis["lesion_highlight"].get("annotated_image")
        if vis_img is not None:
            buf = io.BytesIO()
            vis_img.save(buf, format="JPEG", quality=90)
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            annotated_b64 = prep_b64

        raw_preds = diagnosis["predictions"]
        top_raw = raw_preds[0]
        top_cls = top_raw["class_name"]

        # Look up curated care guide from knowledge base
        top_guide = kb.get(top_cls, {
            "plant_name": target_group.replace("_", " "),
            "condition_name": top_cls.replace("_", " "),
            "is_healthy": "healthy" in top_cls.lower(),
            "severity": "Moderate" if "healthy" not in top_cls.lower() else "None",
            "category": "Pathology" if "healthy" not in top_cls.lower() else "Healthy",
            "summary": "Pathological symptoms identified on foliage surface.",
            "immediate_actions": ["Isolate specimen to avoid spore drift.", "Cease foliar misting."],
            "treatment": {
                "organic": ["Apply neem oil spray (1 tsp per L water)."],
                "chemical": ["Apply copper fungicide as per label."]
            },
            "prevention": ["Ensure proper soil drainage.", "Avoid overhead watering."],
            "symptoms": ["Visible spotting on leaves."],
            "causes": ["High humidity or fungal spores."]
        })

        # Enrich all predictions with individual care guides
        enriched_predictions = []
        for p in raw_preds:
            cls_name = p["class_name"]
            guide = kb.get(cls_name, {})
            cond_display = guide.get("condition_name", cls_name.replace("_", " "))
            enriched_predictions.append({
                "rank": p["rank"],
                "class_name": cls_name,
                "condition_name": cond_display,
                "confidence": p["confidence"],
                "is_healthy": p["is_healthy"],
                "severity": guide.get("severity", "Moderate" if not p["is_healthy"] else "None"),
                "category": guide.get("category", "Pathology" if not p["is_healthy"] else "Healthy"),
                "care_guide": guide
            })

        display_plant = botanical_species if plant == "auto" else target_group.replace("_", " ")
        spots_count = diagnosis["lesion_highlight"]["boxes_count"]
        inf_area = diagnosis["lesion_highlight"]["severity_pct"]
        has_spots = spots_count > 0 and not top_raw["is_healthy"]
        
        # Stage-1 Foliar Health Gate Telemetry
        hg_data = diagnosis.get("health_gate", {"p_healthy": 50.0, "p_disease": 50.0})
        p_healthy = hg_data["p_healthy"]

        # Dual-Safety Health Verification Gate:
        # Prevents false-positive disease verdicts on healthy foliage, spines, ribs, and variegated leaves
        is_health_gated = (
            top_raw["is_healthy"] or
            p_healthy >= 75.0 or
            (target_group == "Universal" and plant == "auto" and not has_spots and top_raw["confidence"] < 50.0)
        )

        if is_health_gated and not top_raw["is_healthy"]:
            # Promote Healthy Baseline
            healthy_key = f"{target_group}_Healthy"
            if healthy_key in kb:
                top_guide = kb[healthy_key]
                top_condition_name = f"{display_plant} Healthy Baseline"
            else:
                top_guide = {
                    "plant_name": display_plant,
                    "condition_name": "Healthy Baseline (No Active Pests)",
                    "is_healthy": True,
                    "severity": "None",
                    "category": "Healthy Baseline",
                    "summary": f"Foliage inspection confirms clean and thriving tissue with no detected pest colonies, webbing, honeydew, or fungal spore clusters.",
                    "immediate_actions": [
                        "No isolation or quarantine required.",
                        "Continue normal ambient care and watering schedule.",
                        "Conduct periodic routine leaf checks during seasonal changes."
                    ],
                    "treatment": {
                        "organic": ["No treatment required. Wipe foliage with a clean damp microfiber cloth as routine hygiene."],
                        "chemical": ["No chemical pesticides or fungicides needed."]
                    },
                    "prevention": [
                        "Maintain adequate airflow between potted plants.",
                        "Avoid water accumulation in saucers or stagnant leaf moisture."
                    ],
                    "symptoms": ["Clean, vibrant green leaf surface with no visible discoloration, chlorosis, or pest colonies."],
                    "causes": ["Healthy plant environment and proper cultivation."]
                }
                top_condition_name = "Healthy Baseline (No Active Pest / Fungal Markers)"
            top_category = "Healthy Baseline"
            top_is_healthy = True
            top_severity = "None"
            top_confidence = p_healthy
            status_msg = f"Clean foliage surface ({p_healthy}% healthy tissue index). Natural anatomical structures, spines, or variegation verified non-pathological."
            has_spots = False
            spots_count = 0
        else:
            top_condition_name = top_guide.get("condition_name", top_cls.replace("_", " "))
            top_category = top_guide.get("category", "Pathology")
            top_is_healthy = top_raw["is_healthy"]
            top_severity = top_guide.get("severity", "Moderate" if not top_raw["is_healthy"] else "None")
            top_confidence = top_raw["confidence"]
            status_msg = (
                f"{spots_count} active lesion spot(s) localized ({inf_area}% leaf tissue)"
                if has_spots else
                "Clean foliage surface. No necrotic lesions or fungal spore clusters detected."
            )

        abiotic_res = analyze_abiotic_stress(pil_img, spots_count=spots_count if has_spots else 0)
        
        # 4. Quantitative PDI (Plant Disease Index) Severity Meter (EPPO / USDA Standard)
        if top_is_healthy or not has_spots:
            pdi_grade = 0
            pdi_label = "Grade 0: Optimal Foliar Health (Clean)"
            pdi_color = "#10b981"
            pdi_action = "No pathological interventions required. Maintain routine botanical care."
        elif inf_area < 2.0:
            pdi_grade = 1
            pdi_label = "Grade 1: Trace Inoculum (< 2% Leaf Area)"
            pdi_color = "#3b82f6"
            pdi_action = "Early-stage intervention. Selective pruning or gentle horticultural soap spray."
        elif inf_area < 10.0:
            pdi_grade = 2
            pdi_label = "Grade 2: Mild Infection (2 - 10% Leaf Area)"
            pdi_color = "#f59e0b"
            pdi_action = "Active treatment needed. Apply organic neem oil spray or botanical bio-fungicide."
        elif inf_area <= 25.0:
            pdi_grade = 3
            pdi_label = "Grade 3: Moderate Infection (10 - 25% Leaf Area)"
            pdi_color = "#ef4444"
            pdi_action = "Isolate specimen immediately to prevent drift. Apply targeted systemic chemical control."
        else:
            pdi_grade = 4
            pdi_label = "Grade 4: Severe / Critical Breakdown (> 25% Leaf Area)"
            pdi_color = "#7f1d1d"
            pdi_action = "Critical quarantine. Radical surgical excision of infected foliage or sacrificial stem pruning."

        pdi_rating = {
            "grade": pdi_grade,
            "label": pdi_label,
            "infected_pct": inf_area if has_spots else 0.0,
            "color": pdi_color,
            "clinical_action": pdi_action
        }

        # 5. Co-Infection & Physiological Coupling Alerts
        co_alerts = []
        if not top_is_healthy:
            if "aphids" in top_cls.lower():
                co_alerts.append({
                    "type": "Secondary Fungal Risk",
                    "badge": "Sooty Mold Alert",
                    "message": "Aphid colonies excrete honeydew that rapidly colonizes with black Sooty Mold. Wash foliage surfaces with lukewarm soapy water."
                })
            if "spider_mites" in top_cls.lower():
                co_alerts.append({
                    "type": "Atmospheric Vector Alert",
                    "badge": "Drought-Induced Outbreak",
                    "message": "Low ambient humidity accelerates spider mite reproduction. Increase room humidity to >= 50% and mist leaf undersides."
                })
            if "whiteflies" in top_cls.lower():
                co_alerts.append({
                    "type": "Viral Vector Risk",
                    "badge": "Begomovirus Risk",
                    "message": "Whiteflies vector plant viruses. Check leaf undersides for translucent scale-like nymphs and place yellow sticky traps."
                })
            if "scale" in top_cls.lower():
                co_alerts.append({
                    "type": "Protective Armor Barrier",
                    "badge": "Waxy Cuticle Resistance",
                    "message": "Armored scales resist standard foliar contact sprays. Dab individual scales directly with a cotton swab soaked in 70% isopropyl alcohol."
                })
            if "bacterial_wilt" in top_cls.lower():
                co_alerts.append({
                    "type": "Vascular Pathogen",
                    "badge": "Xylem Occlusion",
                    "message": "Bacterial pathogens clog vascular bundles. Sterilize shears with 70% alcohol between cuts to prevent spreading to neighboring plants."
                })

        inference_time_ms = round((time.time() - t0) * 1000, 1)

        return {
            "status": "success",
            "inference_time_ms": inference_time_ms,
            "disease_available": True,
            "species": {
                "name": display_plant,
                "confidence": botanical_conf if plant == "auto" else 100.0,
                "detected_mode": detected_mode,
                "top_candidates": candidates
            },
            "diagnosis": {
                "condition_name": top_condition_name,
                "category": top_category,
                "is_healthy": top_is_healthy,
                "confidence": top_confidence,
                "severity": top_severity
            },
            "spots": {
                "has_spots": has_spots,
                "spots_count": spots_count if has_spots else 0,
                "infected_area_pct": inf_area if has_spots else 0.0,
                "status_message": status_msg,
                "annotated_image_base64": annotated_b64
            },
            "pdi_rating": pdi_rating,
            "co_alerts": co_alerts,
            "care_guide": top_guide,
            "predictions": enriched_predictions,
            "abiotic": abiotic_res,
            "preprocessed_image_base64": prep_b64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {e}")

@app.post("/api/model/reload")
async def api_reload():
    load_diagnostic_system(reload=True)
    return {"status": "reloaded"}


