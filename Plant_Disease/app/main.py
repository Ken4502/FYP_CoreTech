"""
FastAPI Server for Hierarchical Plant Disease Diagnosis & Spot Localizer Web Prototype.
"""

import os
import sys
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# Ensure project root is in sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
coretech_root = project_root.parent
if str(coretech_root) not in sys.path:
    sys.path.insert(0, str(coretech_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import config
from src.predict import diagnose_image
from src.disease_knowledge import get_disease_info, DISEASE_KNOWLEDGE_BASE

app = FastAPI(
    title="PhytoCare - Hierarchical Plant Disease Diagnosis & Spot Detector",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = current_dir / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>PhytoCare Web Prototype</h1><p>Frontend template not found.</p>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    plant: str = Form("auto")
):
    """
    Receives an uploaded leaf or plant photo and optional target plant mode.
    Executes hierarchical diagnosis:
      - Automatically detects or filters by plant species
      - Evaluates disease condition strictly within that species group
      - Detects and outlines unhealthy lesion spots with bounding boxes
      - Confirms clean healthy leaf status if no spots are detected
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be a valid image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        start_time = time.time()
        result = diagnose_image(image, target_species=plant, top_k=5)
        inference_time_ms = round((time.time() - start_time) * 1000, 1)

        result["filename"] = file.filename
        result["inference_time_ms"] = inference_time_ms

        return JSONResponse(result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint not found. Please train the model first by running 'python run_pipeline.py'."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnosis error: {str(e)}")

@app.get("/api/info")
async def api_info():
    """Returns model metadata, active device, and metrics."""
    model_ready = config.BEST_MODEL_PATH.exists()
    metrics = {}

    if config.METRICS_PATH.exists():
        import json
        with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return {
        "model_name": config.MODEL_NAME,
        "architecture_blueprint": "Two-Layer Hierarchical: 47-Species Identification -> Group Pathology & Spot Localizer",
        "model_ready": model_ready,
        "supported_species": config.SPECIES_NAMES,
        "supported_disease_species": config.SUPPORTED_DISEASE_SPECIES_DISPLAY,
        "plant_groups": config.PLANT_GROUPS,
        "device": str(config.DEVICE),
        "metrics": metrics
    }

@app.get("/api/classes")
async def api_list_classes():
    """Returns all supported conditions and plant species."""
    return JSONResponse({
        "species": config.SPECIES_NAMES,
        "groups": config.PLANT_GROUPS,
        "knowledge_base": DISEASE_KNOWLEDGE_BASE
    })

if __name__ == "__main__":
    import uvicorn
    print("Starting PhytoCare Hierarchical Plant Disease Web Server on http://127.0.0.1:8001 ...")
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True, app_dir=str(current_dir))
