"""
FastAPI Server for Plant Identification Web Prototype.
Provides image prediction API and serves static frontend.
"""

import os
import sys
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# Ensure src is importable
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
coretech_root = project_root.parent
if str(coretech_root) not in sys.path:
    sys.path.insert(0, str(coretech_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import config
from src.predict import predict_image

app = FastAPI(title="FloraID - Plant Identification Prototype", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static directory
STATIC_DIR = current_dir / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>FloraID Web Prototype</h1><p>Frontend file not found.</p>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    """Receives an uploaded leaf or plant image and returns top predictions."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        start_time = time.time()
        predictions, clean_img = predict_image(image, top_k=5, return_preprocessed=True)
        inference_time_ms = round((time.time() - start_time) * 1000, 1)

        import base64
        buf = io.BytesIO()
        clean_img.save(buf, format="JPEG", quality=90)
        prep_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "inference_time_ms": inference_time_ms,
            "predictions": predictions,
            "preprocessed_image_base64": prep_b64
        })
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint not found. Please train the model first by running 'python run_pipeline.py'."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/info")
async def api_info():
    """Returns model metadata and status."""
    model_ready = config.BEST_MODEL_PATH.exists()
    classes_count = 0
    metrics = {}

    if config.METRICS_PATH.exists():
        import json
        with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    if config.CLASS_INDICES_PATH.exists():
        import json
        with open(config.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            classes_count = len(data.get("classes", []))

    return {
        "model_name": config.MODEL_NAME,
        "architecture_blueprint": "MobileNetV3 (Trained From Scratch)",
        "model_ready": model_ready,
        "classes_count": classes_count,
        "device": str(config.DEVICE),
        "metrics": metrics
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting FloraID Web Prototype on http://127.0.0.1:8000 ...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(current_dir))

