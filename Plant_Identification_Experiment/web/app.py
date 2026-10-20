"""
FloraNet-100 Interactive Web Application
Built with FastAPI & PyTorch GPU Inference.
"""

import io
import sys
import base64
import time
from pathlib import Path
from typing import List, Optional

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

CURRENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = CURRENT_DIR.parent
CORETECH_ROOT = EXPERIMENT_ROOT.parent

if str(CORETECH_ROOT) not in sys.path:
    sys.path.insert(0, str(CORETECH_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src import config
from src.predict import load_inference_model, predict_image
from tools.harvester import BOTANICAL_TAXON_MAP
from web.benchmark import (
    fetch_random_internet_plant,
    run_all_model_benchmarks,
    download_image_from_url,
    batch_benchmark_manager
)

app = FastAPI(
    title="FloraNet-100 Botanical Classifier",
    description="Web Interface for 100-Species Deepened & Widened Plant Identification",
    version="1.0.0"
)

# Pre-load model on startup
print("[FloraNet-100] Pre-loading inference model on GPU...")
load_inference_model()
print("[FloraNet-100] Model loaded and ready for incoming inference requests!")

SAMPLE_SPECIES = [
    "Golden Barrel Cactus (Echinocactus grusonii)",
    "Columnar Cactus (Pachycereus pringlei)",
    "Pothos (Ivy arum)",
    "Watermelon Peperomia (Peperomia argyreia)",
    "Wax Plant (Hoya carnosa)",
    "Calathea Medallion (Goeppertia roseopicta)",
    "Snake plant (Sanseviera)",
    "Monstera Deliciosa (Monstera deliciosa)",
    "Fiddle Leaf Fig (Ficus lyrata)",
    "Peace lily"
]

def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = CURRENT_DIR / "templates" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/compare", response_class=HTMLResponse)
async def serve_compare():
    compare_file = CURRENT_DIR / "templates" / "compare.html"
    if not compare_file.exists():
        raise HTTPException(status_code=404, detail="compare.html not found")
    with open(compare_file, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/log", response_class=HTMLResponse)
async def serve_log():
    log_file = CURRENT_DIR / "templates" / "log.html"
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="log.html not found")
    with open(log_file, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/pipeline/log")
async def get_pipeline_log():
    log_path = EXPERIMENT_ROOT / "reports" / "mega_pipeline.log"
    if not log_path.exists():
        return {"status": "empty", "content": "Pipeline log not created yet."}
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return {"status": "ok", "content": content}

@app.get("/api/pipeline/download-log")
async def download_pipeline_log():
    log_path = EXPERIMENT_ROOT / "reports" / "mega_pipeline.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file does not exist")
    return FileResponse(
        str(log_path),
        media_type="text/plain",
        filename="mega_pipeline.log"
    )

@app.get("/api/benchmark/random-plant")
async def get_random_internet_plant():
    return fetch_random_internet_plant()

class CompareRequest(BaseModel):
    image_url: Optional[str] = None
    ground_truth_common: Optional[str] = ""
    ground_truth_scientific: Optional[str] = ""

@app.post("/api/benchmark/compare")
async def compare_models(request: Request):
    try:
        content_type = request.headers.get("content-type", "")
        gt_common = ""
        gt_sci = ""
        if "multipart/form-data" in content_type:
            form = await request.form()
            file_item = form.get("file")
            if file_item is None:
                raise HTTPException(status_code=400, detail="No file in multipart form data")
            content = await file_item.read()
            pil_img = Image.open(io.BytesIO(content)).convert("RGB")
            gt_common = str(form.get("ground_truth_common", ""))
            gt_sci = str(form.get("ground_truth_scientific", ""))
        else:
            data = await request.json()
            img_url = data.get("image_url")
            if not img_url:
                raise HTTPException(status_code=400, detail="Missing image_url in JSON body")
            pil_img = download_image_from_url(img_url)
            gt_common = data.get("ground_truth_common", "")
            gt_sci = data.get("ground_truth_scientific", "")

        results = run_all_model_benchmarks(pil_img, gt_common=gt_common, gt_sci=gt_sci)
        return {"models": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark comparison error: {e}")

class BatchStartRequest(BaseModel):
    total_images: Optional[int] = 500
    category: Optional[str] = "wikimedia"

@app.post("/api/benchmark/batch/start")
async def start_batch_benchmark(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    total = int(data.get("total_images", 500))
    cat = str(data.get("category", "wikimedia"))
    return batch_benchmark_manager.start(total_images=total, category=cat)

@app.get("/api/benchmark/batch/status")
async def get_batch_benchmark_status(limit: int = 50):
    return batch_benchmark_manager.get_status(limit=limit)

@app.post("/api/benchmark/batch/stop")
async def stop_batch_benchmark():
    return batch_benchmark_manager.stop()

@app.get("/api/benchmark/batch/export")
async def export_batch_benchmark_results():
    csv_data = batch_benchmark_manager.export_csv()
    return StreamingResponse(
        io.StringIO(csv_data),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=floranet_500_internet_benchmark.csv"}
    )

@app.get("/api/benchmark/batch/analyze")
async def analyze_batch_benchmark():
    return batch_benchmark_manager.analyze_batch_results()

@app.post("/api/benchmark/batch/harvest")
async def harvest_hard_failures():
    analysis = batch_benchmark_manager.analyze_batch_results()
    if analysis.get("status") != "success":
        return {"status": "error", "message": "No evaluated results available to harvest."}
    
    candidates = analysis.get("all_hard_failures", [])
    if not candidates:
        return {"status": "empty", "message": "No hard failure cases found in batch."}

    from tools.analyze_and_improve import harvest_active_learning_images
    harvest_target = EXPERIMENT_ROOT / "data" / "active_learning_reinforcement"
    harvest_active_learning_images(candidates, harvest_target)
    return {
        "status": "success",
        "harvested_count": len(candidates),
        "target_dir": str(harvest_target)
    }

@app.post("/api/model/reload")
async def reload_model_checkpoint():
    from src.predict import load_inference_model
    model, classes = load_inference_model(reload=True)
    return {"status": "reloaded", "classes_count": len(classes)}

@app.get("/api/samples")
async def get_sample_list():
    test_dir = EXPERIMENT_ROOT / "data" / "test"
    available = []
    for sp in SAMPLE_SPECIES:
        folder = test_dir / sp
        if folder.exists() and any(folder.glob("*.*")):
            available.append({
                "species": sp,
                "scientific": BOTANICAL_TAXON_MAP.get(sp, sp),
                "image_url": f"/api/sample-image?species={sp}"
            })
    return {"samples": available}

@app.get("/api/sample-image")
async def get_sample_image(species: str):
    test_dir = EXPERIMENT_ROOT / "data" / "test" / species
    if not test_dir.exists():
        raise HTTPException(status_code=404, detail="Species folder not found")
    images = list(test_dir.glob("*.*"))
    if not images:
        raise HTTPException(status_code=404, detail="No images for species")
    return StreamingResponse(open(images[0], "rb"), media_type="image/jpeg")

@app.post("/api/predict")
async def predict_plant(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    try:
        predictions, clean_img = predict_image(pil_img, top_k=5, return_preprocessed=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    inference_ms = round((time.time() - start_time) * 1000, 1)

    enriched_preds = []
    for p in predictions:
        species_name = p["species"]
        enriched_preds.append({
            "rank": p["rank"],
            "species": species_name,
            "scientific_name": BOTANICAL_TAXON_MAP.get(species_name, species_name),
            "confidence": p["confidence"]
        })

    return {
        "top_species": enriched_preds[0]["species"],
        "scientific_name": enriched_preds[0]["scientific_name"],
        "confidence": enriched_preds[0]["confidence"],
        "predictions": enriched_preds,
        "inference_time_ms": inference_ms,
        "preprocessed_image": image_to_base64(clean_img)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
