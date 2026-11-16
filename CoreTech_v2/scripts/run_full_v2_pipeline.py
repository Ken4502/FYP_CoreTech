"""
Master Sequential GPU Training & Calibration Orchestrator for CoreTech v2.
Runs all GPU training phases safely on RTX 3060 with checkpoint resume & full logging:
  1. Stage 1B: YOLO11m-Seg Dominant Plant Segmenter
  2. Stage 3B: YOLO11m-Det Surface Lesion Detector
  3. Complete Phase 1B iNaturalist Abiotic/Pest expansion + splits (once Phase 1A finishes)
  4. Stage 3A: Surface Pathology Multi-Tile MIL Ensemble (ConvNeXt-Base + DINOv2-Base)
  5. Stage 2: Botanical ID Foundation Ensemble (BioCLIP-2 + ConvNeXt-Base + EVA-02-448)
  6. Stage 5: Out-of-Fold Calibration + Unseen Holdout Benchmark
"""
import sys
import time
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

V2_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = V2_ROOT / "scripts"
ARTIFACTS_DIR = V2_ROOT / "artifacts"
REPORTS_DIR = V2_ROOT / "reports"
PYTHON_EXE = Path(r"c:\Users\User\Desktop\FYP\CoreTech\Plant_Identification\.venv\Scripts\python.exe")
MASTER_LOG = REPORTS_DIR / "master_pipeline_v2.log"


def log_step(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_stage(name: str, cmd: list, skip_if_exists: Path = None):
    if skip_if_exists is not None and skip_if_exists.exists():
        log_step(f"[SKIP] {name}: checkpoint already exists ({skip_if_exists.name})")
        return True
    log_step(f"[START] {name}: {' '.join(str(x) for x in cmd)}")
    t0 = time.time()
    proc = subprocess.run([str(PYTHON_EXE)] + [str(c) for c in cmd], cwd=str(V2_ROOT))
    hrs = (time.time() - t0) / 3600.0
    if proc.returncode != 0:
        log_step(f"[ERROR] {name} failed with exit code {proc.returncode} after {hrs:.2f}h")
        return False
    log_step(f"[DONE] {name} completed cleanly in {hrs:.2f}h")
    return True


def main():
    log_step("=" * 72)
    log_step("CORETECH V2 MASTER TRAINING ORCHESTRATOR STARTED")
    log_step("=" * 72)

    # 1. Dominant Plant Segmenter (ready as soon as 03_build_segmentation_and_lesion_sets finishes)
    run_stage("Stage 1B YOLO11m-Seg Plant Isolator", [SCRIPTS_DIR / "04_train_plant_segmenter.py"], skip_if_exists=ARTIFACTS_DIR / "yolo_plant_seg.pt")

    # 2. Surface Lesion Detector
    run_stage("Stage 3B YOLO11m-Det Lesion Detector", [SCRIPTS_DIR / "07_train_lesion_detector.py"], skip_if_exists=ARTIFACTS_DIR / "yolo_lesion_det.pt")

    # 3. Finish Phase 1B splits if not yet materialized
    if not (V2_ROOT / "data" / "surface_pathology" / "test").exists():
        run_stage("Phase 1B Pathology Finalize & Splits", [SCRIPTS_DIR / "02_build_pathology_corpus.py", "--stage", "all"])

    # 4. Surface Pathology MIL Models
    run_stage("Stage 3A Pathology ConvNeXt-Base MIL", [SCRIPTS_DIR / "06_train_pathology_mil.py", "--arch", "convnext"], skip_if_exists=ARTIFACTS_DIR / "pathology_convnext_mil.pth")
    run_stage("Stage 3A Pathology DINOv2-Base MIL", [SCRIPTS_DIR / "06_train_pathology_mil.py", "--arch", "dinov2"], skip_if_exists=ARTIFACTS_DIR / "pathology_dinov2_mil.pth")

    # 5. Ensure Phase 1A Botanical ID corpus finished before training Stage 2 ensemble
    if not (V2_ROOT / "reports" / "id_corpus_manifest.json").exists():
        run_stage("Phase 1A Botanical ID Corpus Completion", [SCRIPTS_DIR / "01_expand_and_build_id_corpus.py"])

    # 6. Foundation Botanical ID Ensemble (3 heterogeneous models)
    run_stage("Stage 2 Flora BioCLIP-2 ViT-L/14", [SCRIPTS_DIR / "05_train_flora_models.py", "--arch", "bioclip2"], skip_if_exists=ARTIFACTS_DIR / "flora_bioclip2.pth")
    run_stage("Stage 2 Flora ConvNeXt-Base 384", [SCRIPTS_DIR / "05_train_flora_models.py", "--arch", "convnext"], skip_if_exists=ARTIFACTS_DIR / "flora_convnext.pth")
    run_stage("Stage 2 Flora EVA-02 Base 448", [SCRIPTS_DIR / "05_train_flora_models.py", "--arch", "eva02"], skip_if_exists=ARTIFACTS_DIR / "flora_eva02.pth")

    # 7. Final Calibration & Leak-Free Benchmark
    run_stage("Phase 5 Calibration & Holdout Benchmark", [SCRIPTS_DIR / "08_calibrate_and_benchmark.py"])
    log_step("ALL CORETECH V2 TRAINING & VERIFICATION STAGES COMPLETE!")


if __name__ == "__main__":
    main()
