"""
CoreTech v2 Phase 4B: Train Surface Lesion & Disease Localizer (YOLO11m-Det) on RTX 3060.
Replaces ExG green color filtering + coarse Grad-CAM heatmaps with trained spatial bounding boxes
on fungal spots, blights, rusts, mildews, and healthy leaves.
"""
import sys
import shutil
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))
from configs.v2_config import LESION_DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR


def main():
    from ultralytics import YOLO
    data_yaml = LESION_DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Lesion detector dataset not found: {data_yaml}")

    model = YOLO("yolo11m.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=60,
        patience=15,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        amp=True,
        project=str(REPORTS_DIR / "yolo_lesion_runs"),
        name="lesion_det_v2",
        exist_ok=True,
        mosaic=0.6,
        mixup=0.1,
        degrees=15.0,
        fliplr=0.5,
        flipud=0.25,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.25,
        cos_lr=True,
    )
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    dst = ARTIFACTS_DIR / "yolo_lesion_det.pt"
    shutil.copy2(best_pt, dst)
    print(f"[✓] Saved Stage 3B Surface Lesion Detector to {dst}")


if __name__ == "__main__":
    main()
