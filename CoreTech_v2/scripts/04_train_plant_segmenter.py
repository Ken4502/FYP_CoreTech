"""
CoreTech v2 Phase 2: Train Dominant Plant Instance Segmenter (YOLO11m-Seg) on RTX 3060.
Learns to separate overlapping plants, exclude pots/soil/tables, and reject indoor non-plant scenes.
"""
import sys
import shutil
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))
from configs.v2_config import SEG_DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR


def main():
    from ultralytics import YOLO
    data_yaml = SEG_DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Segmentation dataset yaml not found: {data_yaml}. Run 03_build_segmentation_and_lesion_sets.py first.")

    model = YOLO("yolo11m-seg.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=45,
        patience=12,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        amp=True,
        project=str(REPORTS_DIR / "yolo_seg_runs"),
        name="plant_seg_v2",
        exist_ok=True,
        mosaic=0.65,
        mixup=0.12,
        degrees=15.0,
        scale=0.35,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.35,
        hsv_v=0.35,
        cos_lr=True,
    )
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    dst = ARTIFACTS_DIR / "yolo_plant_seg.pt"
    shutil.copy2(best_pt, dst)
    print(f"[✓] Saved Stage 1B Dominant Plant Segmenter to {dst}")


if __name__ == "__main__":
    main()
