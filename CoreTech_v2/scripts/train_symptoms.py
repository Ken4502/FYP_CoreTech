"""
CoreTech: Train Unified 6-Class Plant Symptom YOLO11 Detector.
Configured for residential indoor/outdoor foliar symptoms with indoor lighting augmentations:
  - yellowing_chlorosis
  - crispy_brown_edges
  - dark_spots_necrosis
  - white_powdery_residue
  - chewed_holes_stippling
  - healthy_leaf

Usage:
  python train_symptoms.py --data path/to/data.yaml --epochs 50 --batch 16
"""
import sys
import os
import shutil
import argparse
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CORETECH_ROOT = ROOT_DIR.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SYMPTOM_CLASSES = [
    "yellowing_chlorosis",
    "crispy_brown_edges",
    "dark_spots_necrosis",
    "white_powdery_residue",
    "chewed_holes_stippling",
    "healthy_leaf",
]


def ensure_default_data_yaml(yaml_path: Path) -> Path:
    """Creates a template data.yaml for the 6 symptom classes if missing."""
    if not yaml_path.exists():
        data_content = {
            "path": str(yaml_path.parent.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(SYMPTOM_CLASSES)}
        }
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_content, f, sort_keys=False)
        print(f"[*] Generated template symptom dataset config: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 6-Class Foliar Symptom Detector")
    parser.add_argument("--data", type=str, default="", help="Path to data.yaml dataset definition")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model architecture (yolo11n.pt / yolo11s.pt / yolo11m.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Error] 'ultralytics' is not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Resolve data.yaml path
    if args.data:
        data_yaml = Path(args.data)
    else:
        default_dir = ROOT_DIR / "data" / "symptoms_dataset"
        data_yaml = default_dir / "data.yaml"
        ensure_default_data_yaml(data_yaml)

    if not data_yaml.exists():
        print(f"[Error] Dataset configuration not found: {data_yaml}")
        print("Please prepare your dataset in YOLO format with the 6 symptom classes.")
        sys.exit(1)

    print("=" * 70)
    print("  UrbanGreen PhytoCare CoreTech: Training 6-Class Symptom Detector")
    print(f"  Base Architecture : {args.model}")
    print(f"  Dataset Config    : {data_yaml}")
    print(f"  Classes ({len(SYMPTOM_CLASSES)})     : {', '.join(SYMPTOM_CLASSES)}")
    print(f"  Epochs / Batch    : {args.epochs} / {args.batch}")
    print("=" * 70)

    model = YOLO(args.model)

    # Train with augmentations specifically tuned for indoor residential plant conditions:
    # - HSV jitter for warm/cool LED and incandescent lighting
    # - Mosaic and Mixup for multi-symptom co-occurrence
    # - Perspective & rotation for handheld phone angles
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        project=str(REPORTS_DIR / "yolo_symptom_runs"),
        name="symptom_det_v1",
        exist_ok=True,
        # Indoor lighting & orientation augmentations
        hsv_h=0.015,   # Hue variance for lighting differences
        hsv_s=0.30,    # Saturation variance (neutralizing artificial light)
        hsv_v=0.30,    # Value variance (indoor shadows / window backlighting)
        degrees=20.0,  # Rotational tilt variance
        translate=0.1, # Translation
        scale=0.3,     # Multi-scale leaf viewing
        fliplr=0.5,    # Horizontal flip
        flipud=0.2,    # Gentle vertical flip
        mosaic=0.7,    # Mosaic multi-symptom composition
        mixup=0.15,    # Mixup
        cos_lr=True,   # Cosine annealing learning rate schedule
        patience=15,   # Early stopping patience
        save=True,
    )

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    if best_pt.exists():
        dst_pt = ARTIFACTS_DIR / "symptoms_yolo11.pt"
        shutil.copy2(best_pt, dst_pt)
        print(f"\n[✓] Training complete! Best weights saved to: {dst_pt}")
        print(f"[✓] Set SYMPTOM_YOLO_WEIGHTS_PATH={dst_pt} or keep it at {dst_pt} for automatic pickup.")
    else:
        print(f"\n[!] Training finished but {best_pt} was not found.")


if __name__ == "__main__":
    main()
