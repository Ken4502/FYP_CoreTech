"""
UrbanGreen CoreTech: YOLO11s Transfer Learning Pipeline for 6-Class Symptom Detector.
Auto-verifies dataset before training, monitors GPU/VRAM, applies indoor-resilient augmentations,
and logs final mAP50, Precision, and Recall metrics.

Usage:
  python train_symptoms.py [--data path/to/data.yaml] [--epochs 60] [--batch -1]
"""
import sys
import os
import argparse
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CORETECH_ROOT = SCRIPT_DIR.parent.parent

# Import verifier
try:
    from verify_dataset import verify_dataset
except ImportError:
    sys.path.append(str(SCRIPT_DIR))
    from verify_dataset import verify_dataset


def check_gpu() -> str:
    """Detects available computing hardware and logs GPU VRAM details."""
    if torch.cuda.is_available():
        device_id = 0
        device_name = torch.cuda.get_device_name(device_id)
        vram_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
        print(f"[✓] CUDA GPU Detected: {device_name} ({vram_gb:.2f} GB VRAM)")
        return "0"
    else:
        print("[!] WARNING: No CUDA GPU detected. Training will run on CPU (significantly slower).")
        return "cpu"


def main():
    parser = argparse.ArgumentParser(description="UrbanGreen CoreTech: Train YOLO11s Symptom Detector")
    parser.add_argument(
        "--data",
        type=str,
        default=str(SCRIPT_DIR / "datasets" / "symptom_dataset" / "data.yaml"),
        help="Path to data.yaml dataset config"
    )
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Base model weights (yolo11s.pt)")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto-batch)")
    parser.add_argument("--allow-no-negatives", action="store_true", help="Bypass negative check in verifier")
    parser.add_argument("--allow-empty-classes", action="store_true", default=True, help="Bypass empty class check in verifier")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"[FATAL] data.yaml not found at {data_yaml}. Run prepare_dataset.py first!")
        sys.exit(1)

    # 1. Run Automated Pre-Flight Verification Gate
    print("[*] Running pre-flight dataset verification gate...")
    dataset_dir = data_yaml.parent
    is_valid = verify_dataset(
        dataset_dir,
        allow_no_negatives=args.allow_no_negatives,
        allow_empty_classes=args.allow_empty_classes
    )
    if not is_valid:
        print("[FATAL] Pre-flight dataset verification failed! Aborting training.")
        sys.exit(1)

    # 2. Check Hardware
    device_str = check_gpu()

    # 3. Import Ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[FATAL] 'ultralytics' library not installed. Install via pip install ultralytics")
        sys.exit(1)

    # 4. Initialize Base Model
    print(f"\n[*] Loading base architecture: {args.model}")
    model = YOLO(args.model)

    project_dir = SCRIPT_DIR / "runs" / "detect"
    run_name = "symptom_yolo11s"

    print("=" * 70)
    print("  STARTING YOLO11s SYMPTOM TRANSFER LEARNING")
    print(f"  Data Config  : {data_yaml}")
    print(f"  Epochs       : {args.epochs} (Patience: {args.patience})")
    print(f"  Image Size   : {args.imgsz}")
    print(f"  Batch Size   : {args.batch} (Auto-batch)")
    print(f"  Output Dir   : {project_dir / run_name}")
    print("=" * 70)

    # 5. Train with Indoor-Resilient Augmentations
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device_str,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        # Indoor-resilient augmentations
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        fliplr=0.5,
        scale=0.5,
        mosaic=1.0,
        save=True,
    )

    # 6. Evaluate and Print Final Validation Metrics
    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"

    print("\n" + "=" * 70)
    print("  TRAINING SESSION FINISHED")
    print("=" * 70)
    if best_pt.exists():
        print(f"[✓] Best weights successfully saved to:\n    {best_pt}")
    else:
        print(f"[!] Warning: best.pt was not found in {save_dir / 'weights'}")

    try:
        metrics = model.val()
        map50 = getattr(metrics.box, "map50", None)
        mp = getattr(metrics.box, "mp", None)
        mr = getattr(metrics.box, "mr", None)

        print("\n  FINAL VALIDATION METRICS:")
        if map50 is not None:
            print(f"    mAP50     : {map50:.4f} ({map50*100:.1f}%)")
        if mp is not None:
            print(f"    Precision : {mp:.4f} ({mp*100:.1f}%)")
        if mr is not None:
            print(f"    Recall    : {mr:.4f} ({mr*100:.1f}%)")
    except Exception as e:
        print(f"[!] Could not calculate validation summary: {e}")

    print("=" * 70)
    print("Next step: Verify weights using test_inference.py before deploying to models/symptom_yolo11s.pt.")


if __name__ == "__main__":
    main()
