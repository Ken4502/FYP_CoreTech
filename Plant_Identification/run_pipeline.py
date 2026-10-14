"""
Master Pipeline Runner for Plant Identification Project.
Orchestrates:
  1. Data verification & stratified 80/10/10 splitting
  2. MobileNetV3-Large transfer learning & fine-tuning
  3. Comprehensive test set evaluation & confusion matrix generation
  4. Model export to ONNX & TorchScript
  5. Launching local web prototype
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from organize_dataset import organize_dataset
from src.train import run_training
from src.evaluate import evaluate_test_set
from src.export import export_models

def main():
    parser = argparse.ArgumentParser(description="FloraID - Master Pipeline Runner")
    parser.add_argument("--reorganize", action="store_true", help="Force dataset reorganization")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--eval_only", action="store_true", help="Only run test set evaluation")
    parser.add_argument("--export_only", action="store_true", help="Only run model export")
    parser.add_argument("--serve", action="store_true", help="Launch web prototype after pipeline completes")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    args = parser.parse_args()

    print("=" * 60)
    print("🌿 FloraID: Plant Identification Deep Learning Pipeline")
    print("=" * 60)

    # 1. Dataset Organization
    data_exists = (config.TRAIN_DIR.exists() and any(config.TRAIN_DIR.iterdir()))
    if not args.eval_only and not args.export_only:
        if args.reorganize or not data_exists:
            print("\n>>> Step 1/4: Reorganizing & Verifying Dataset (80/10/10)...")
            organize_dataset(
                source_dir=str(PROJECT_ROOT.parent / "Datasets" / "house_plant_species"),
                target_dir=str(config.DATA_DIR),
                artifacts_dir=str(config.ARTIFACTS_DIR)
            )
        else:
            print("\n>>> Step 1/4: Dataset already organized. Skipping (use --reorganize to force).")

    # 2. Model Training
    model_exists = config.BEST_MODEL_PATH.exists()
    if not args.eval_only and not args.export_only:
        if args.retrain or not model_exists:
            print("\n>>> Step 2/4: Training MobileNetV3-Large Model...")
            run_training()
        else:
            print(f"\n>>> Step 2/4: Found existing model at {config.BEST_MODEL_PATH}. (use --retrain to force).")

    # 3. Evaluation
    if not args.export_only:
        print("\n>>> Step 3/4: Evaluating on Test Set...")
        evaluate_test_set()

    # 4. Model Export
    if not args.eval_only:
        print("\n>>> Step 4/4: Exporting Model to ONNX and TorchScript...")
        export_models()

    print("\n" + "=" * 60)
    print("🎉 Pipeline Completed Successfully!")
    print(f"  Best Model:     {config.BEST_MODEL_PATH}")
    print(f"  ONNX Export:    {config.ONNX_MODEL_PATH}")
    print(f"  Confusion Map:  {config.CONFUSION_MATRIX_PATH}")
    print("=" * 60)

    # 5. Launch Web Prototype
    if args.serve:
        import uvicorn
        print(f"\n🚀 Launching FloraID Web Prototype on http://127.0.0.1:{args.port} ...")
        app_path = PROJECT_ROOT / "app"
        uvicorn.run("main:app", host="127.0.0.1", port=args.port, reload=False, app_dir=str(app_path))

if __name__ == "__main__":
    main()

