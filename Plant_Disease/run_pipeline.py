"""
Master Pipeline Runner for Plant Disease Diagnosis Project.
Orchestrates:
  1. Dataset verification & link organization (Train/Val/Test)
  2. MobileNetV3 (FloraNet-V3) transfer learning & fine-tuning with class balancing
  3. Comprehensive test set evaluation & confusion matrix generation
  4. Model export to ONNX & TorchScript
  5. Launching local web prototype with treatment advisory
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src import config
from organize_dataset import organize_dataset
from src.train import run_training
from src.evaluate import evaluate_test_set
from src.export import export_models

def main():
    parser = argparse.ArgumentParser(description="PhytoCare - Plant Disease Pipeline Runner")
    parser.add_argument("--reorganize", action="store_true", help="Force dataset reorganization")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch without botanical backbone")
    parser.add_argument("--epochs", type=int, default=config.TOTAL_EPOCHS, help="Number of training epochs")
    parser.add_argument("--eval_only", action="store_true", help="Only run test set evaluation")
    parser.add_argument("--export_only", action="store_true", help="Only run model export")
    parser.add_argument("--serve", action="store_true", help="Launch web prototype after pipeline completes")
    parser.add_argument("--port", type=int, default=8001, help="Web server port (default: 8001)")
    args = parser.parse_args()

    print("=" * 65)
    print("🔬 PhytoCare: Indoor Plant Disease Diagnosis Pipeline")
    print("=" * 65)

    # 1. Dataset Organization
    data_exists = (config.TRAIN_DIR.exists() and any(config.TRAIN_DIR.iterdir()))
    if not args.eval_only and not args.export_only:
        if args.reorganize or not data_exists:
            print("\n>>> Step 1/4: Organizing & Linking Dataset...")
            organize_dataset(
                source_dir=str(PROJECT_ROOT.parent / "Datasets" / "indoor"),
                target_dir=str(config.DATA_DIR),
                artifacts_dir=str(config.ARTIFACTS_DIR)
            )
        else:
            print("\n>>> Step 1/4: Dataset already organized. Skipping (use --reorganize to force).")

    # 2. Model Training
    model_exists = config.BEST_MODEL_PATH.exists()
    if not args.eval_only and not args.export_only:
        if args.retrain or not model_exists:
            print(f"\n>>> Step 2/4: Training FloraNet-V3 Model ({args.epochs} epochs)...")
            run_training(pretrained=not args.scratch, epochs=args.epochs)
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

    print("\n" + "=" * 65)
    print("🎉 Pipeline Execution Completed Successfully!")
    print(f"  Best Model:     {config.BEST_MODEL_PATH}")
    print(f"  ONNX Export:    {config.ONNX_MODEL_PATH}")
    print(f"  TorchScript:    {config.TORCHSCRIPT_MODEL_PATH}")
    print(f"  Confusion Map:  {config.CONFUSION_MATRIX_PATH}")
    print("=" * 65)

    # 5. Launch Web Prototype
    if args.serve:
        import uvicorn
        print(f"\n🚀 Launching PhytoCare Web Prototype on http://127.0.0.1:{args.port} ...")
        app_path = PROJECT_ROOT / "app"
        uvicorn.run("main:app", host="127.0.0.1", port=args.port, reload=False, app_dir=str(app_path))

if __name__ == "__main__":
    main()

