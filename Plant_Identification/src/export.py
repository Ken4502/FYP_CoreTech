"""
Model export module: Exports trained MobileNetV3 to ONNX and TorchScript.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from src import config
from src.model import build_model

def export_models():
    print(f"=== Exporting Model Artifacts ===")
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {config.BEST_MODEL_PATH}")

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    # 1. Export TorchScript
    print(f"Exporting TorchScript model to {config.TORCHSCRIPT_MODEL_PATH}...")
    try:
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(str(config.TORCHSCRIPT_MODEL_PATH))
        print("  [OK] TorchScript export successful.")
    except Exception as e:
        print(f"  [Error] TorchScript export failed: {e}")

    # 2. Export ONNX
    print(f"Exporting ONNX model to {config.ONNX_MODEL_PATH}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(config.ONNX_MODEL_PATH),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False
        )
        print("  [OK] ONNX export successful.")
    except Exception as e:
        print(f"  [Error] ONNX export failed: {e}")

if __name__ == "__main__":
    export_models()

