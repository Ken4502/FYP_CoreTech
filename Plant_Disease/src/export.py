"""
Model export module: Exports trained FloraNet-Hierarchical model to TorchScript and ONNX.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from src import config
from src.model import build_model

class FlattenedHierarchicalWrapper(nn.Module):
    """Wraps FloraNetHierarchical into a flat tuple output for standard ONNX/TorchScript tracers."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        species_logits, disease_outputs = self.model(x)
        # Returns tuple of outputs: species_logits + [aloe, cactus, money, snake, spider]
        return (
            species_logits,
            disease_outputs["Aloe"],
            disease_outputs["Cactus"],
            disease_outputs["Money_Plant"],
            disease_outputs["Snake_Plant"],
            disease_outputs["Spider_Plant"]
        )

def export_models():
    print("=== Exporting Hierarchical Plant Disease Model Artifacts ===")
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {config.BEST_MODEL_PATH}")

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location="cpu")
    plant_groups = checkpoint.get("plant_groups", config.PLANT_GROUPS)

    model = build_model(plant_groups=plant_groups)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wrapped_model = FlattenedHierarchicalWrapper(model)
    wrapped_model.eval()

    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    # 1. Export TorchScript
    print(f"Exporting TorchScript model to {config.TORCHSCRIPT_MODEL_PATH}...")
    try:
        traced_module = torch.jit.trace(wrapped_model, dummy_input)
        traced_module.save(str(config.TORCHSCRIPT_MODEL_PATH))
        print("  [OK] TorchScript export successful.")
    except Exception as e:
        print(f"  [Error] TorchScript export failed: {e}")

    # 2. Export ONNX
    print(f"Exporting ONNX model to {config.ONNX_MODEL_PATH}...")
    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            str(config.ONNX_MODEL_PATH),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["species", "aloe", "cactus", "money_plant", "snake_plant", "spider_plant"],
            dynamic_axes={"input": {0: "batch_size"}},
            dynamo=False
        )
        print("  [OK] ONNX export successful.")
    except Exception as e:
        print(f"  [Error] ONNX export failed: {e}")

if __name__ == "__main__":
    export_models()
