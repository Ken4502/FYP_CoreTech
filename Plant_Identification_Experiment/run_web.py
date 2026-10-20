"""
Launcher for FloraNet-100 Interactive Web Application
"""
import sys
from pathlib import Path

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

EXPERIMENT_ROOT = Path(__file__).resolve().parent
CORETECH_ROOT = EXPERIMENT_ROOT.parent

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(CORETECH_ROOT) not in sys.path:
    sys.path.insert(0, str(CORETECH_ROOT))

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("[FloraNet-250] Interactive Botanical Web App")
    print("[FloraNet-250] Model: MobileNetV3-Large (250 Botanical Classes, RTX 3060 AMP)")
    print("[FloraNet-250] Web Server: http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, log_level="info")

