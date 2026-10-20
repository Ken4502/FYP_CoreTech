"""
Launcher for PhytoCare Plant Disease Diagnostic Web Application (Port 8001)
Runs concurrently with FloraNet Plant Identification (Port 8000).
"""

import sys
from pathlib import Path

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
    print("=" * 65)
    print("[PhytoCare Ultra] Plant Disease Clinical Diagnosis & Advisory")
    print("[PhytoCare Ultra] Model: ConvNeXt-Small 384px + Swin-T Ensemble")
    print("[PhytoCare Ultra] Web Service: http://127.0.0.1:8001")
    print("=" * 65)
    uvicorn.run("web.app:app", host="127.0.0.1", port=8001, log_level="info")

