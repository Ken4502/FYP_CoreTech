"""
CoreTech v2 Master Training Orchestrator.
Chains remaining model training stages sequentially on RTX 3060:
  1. Pathology MIL Pillar 2: DINOv2-Base (15 epochs)
  2. Botanical ID Pillar 1: BioCLIP-2 ViT-L/14 (15 epochs)
  3. Botanical ID Pillar 2: ConvNeXt-Base (18 epochs)
  4. Botanical ID Pillar 3: EVA-02 Base (14 epochs)
  5. Out-of-fold Calibration & Strictly Unseen Holdout Benchmark
"""
import sys
import time
import subprocess
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = Path(sys.executable)

QUEUE = [
    {
        "name": "Stage 3B: DINOv2-Base Multi-Tile MIL (Pathology)",
        "cmd": [str(PYTHON_EXE), str(V2_ROOT / "scripts" / "06_train_pathology_mil.py"), "--arch", "dinov2", "--epochs", "15"],
        "artifact": V2_ROOT / "artifacts" / "pathology_dinov2_mil.pth",
    },
    {
        "name": "Stage 2A: BioCLIP-2 ViT-L/14 (Botanical ID)",
        "cmd": [str(PYTHON_EXE), str(V2_ROOT / "scripts" / "05_train_flora_models.py"), "--arch", "bioclip2", "--epochs", "15"],
        "artifact": V2_ROOT / "artifacts" / "flora_bioclip2.pth",
    },
    {
        "name": "Stage 2B: ConvNeXt-Base (Botanical ID)",
        "cmd": [str(PYTHON_EXE), str(V2_ROOT / "scripts" / "05_train_flora_models.py"), "--arch", "convnext", "--epochs", "18"],
        "artifact": V2_ROOT / "artifacts" / "flora_convnext.pth",
    },
    {
        "name": "Stage 2C: EVA-02 Base (Botanical ID)",
        "cmd": [str(PYTHON_EXE), str(V2_ROOT / "scripts" / "05_train_flora_models.py"), "--arch", "eva02", "--epochs", "14"],
        "artifact": V2_ROOT / "artifacts" / "flora_eva02.pth",
    },
    {
        "name": "Phase 5: Calibration & Unseen Holdout Benchmark",
        "cmd": [str(PYTHON_EXE), str(V2_ROOT / "scripts" / "08_calibrate_and_benchmark.py")],
        "artifact": V2_ROOT / "artifacts" / "flora_calibration.json",
    },
]

def main():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting CoreTech v2 Master Training Queue...")
    print(f"Python interpreter: {PYTHON_EXE}")
    print(f"Root: {V2_ROOT}\n")

    for idx, item in enumerate(QUEUE, 1):
        name = item["name"]
        cmd = item["cmd"]
        artifact = item["artifact"]

        if artifact.exists():
            print(f"\n{'='*70}")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ALREADY COMPLETE] Skipping: {name}")
            print(f"  Artifact verified on disk: {artifact} ({artifact.stat().st_size / 1e6:.1f} MB)")
            print(f"{'='*70}\n", flush=True)
            continue

        print(f"\n{'='*70}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{idx}/{len(QUEUE)}] Launching: {name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Target Artifact: {artifact}")
        print(f"{'='*70}\n", flush=True)

        t_start = time.time()
        ret = subprocess.run(cmd, cwd=str(V2_ROOT))
        elapsed_min = (time.time() - t_start) / 60.0

        if ret.returncode != 0:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Stage failed with return code {ret.returncode}: {name}", flush=True)
            sys.exit(ret.returncode)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [SUCCESS] Finished in {elapsed_min:.1f} min: {name}")
        if artifact.exists():
            print(f"  Artifact verified: {artifact} ({artifact.stat().st_size / 1e6:.1f} MB)", flush=True)
        else:
            print(f"  [Warning] Expected artifact {artifact} not found!", flush=True)

        # 60s Inter-Stage Thermal Cooldown
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [❄️ Thermal Rest] Pausing for 60s to let heatsinks cool before next stage...", flush=True)
        time.sleep(60)

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ALL STAGES IN MASTER QUEUE COMPLETED SUCCESSFULLY!", flush=True)

if __name__ == "__main__":
    main()
