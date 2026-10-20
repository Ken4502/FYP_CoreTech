"""
FYP CoreTech: One-Click Master Launcher
========================================
Launches both FloraNet-250 (Port 8000) and PhytoCare Ultra (Port 8001)
concurrently with automatic health checks, Git LFS verification,
and automatic browser opening.

Usage:
    python launch.py                  # Launch Flagship AI models (Recommended)
    python launch.py --mode baseline  # Launch MobileNetV3 baseline models
    python launch.py --no-browser     # Launch without opening browser tabs
"""

import sys
import os
import time
import signal
import argparse
import subprocess
import webbrowser
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

BANNER = r"""
=============================================================================
       ______   ______     ______               _____           _     
      |  ____| \ \ / /   |  ____|             / ____|         | |    
      | |__     \ V /    | |__    _   _ _ __ | |     ___  _ __| |__  
      |  __|     > <     |  __|  | | | | '_ \| |    / _ \| '__| '_ \ 
      | |       / . \    | |____ | |_| | |_) | |___| (_) | |  | | | |
      |_|      /_/ \_\   |______(_)__, | .__/ \_____\___/|_|  |_| |_|
                                   __/ | |                           
                                  |___/|_|                           
=============================================================================
"""

def verify_git_lfs():
    """Checks if critical model weights are Git LFS pointers (<10KB) and auto-pulls them."""
    checkpoints = [
        ROOT_DIR / "Plant_Disease_Experiment" / "artifacts" / "best_model.pth",
        ROOT_DIR / "Plant_Identification_Experiment" / "artifacts" / "best_model.pth"
    ]
    needs_pull = False
    for ckpt in checkpoints:
        if ckpt.exists() and ckpt.stat().st_size < 10240:
            needs_pull = True
            print(f"[!] Detected Git LFS pointer file for: {ckpt.name} ({ckpt.stat().st_size} bytes)")

    if needs_pull:
        print("[*] Pulling full binary model weights from Git LFS...")
        try:
            subprocess.run(["git", "lfs", "pull"], cwd=str(ROOT_DIR), check=True)
            print("[✓] Model weights successfully downloaded via Git LFS.\n")
        except Exception as e:
            print(f"[X] Automatic 'git lfs pull' failed: {e}")
            print("    Please ensure Git LFS is installed (https://git-lfs.github.com) and run: git lfs pull\n")

def wait_for_port(port: int, name: str, timeout: int = 40) -> bool:
    """Polls an HTTP endpoint until it answers with HTTP 200 or timeout is reached."""
    url = f"http://127.0.0.1:{port}/"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if resp.status in (200, 301, 302, 404):
                    return True
        except Exception:
            time.sleep(0.8)
    return False

def main():
    parser = argparse.ArgumentParser(description="FYP CoreTech One-Click Multi-Service Launcher")
    parser.add_argument("--mode", choices=["flagship", "baseline"], default="flagship",
                        help="Choose 'flagship' (250-species + 31-disease ConvNeXt) or 'baseline' (MobileNetV3)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser tabs")
    args = parser.parse_args()

    print(BANNER)
    verify_git_lfs()

    python_exe = sys.executable

    if args.mode == "flagship":
        id_script = ROOT_DIR / "Plant_Identification_Experiment" / "run_web.py"
        disease_script = ROOT_DIR / "Plant_Disease_Experiment" / "run_web.py"
        id_title = "FloraNet-250 (ConvNeXt-Swin Ensemble — 250 Species)"
        disease_title = "PhytoCare Ultra v5.0 (31 Classes + 500-Test Arena)"
    else:
        id_script = ROOT_DIR / "Plant_Identification" / "app" / "main.py"
        disease_script = ROOT_DIR / "Plant_Disease" / "app" / "main.py"
        id_title = "FloraID Baseline (MobileNetV3 — 47 Species)"
        disease_title = "PhytoCare Baseline (MobileNetV3 — 16 Classes)"

    print(f"[*] Mode: {args.mode.upper()}")
    print(f"[*] Starting Service 1: {id_title} on Port 8000...")
    p1 = subprocess.Popen([python_exe, str(id_script)], cwd=str(id_script.parent))

    print(f"[*] Starting Service 2: {disease_title} on Port 8001...")
    p2 = subprocess.Popen([python_exe, str(disease_script)], cwd=str(disease_script.parent))

    # Graceful shutdown handler
    def cleanup(signum=None, frame=None):
        print("\n\n[*] Shutting down services cleanly...")
        for p in (p1, p2):
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                p.kill()
        print("[✓] All services stopped. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("\n[*] Waiting for web engines to initialize...")
    id_ok = wait_for_port(8000, "Species Identification")
    disease_ok = wait_for_port(8001, "Disease Diagnosis")

    print("\n" + "="*75)
    print(" 🌿 FYP CORETECH AI SERVICES ACTIVE & READY")
    print("="*75)
    print(f"  • [Port 8000] Species Identification : http://localhost:8000  ({'ONLINE' if id_ok else 'STARTING'})")
    print(f"  • [Port 8001] Disease & Treatment     : http://localhost:8001  ({'ONLINE' if disease_ok else 'STARTING'})")
    if args.mode == "flagship":
        print(f"  • [Port 8001] 500-Specimen Arena     : http://localhost:8001/benchmark")
    print("="*75)
    print("  Press Ctrl+C at any time in this window to stop both servers.")
    print("="*75 + "\n")

    if not args.no_browser:
        time.sleep(1.0)
        webbrowser.open("http://localhost:8000")
        webbrowser.open("http://localhost:8001")

    # Monitor running processes
    try:
        while True:
            time.sleep(1)
            if p1.poll() is not None:
                print(f"[!] Service 1 (Port 8000) exited unexpectedly with code {p1.returncode}")
                cleanup()
            if p2.poll() is not None:
                print(f"[!] Service 2 (Port 8001) exited unexpectedly with code {p2.returncode}")
                cleanup()
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()

