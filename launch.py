"""
FYP CoreTech: One-Click Master Launcher
========================================
Launches both FloraNet-261 (Port 8000) and PhytoCare Ultra (Port 8001)
concurrently with automatic health checks, Git LFS verification,
optional Cloudflare HTTPS public tunnel, and automatic browser opening.

Usage:
    python launch.py                  # Launch Flagship AI models locally
    python launch.py --tunnel         # Launch with free Cloudflare HTTPS Tunnel (Worldwide Mobile)
    python launch.py --mode baseline  # Launch MobileNetV3 baseline models
    python launch.py --no-browser     # Launch without opening browser tabs
"""

import sys
import os
import re
import time
import signal
import socket
import shutil
import argparse
import subprocess
import webbrowser
import threading
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

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

def get_local_ip() -> str:
    """Detects active local Wi-Fi / LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def find_cloudflared() -> Path:
    """Finds portable or system-installed cloudflared binary."""
    candidates = [
        ROOT_DIR / "bin" / "cloudflared.exe",
        ROOT_DIR.parent / "AppIntegration" / "tools" / "cloudflared.exe",
        ROOT_DIR / "cloudflared.exe"
    ]
    for p in candidates:
        if p.exists():
            return p
    which_p = shutil.which("cloudflared")
    if which_p:
        return Path(which_p)
    return None

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
                        help="Choose 'flagship' (261-species + 31-disease ConvNeXt) or 'baseline' (MobileNetV3)")
    parser.add_argument("--tunnel", "--online", action="store_true",
                        help="Start Cloudflare HTTPS public tunnel for free mobile access anywhere")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser tabs")
    args = parser.parse_args()

    print(BANNER)
    verify_git_lfs()

    python_exe = sys.executable
    local_ip = get_local_ip()

    if args.mode == "flagship":
        id_script = ROOT_DIR / "Plant_Identification_Experiment" / "run_web.py"
        disease_script = ROOT_DIR / "Plant_Disease_Experiment" / "run_web.py"
        id_title = "FloraNet-261 (ConvNeXt-Small + SEA Expansion — 261 Species)"
        disease_title = "PhytoCare Ultra v5.0 (31 Classes + 500-Test Arena)"
    else:
        id_script = ROOT_DIR / "Plant_Identification" / "app" / "main.py"
        disease_script = ROOT_DIR / "Plant_Disease" / "app" / "main.py"
        id_title = "FloraID Baseline (MobileNetV3 — 47 Species)"
        disease_title = "PhytoCare Baseline (MobileNetV3 — 16 Classes)"

    print(f"[*] Mode: {args.mode.upper()}")
    print(f"[*] Local Network IP: {local_ip}")
    print(f"[*] Starting Service 1: {id_title} on Port 8000...")
    p1 = subprocess.Popen([python_exe, str(id_script)], cwd=str(id_script.parent))

    print(f"[*] Starting Service 2: {disease_title} on Port 8001...")
    p2 = subprocess.Popen([python_exe, str(disease_script)], cwd=str(disease_script.parent))

    p_tunnel = None
    tunnel_url = None

    # Graceful shutdown handler
    def cleanup(signum=None, frame=None):
        print("\n\n[*] Shutting down all services cleanly...")
        for p in (p1, p2, p_tunnel):
            if p is not None:
                try:
                    p.terminate()
                    p.wait(timeout=3)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        print("[✓] All services stopped. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("\n[*] Waiting for web engines to initialize...")
    id_ok = wait_for_port(8000, "Species Identification")
    disease_ok = wait_for_port(8001, "Disease Diagnosis")

    # Start Cloudflare Tunnel if requested
    if args.tunnel:
        cf_bin = find_cloudflared()
        if cf_bin is None:
            print("[!] cloudflared.exe was not found. Tunnel cannot be started.")
        else:
            print(f"[*] Initializing Cloudflare Secure Tunnel using {cf_bin.name}...")
            cf_cmd = [str(cf_bin), "tunnel", "--url", "http://127.0.0.1:8001"]
            p_tunnel = subprocess.Popen(
                cf_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
            start_t = time.time()
            for line in iter(p_tunnel.stdout.readline, ""):
                m = url_pattern.search(line)
                if m:
                    tunnel_url = m.group(0)
                    break
                if time.time() - start_t > 25:
                    break

            if tunnel_url:
                target_files = [
                    ROOT_DIR / "cloud_ai_url.txt",
                    ROOT_DIR.parent / "cloud_ai_url.txt",
                    ROOT_DIR.parent / "AppIntegration" / "cloud_ai_url.txt"
                ]
                for tf in target_files:
                    try:
                        tf.parent.mkdir(parents=True, exist_ok=True)
                        with open(tf, "w", encoding="utf-8") as f:
                            f.write(tunnel_url.strip() + "\n")
                    except Exception:
                        pass

    print("\n" + "="*78)
    print(" 🌿 FYP CORETECH AI SERVICES ACTIVE & READY")
    print("="*78)
    print(f"  • Localhost (This PC) : http://localhost:8000  (FloraNet-261)")
    print(f"  • Localhost (This PC) : http://localhost:8001  (PhytoCare Ultra)")
    print(f"  • Local Wi-Fi (Phone) : http://{local_ip}:8001  (PhytoCare Mobile API)")
    if tunnel_url:
        print("-" * 78)
        print("  🌐 CLOUDFLARE PUBLIC HTTPS TUNNEL (WORLDWIDE FREE MOBILE ACCESS):")
        print(f"  • Public HTTPS URL    : {tunnel_url}")
        print(f"  • Mobile API Endpoint : {tunnel_url}/api/predict")
        print(f"  • Saved To            : cloud_ai_url.txt")
        print("-" * 78)
        print("  📱 HOW TO CONNECT MOBILE APP ANYWHERE IN THE WORLD:")
        print("  1. In the UrbanGreen Flutter app, tap Settings on Camera screen.")
        print(f"  2. Paste this URL: {tunnel_url}")
        print("  3. Tap 'Test Ping' to confirm green checkmark, then 'Save & Retry'.")
    print("="*78)
    print("  Press Ctrl+C at any time in this window to stop all services.")
    print("="*78 + "\n")

    if not args.no_browser:
        time.sleep(1.0)
        webbrowser.open("http://localhost:8000")
        webbrowser.open("http://localhost:8001")
        if tunnel_url:
            webbrowser.open(tunnel_url)

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
            if p_tunnel is not None and p_tunnel.poll() is not None:
                print(f"[!] Cloudflare tunnel closed unexpectedly with code {p_tunnel.returncode}")
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
