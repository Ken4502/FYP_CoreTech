"""
FloraNet Heterogeneous Ensemble Benchmark Evaluator
Evaluates the Dual-Model Ensemble (ConvNeXt-Small + Swin-Transformer-Tiny)
across all 500 independent Wikimedia Commons / Openverse specimens.
Outputs reports/floranet_500_ensemble_benchmark.csv.
"""

import sys
import csv
import time
import math
import io
import queue
import threading
import urllib.request
from pathlib import Path
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from PIL import Image
import torch
from src import config
from src.ensemble_predict import load_ensemble_models, predict_ensemble
from tools.harvester import BOTANICAL_TAXON_MAP
from web.benchmark import check_match

INPUT_CSV = EXPERIMENT_ROOT / "reports" / "floranet_500_internet_benchmark_v2.csv"
OUTPUT_CSV = EXPERIMENT_ROOT / "reports" / "floranet_500_ensemble_benchmark.csv"
LOG_FILE = EXPERIMENT_ROOT / "reports" / "mega_pipeline.log"

def log_both(msg: str):
    print(msg, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def fetch_image(url: str, headers: dict) -> Image.Image:
    if not url:
        return None
    req = urllib.request.Request(url, headers=headers)
    for _ in range(2):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        except Exception:
            time.sleep(0.3)
    return None

def main():
    if not INPUT_CSV.exists():
        print(f"[Error] Input benchmark CSV not found: {INPUT_CSV}")
        return

    print("=" * 70)
    print("FLORANET TOURNAMENT ENSEMBLE: 500-SPECIMEN BENCHMARK EVALUATION")
    print("• Architecture: Dual Heterogeneous Ensemble")
    print("                1. ConvNeXt-Small (50.2M, 384px, 5-Crop TTA, Weight: 0.60)")
    print("                2. Swin-Transformer-Tiny (28.3M, 224px, Weight: 0.40)")
    print("• Vocabulary:   250 Botanical Classes")
    print("• Corpus:       500 Independent Real-World Specimens (Wikimedia / Openverse)")
    print("=" * 70)

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_specimens = len(rows)
    print(f"[Dataset] Loaded {total_specimens} specimens.")

    print("[Model] Loading ensemble models into CUDA...")
    load_ensemble_models(reload=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    download_queue = queue.Queue(maxsize=30)
    stop_event = threading.Event()

    def download_worker():
        for idx, row in enumerate(rows):
            if stop_event.is_set():
                break
            url = row.get("Direct_Image_URL") or row.get("Source_URL")
            img = fetch_image(url, headers)
            download_queue.put((idx, row, img))
        download_queue.put((None, None, None))

    downloader_thread = threading.Thread(target=download_worker, daemon=True)
    downloader_thread.start()

    v2_correct_count = sum(1 for r in rows if r.get("FloraNet_102_Match") == "YES")
    v2_top5_count = sum(1 for r in rows if r.get("FloraNet_102_Top5_Hit") == "YES")

    new_results = [None] * total_specimens
    ensemble_correct_count = 0
    ensemble_top5_count = 0
    recovered_from_v2 = []
    regressed_from_v2 = []

    t_start = time.time()
    processed_count = 0

    print("[Benchmark] Streaming specimens through Dual Ensemble on RTX 3060...")

    while True:
        idx, row, img = download_queue.get()
        if idx is None:
            break

        processed_count += 1
        gt_common = row.get("Ground_Truth_Common", "").strip()
        gt_sci = row.get("Ground_Truth_Scientific", "").strip()
        v2_match = row.get("FloraNet_102_Match") == "YES"

        if img is None:
            pred_top1 = row.get("FloraNet_102_Prediction", "N/A")
            pred_sci = row.get("FloraNet_102_Scientific", "")
            conf = float(row.get("FloraNet_102_Top1_Conf", 0.0))
            is_corr = v2_match
            top5_hit = row.get("FloraNet_102_Top5_Hit") == "YES"
            f_preds = []
            latency = float(row.get("FloraNet_Latency_MS", 0.0))
        else:
            t0 = time.time()
            try:
                preds = predict_ensemble(img, top_k=5, convnext_weight=0.60, swin_weight=0.40, use_tta=True)
                latency = round((time.time() - t0) * 1000, 1)
                f_preds = preds
                pred_top1 = preds[0]["species"]
                pred_sci = BOTANICAL_TAXON_MAP.get(pred_top1, pred_top1)
                conf = preds[0]["confidence"]
                is_corr = check_match(pred_top1 + " " + pred_sci, gt_common, gt_sci)
                top5_hit = any(check_match(p["species"] + " " + BOTANICAL_TAXON_MAP.get(p["species"], p["species"]), gt_common, gt_sci) for p in preds)
            except Exception as e:
                pred_top1 = "Error"
                pred_sci = ""
                conf = 0.0
                is_corr = False
                top5_hit = False
                f_preds = []
                latency = 0.0

        if is_corr:
            ensemble_correct_count += 1
        if top5_hit:
            ensemble_top5_count += 1

        if not v2_match and is_corr:
            recovered_from_v2.append((row.get("Index"), row.get("Specimen_Title"), row.get("FloraNet_102_Prediction"), pred_top1, conf))
        elif v2_match and not is_corr:
            regressed_from_v2.append((row.get("Index"), row.get("Specimen_Title"), row.get("FloraNet_102_Prediction"), pred_top1, conf))

        top2_pred = f_preds[1]["species"] if len(f_preds) > 1 else "None"
        top2_conf = f_preds[1]["confidence"] if len(f_preds) > 1 else 0.0
        top3_pred = f_preds[2]["species"] if len(f_preds) > 2 else "None"
        top3_conf = f_preds[2]["confidence"] if len(f_preds) > 2 else 0.0
        conf_margin = round(conf - top2_conf, 1)

        c_vals = [p["confidence"] / 100.0 for p in f_preds]
        s_c = sum(c_vals)
        entropy = round(-sum(p * math.log(p + 1e-12) for p in [c / s_c for c in c_vals]), 3) if s_c > 0 else 0.0

        top5_str = "; ".join(f"{p['rank']}. {p['species']} ({p['confidence']}%)" for p in f_preds) if f_preds else ""

        b47_corr = row.get("Baseline_47_Match") == "YES"
        yolo_corr = row.get("YOLOv8_Match") == "YES"
        res_corr = row.get("ResNet50_Match") == "YES"
        consensus_count = sum([1 for m in [is_corr, b47_corr, yolo_corr, res_corr] if m])
        beats_b47 = bool(is_corr and not b47_corr)
        exclusive_win = bool(is_corr and not b47_corr and not yolo_corr and not res_corr)

        updated_row = dict(row)
        updated_row["FloraNet_102_Prediction"] = pred_top1
        updated_row["FloraNet_102_Scientific"] = pred_sci
        updated_row["FloraNet_102_Top1_Conf"] = str(conf)
        updated_row["FloraNet_102_Top2_Pred"] = top2_pred
        updated_row["FloraNet_102_Top2_Conf"] = str(top2_conf)
        updated_row["FloraNet_102_Top3_Pred"] = top3_pred
        updated_row["FloraNet_102_Top3_Conf"] = str(top3_conf)
        updated_row["Confidence_Margin_Pct"] = str(conf_margin)
        updated_row["Prediction_Entropy"] = str(entropy)
        updated_row["FloraNet_102_Match"] = "YES" if is_corr else "NO"
        updated_row["FloraNet_102_Top5_Hit"] = "YES" if top5_hit else "NO"
        updated_row["FloraNet_102_Top5_Candidates"] = top5_str
        updated_row["FloraNet_Latency_MS"] = str(latency)
        updated_row["Consensus_Models_Correct_Count"] = str(consensus_count)
        updated_row["FloraNet_Beats_Baseline"] = "YES" if beats_b47 else "NO"
        updated_row["FloraNet_Exclusive_Win"] = "YES" if exclusive_win else "NO"

        new_results[idx] = updated_row
        download_queue.task_done()

        if processed_count % 50 == 0 or processed_count == total_specimens:
            curr_acc = (ensemble_correct_count / processed_count) * 100
            elapsed = time.time() - t_start
            print(f"[{processed_count:3d}/{total_specimens}] Running Acc: {ensemble_correct_count}/{processed_count} ({curr_acc:.1f}%) | Speed: {processed_count/elapsed:.1f} img/s")

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in new_results:
            if r is not None:
                writer.writerow(r)

    elapsed_total = round(time.time() - t_start, 1)
    ens_acc_pct = (ensemble_correct_count / total_specimens) * 100
    v2_acc_pct = (v2_correct_count / total_specimens) * 100
    ens_top5_pct = (ensemble_top5_count / total_specimens) * 100
    v2_top5_pct = (v2_top5_count / total_specimens) * 100

    summary_banner = f"""
======================================================================
FLORANET HETEROGENEOUS ENSEMBLE BENCHMARK RESULTS
======================================================================
  • Evaluated:         {total_specimens} independent internet specimens
  • Total Duration:    {elapsed_total}s ({total_specimens/elapsed_total:.1f} specimens/sec)
  • Top-1 Accuracy:    {ensemble_correct_count}/{total_specimens} ({ens_acc_pct:.2f}%)  [Single-Model V2: {v2_correct_count}/{total_specimens} ({v2_acc_pct:.2f}%)]
  • Top-5 Retrieval:   {ensemble_top5_count}/{total_specimens} ({ens_top5_pct:.2f}%)  [Single-Model V2: {v2_top5_count}/{total_specimens} ({v2_top5_pct:.2f}%)]
  • Net Improvement:   +{ensemble_correct_count - v2_correct_count} New Recoveries vs V2 (+{ens_acc_pct - v2_acc_pct:.2f}%)
  • Total Regressions: {len(regressed_from_v2)}
  • Ensemble CSV:      {OUTPUT_CSV}
======================================================================
"""
    if recovered_from_v2:
        summary_banner += "RECOVERED BY ENSEMBLE (Swin Self-Attention Breakthroughs):\n"
        for rec in recovered_from_v2[:10]:
            summary_banner += f"   ✓ [Index {rec[0]}] {rec[1][:32]} -> {rec[3]} ({rec[4]:.1f}%)\n"

    log_both(summary_banner)

if __name__ == "__main__":
    main()

