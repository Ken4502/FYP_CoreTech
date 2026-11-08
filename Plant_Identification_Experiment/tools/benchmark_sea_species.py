"""
Live Internet Benchmark for FloraNet-261 Southeast Asian Species Expansion.
Fetches unseen test specimens from live botanical APIs (iNaturalist / Wikimedia Commons)
and evaluates Top-1 and Top-3 prediction accuracy on the newly fine-tuned FloraNet-261 model.
"""

import sys
import os
import io
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
import csv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as tv_models

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = EXPERIMENT_ROOT / "artifacts" / "best_model.pth"
CLASS_INDICES_PATH = EXPERIMENT_ROOT / "artifacts" / "class_indices.json"
REPORT_CSV_PATH = EXPERIMENT_ROOT / "reports" / "floranet_sea_10_internet_benchmark.csv"

SEA_TARGETS = [
    {"name": "Curry Leaf Tree (Murraya koenigii)", "taxon": "Murraya koenigii", "common": "Curry Leaf / Daun Kari"},
    {"name": "Kaffir Lime (Citrus hystrix)", "taxon": "Citrus hystrix", "common": "Kaffir Lime / Limau Purut"},
    {"name": "Calamansi (Citrus microcarpa)", "taxon": "Citrus microcarpa", "common": "Calamansi / Limau Kasturi"},
    {"name": "Butterfly Pea (Clitoria ternatea)", "taxon": "Clitoria ternatea", "common": "Butterfly Pea / Bunga Telang"},
    {"name": "Vietnamese Coriander (Persicaria odorata)", "taxon": "Persicaria odorata", "common": "Vietnamese Coriander / Daun Kesum"},
    {"name": "Turmeric (Curcuma longa)", "taxon": "Curcuma longa", "common": "Turmeric / Kunyit"},
    {"name": "Torch Ginger (Etlingera elatior)", "taxon": "Etlingera elatior", "common": "Torch Ginger / Bunga Kantan"},
    {"name": "Betel Leaf (Piper betle)", "taxon": "Piper betle", "common": "Betel Leaf / Daun Sireh"},
    {"name": "Water Spinach (Ipomoea aquatica)", "taxon": "Ipomoea aquatica", "common": "Water Spinach / Kangkung"},
    {"name": "Bitter Gourd (Momordica charantia)", "taxon": "Momordica charantia", "common": "Bitter Gourd / Peria Katak"}
]

def load_model(device):
    print(f"[*] Loading model from {MODEL_PATH}...")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    classes = ckpt.get("classes", [])
    print(f"[*] Total classes in checkpoint: {len(classes)}")

    model = tv_models.convnext_small(weights=None)
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(768, len(classes))
    )

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, classes

def fetch_live_photos_for_taxon(taxon: str, count: int = 10):
    url = f"https://api.inaturalist.org/v1/observations?taxon_name={urllib.parse.quote(taxon)}&photos=true&per_page=30&order=desc&order_by=created_at&fields=(id:!t,photos:(url:!t))"
    headers = {"User-Agent": "FloraNetSEABenchmark/1.0 (academic research)"}
    photos = []
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for obs in data.get("results", []):
            obs_id = obs.get("id")
            for p in obs.get("photos", []):
                u = p.get("url", "")
                if u:
                    hi_res = u.replace("square.", "medium.").replace("small.", "medium.")
                    photos.append((hi_res, obs_id))
                    if len(photos) >= count:
                        return photos
    except Exception as e:
        print(f"  [!] Live fetch error for {taxon}: {e}")
    return photos

def main():
    print("=" * 80)
    print("FLORANET-261: LIVE INTERNET BENCHMARK ON NEW SOUTHEAST ASIAN SPECIES")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    model, classes = load_model(device)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = []
    top1_correct = 0
    top3_correct = 0
    total_evaluated = 0

    print("\n[*] Starting live internet evaluations across all 10 SEA species (10 live specimens each)...")

    for target in SEA_TARGETS:
        target_name = target["name"]
        taxon = target["taxon"]
        print(f"\n---> Evaluating '{target_name}' ({target['common']})...")
        
        photos = fetch_live_photos_for_taxon(taxon, count=10)
        if not photos:
            print(f"  [!] No live photos fetched for {taxon}")
            continue

        species_top1 = 0
        species_top3 = 0
        species_count = 0

        for url, obs_id in photos:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    img_bytes = resp.read()
                im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                
                tensor = eval_transform(im).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = F.softmax(logits, dim=-1).squeeze(0)
                
                top3_prob, top3_idx = torch.topk(probs, 3)
                top1_idx_val = top3_idx[0].item()
                top1_pred = classes[top1_idx_val]
                top1_conf = top3_prob[0].item() * 100.0

                top3_preds = [classes[idx.item()] for idx in top3_idx]
                top3_confs = [top3_prob[i].item() * 100.0 for i in range(3)]

                is_top1 = (top1_pred == target_name)
                is_top3 = (target_name in top3_preds)

                if is_top1:
                    species_top1 += 1
                    top1_correct += 1
                if is_top3:
                    species_top3 += 1
                    top3_correct += 1
                
                species_count += 1
                total_evaluated += 1

                results.append({
                    "species": target_name,
                    "taxon": taxon,
                    "obs_id": obs_id,
                    "url": url,
                    "top1_pred": top1_pred,
                    "top1_conf": f"{top1_conf:.1f}%",
                    "top1_match": "YES" if is_top1 else "NO",
                    "top3_preds": "; ".join([f"{p} ({c:.1f}%)" for p, c in zip(top3_preds, top3_confs)]),
                    "top3_match": "YES" if is_top3 else "NO",
                })
                print(f"  [{obs_id}] Pred: {top1_pred} ({top1_conf:.1f}%) | Top-1: {'✓' if is_top1 else '✗'} | Top-3: {'✓' if is_top3 else '✗'}")
            except Exception as e:
                print(f"  [!] Specimen error: {e}")

        acc1 = (species_top1 / species_count * 100.0) if species_count > 0 else 0.0
        acc3 = (species_top3 / species_count * 100.0) if species_count > 0 else 0.0
        print(f"  >> '{target_name}' Result: Top-1 = {acc1:.1f}% ({species_top1}/{species_count}) | Top-3 = {acc3:.1f}% ({species_top3}/{species_count})")

    overall_top1_acc = (top1_correct / total_evaluated * 100.0) if total_evaluated > 0 else 0.0
    overall_top3_acc = (top3_correct / total_evaluated * 100.0) if total_evaluated > 0 else 0.0

    print("\n" + "=" * 80)
    print("LIVE INTERNET BENCHMARK SUMMARY:")
    print(f"Total Live Specimens Tested: {total_evaluated}")
    print(f"Top-1 Accuracy: {overall_top1_acc:.2f}% ({top1_correct}/{total_evaluated})")
    print(f"Top-3 Accuracy: {overall_top3_acc:.2f}% ({top3_correct}/{total_evaluated})")
    print("=" * 80)

    # Save to CSV
    REPORT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "species", "taxon", "obs_id", "url", "top1_pred", "top1_conf", "top1_match", "top3_preds", "top3_match"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"[✓] Benchmark report saved to: {REPORT_CSV_PATH}")

if __name__ == "__main__":
    main()

