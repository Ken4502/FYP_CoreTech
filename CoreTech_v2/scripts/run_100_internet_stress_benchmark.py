"""
CoreTech v2: 130-Specimen Live Internet Scraped Stress Benchmark & Evaluation Suite.
Scrapes real-world research-grade observations and web specimens from iNaturalist & Wikimedia across:
  1. Biotic Pests & Infestations (Mealybugs, Aphids, Scale, Spider Mites, Thrips, Whiteflies, Leaf Miners)
  2. Lack of Water / Dehydration (Wilted, Drooping, Limp living plants needing water)
  3. Severe Desiccation / Dried Foliage (Crispy, papery brown withered leaves)
  4. Dead Plants / Beyond Saving (Irreversible tissue mortality, snapped kindling, collapsed tissue)
  5. Other Conditions (Powdery Mildew, Rust Fungi, Sooty Mold, Anthracnose, Chlorosis, Sunburn, Healthy)
"""
import sys
import os
import io
import time
import json
import csv
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

CORETECH_ROOT = Path(__file__).resolve().parent.parent.parent
V2_ROOT = CORETECH_ROOT / "CoreTech_v2"
sys.path.insert(0, str(CORETECH_ROOT))
sys.path.insert(0, str(V2_ROOT))

from CoreTech_v2.src.pipeline_v2 import run_v2_diagnosis

DATA_DIR = V2_ROOT / "data" / "live_internet_stress_benchmark"
REPORTS_DIR = V2_ROOT / "reports"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "CoreTech-v2-Internet-Benchmark/1.0 (academic FYP evaluation)"}

BENCHMARK_SPECS = [
    # --- GROUP A: PESTS & INFESTATIONS (30 images) ---
    {"group": "Pest_Infestation", "expected_class": "Mealybugs_Cochineal", "taxon_id": 49574, "q": "mealybug", "target": 6, "subfolder": "mealybugs"},
    {"group": "Pest_Infestation", "expected_class": "Aphids", "taxon_id": 52381, "q": "black aphid", "target": 6, "subfolder": "aphids"},
    {"group": "Pest_Infestation", "expected_class": "Scale_Insects", "taxon_id": 49456, "q": "scale insect", "target": 5, "subfolder": "scale_insects"},
    {"group": "Pest_Infestation", "expected_class": "Spider_Mites", "taxon_id": 128552, "q": "spider mite", "target": 5, "subfolder": "spider_mites"},
    {"group": "Pest_Infestation", "expected_class": "Thrips_Damage", "taxon_id": 83443, "q": "thrips", "target": 4, "subfolder": "thrips"},
    {"group": "Pest_Infestation", "expected_class": "Whiteflies", "taxon_id": 154784, "q": "whitefly", "target": 4, "subfolder": "whiteflies"},

    # --- GROUP B: LACK OF WATER / UNDERWATERING (25 images) ---
    {"group": "Lack_Of_Water", "expected_class": "Abiotic_Dehydration_Underwatering", "taxon_id": 47126, "q": "wilted plant", "target": 9, "subfolder": "underwatering_wilted"},
    {"group": "Lack_Of_Water", "expected_class": "Abiotic_Dehydration_Underwatering", "taxon_id": 47126, "q": "drooping leaves", "target": 8, "subfolder": "underwatering_drooping"},
    {"group": "Lack_Of_Water", "expected_class": "Abiotic_Dehydration_Underwatering", "taxon_id": 47126, "q": "wilting leaf", "target": 8, "subfolder": "underwatering_wilting"},

    # --- GROUP C: SEVERE DESICCATION / DRIED PLANTS (25 images) ---
    {"group": "Dried_Plant", "expected_class": "Abiotic_Severe_Desiccation_Dried_Plant", "taxon_id": 47126, "q": "dried leaf", "target": 9, "subfolder": "dried_leaf"},
    {"group": "Dried_Plant", "expected_class": "Abiotic_Severe_Desiccation_Dried_Plant", "taxon_id": 47126, "q": "withered plant", "target": 8, "subfolder": "dried_withered"},
    {"group": "Dried_Plant", "expected_class": "Abiotic_Severe_Desiccation_Dried_Plant", "taxon_id": 47126, "q": "desiccated leaves", "target": 8, "subfolder": "dried_desiccated"},

    # --- GROUP D: DEAD PLANTS / BEYOND SAVING (25 images) ---
    {"group": "Dead_Plant", "expected_class": "Abiotic_Terminal_Dead_Plant", "taxon_id": 47126, "q": "dead plant", "target": 9, "subfolder": "dead_plant"},
    {"group": "Dead_Plant", "expected_class": "Abiotic_Terminal_Dead_Plant", "taxon_id": 47126, "q": "dead seedling", "target": 8, "subfolder": "dead_seedling"},
    {"group": "Dead_Plant", "expected_class": "Abiotic_Terminal_Dead_Plant", "taxon_id": 47126, "q": "dead dry branch", "target": 8, "subfolder": "dead_twigs"},

    # --- GROUP E: OTHER PATHOLOGICAL & ABIOTIC CONDITIONS (25 images) ---
    {"group": "Other_Condition", "expected_class": "Powdery_Mildew", "taxon_id": 55525, "q": "", "target": 4, "subfolder": "powdery_mildew"},
    {"group": "Other_Condition", "expected_class": "Rust_Fungi", "taxon_id": 69968, "q": "", "target": 4, "subfolder": "rust_fungi"},
    {"group": "Other_Condition", "expected_class": "Sooty_Mold", "taxon_id": 338190, "q": "", "target": 4, "subfolder": "sooty_mold"},
    {"group": "Other_Condition", "expected_class": "Fungal_Leaf_Spot_Anthracnose", "taxon_id": 327996, "q": "", "target": 4, "subfolder": "leaf_spot"},
    {"group": "Other_Condition", "expected_class": "Abiotic_Nutrient_Chlorosis", "taxon_id": 47126, "q": "chlorosis", "target": 3, "subfolder": "chlorosis"},
    {"group": "Other_Condition", "expected_class": "Abiotic_Sunburn_Scald", "taxon_id": 47126, "q": "sunburn leaf", "target": 3, "subfolder": "sunburn"},
    {"group": "Other_Condition", "expected_class": "Healthy_Foliage", "taxon_id": 69223, "q": "", "target": 3, "subfolder": "healthy_pothos"}, # Epipremnum aureum
]


def fetch_inat_metadata(spec: dict) -> list:
    sub = DATA_DIR / spec["subfolder"]
    sub.mkdir(parents=True, exist_ok=True)
    target = spec["target"]
    q_str = spec.get("q", "")
    tid = spec.get("taxon_id", 47126)

    # First check existing cached images
    cached = list(sub.glob("*.jpg")) + list(sub.glob("*.jpeg"))
    items = []
    for p in cached:
        items.append({
            "local_path": p,
            "group": spec["group"],
            "expected_class": spec["expected_class"],
            "source": "Local Cache",
            "url": "",
        })
        if len(items) >= target:
            return items[:target]

    # Fetch from iNaturalist API
    page = 1
    while len(items) < target and page <= 5:
        endpoint = f"https://api.inaturalist.org/v1/observations?taxon_id={tid}&has[]=photos&per_page=20&page={page}"
        if q_str:
            endpoint += f"&q={urllib.parse.quote(q_str)}"

        try:
            req = urllib.request.Request(endpoint, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if not results:
                break
            for obs in results:
                obs_id = obs.get("id")
                photos = obs.get("photos", [])
                if not photos:
                    continue
                photo_url = photos[0].get("url", "")
                if not photo_url:
                    continue
                # Upgrade to medium/large resolution
                large_url = photo_url.replace("square.", "medium.").replace("small.", "medium.")
                local_file = sub / f"{spec['subfolder']}_{obs_id}.jpg"

                # Download image
                if not local_file.exists():
                    try:
                        img_req = urllib.request.Request(large_url, headers=HEADERS)
                        with urllib.request.urlopen(img_req, timeout=12) as img_resp:
                            img_data = img_resp.read()
                        with open(local_file, "wb") as f:
                            f.write(img_data)
                    except Exception as e:
                        continue

                try:
                    # Validate image opens cleanly
                    with Image.open(local_file) as im:
                        im.verify()
                except Exception:
                    if local_file.exists():
                        local_file.unlink()
                    continue

                items.append({
                    "local_path": local_file,
                    "group": spec["group"],
                    "expected_class": spec["expected_class"],
                    "source": f"iNaturalist #{obs_id}",
                    "url": large_url,
                })
                if len(items) >= target:
                    break
            page += 1
            time.sleep(0.15)
        except Exception as e:
            print(f"[{spec['subfolder']}] Scraping warning: {e}")
            break

    return items[:target]


def run_benchmark():
    print("=" * 80)
    print("  CoreTech v2: 130-Specimen Live Internet Scraped Stress Benchmark")
    print("=" * 80)
    print("[1/3] Harvesting & Verifying 130 Real-World Images from iNaturalist...")
    
    all_specimens = []
    for spec in BENCHMARK_SPECS:
        items = fetch_inat_metadata(spec)
        print(f"  [{spec['group']}] {spec['subfolder']}: {len(items)} / {spec['target']} specimens acquired.")
        all_specimens.extend(items)

    total_count = len(all_specimens)
    print(f"\n[2/3] Total Specimens Ready for Inference: {total_count}")
    print("-" * 80)

    results = []
    group_stats = {
        "Pest_Infestation": {"total": 0, "correct_category": 0, "correct_class": 0},
        "Lack_Of_Water": {"total": 0, "correct_category": 0, "correct_class": 0},
        "Dried_Plant": {"total": 0, "correct_category": 0, "correct_class": 0},
        "Dead_Plant": {"total": 0, "correct_category": 0, "correct_class": 0},
        "Other_Condition": {"total": 0, "correct_category": 0, "correct_class": 0},
    }

    t0_all = time.time()
    for idx, item in enumerate(all_specimens, 1):
        lp = item["local_path"]
        grp = item["group"]
        exp_cls = item["expected_class"]
        
        try:
            im = Image.open(lp).convert("RGB")
            t_start = time.time()
            diag = run_v2_diagnosis(im)
            elapsed_ms = round((time.time() - t_start) * 1000.0, 1)

            pred_cls = diag.get("diagnosis", {}).get("class_id") or "Unknown"
            pred_cond = diag.get("diagnosis", {}).get("condition_name") or "Unknown"
            super_state = diag.get("diagnosis", {}).get("super_state") or "Unknown"
            conf = diag.get("diagnosis", {}).get("confidence", 0.0)
            species = diag.get("species", {}).get("name") or "Unknown"
            ver_meta = diag.get("verification_meta", {})

            # Evaluation Logic:
            # 1. Exact class match
            exact_match = bool(pred_cls == exp_cls)

            # 2. Category match (Clinical Group Precision)
            cat_match = False
            if grp == "Pest_Infestation":
                pest_classes = ("Mealybugs_Cochineal", "Aphids", "Scale_Insects", "Thrips_Damage", "Spider_Mites", "Whiteflies", "Leaf_Miners")
                cat_match = bool(pred_cls in pest_classes or super_state == "Biotic_Pathology" or ver_meta.get("is_infestation", False))
            elif grp == "Lack_Of_Water":
                cat_match = bool(pred_cls == "Abiotic_Dehydration_Underwatering" or "water" in pred_cond.lower() or "dehydration" in pred_cond.lower() or ver_meta.get("is_underwatered", False))
            elif grp == "Dried_Plant":
                cat_match = bool(pred_cls in ("Abiotic_Severe_Desiccation_Dried_Plant", "Abiotic_Terminal_Dead_Plant") or "dried" in pred_cond.lower() or "desiccation" in pred_cond.lower() or ver_meta.get("is_dried", False))
            elif grp == "Dead_Plant":
                cat_match = bool(pred_cls in ("Abiotic_Terminal_Dead_Plant", "Abiotic_Severe_Desiccation_Dried_Plant") or "dead" in pred_cond.lower() or ver_meta.get("is_dead_beyond_saving", False))
            elif grp == "Other_Condition":
                cat_match = bool(exact_match or (super_state == "Biotic_Pathology" and exp_cls in ("Powdery_Mildew", "Rust_Fungi", "Sooty_Mold", "Fungal_Leaf_Spot_Anthracnose")))

            group_stats[grp]["total"] += 1
            if cat_match:
                group_stats[grp]["correct_category"] += 1
            if exact_match:
                group_stats[grp]["correct_class"] += 1

            status_symbol = "OK" if exact_match else ("~" if cat_match else "X")
            print(f"[{idx:3d}/{total_count:3d}] [{status_symbol}] {grp[:16]:16s} | Exp: {exp_cls[:20]:20s} | Pred: {pred_cls[:20]:20s} | Conf: {conf:5.1f}% | {elapsed_ms}ms")

            results.append({
                "index": idx,
                "file": lp.name,
                "group": grp,
                "expected_class": exp_cls,
                "predicted_class": pred_cls,
                "predicted_condition": pred_cond,
                "super_state": super_state,
                "confidence": conf,
                "exact_match": exact_match,
                "category_match": cat_match,
                "latency_ms": elapsed_ms,
                "species": species,
                "source": item["source"],
            })
        except Exception as e:
            print(f"[{idx:3d}/{total_count:3d}] [ERR] Error on {lp.name}: {e}")
            results.append({
                "index": idx,
                "file": lp.name,
                "group": grp,
                "expected_class": exp_cls,
                "predicted_class": "ERROR",
                "predicted_condition": str(e),
                "super_state": "ERROR",
                "confidence": 0.0,
                "exact_match": False,
                "category_match": False,
                "latency_ms": 0.0,
                "species": "ERROR",
                "source": item["source"],
            })

    total_time = round(time.time() - t0_all, 1)
    
    # Compute overall metrics
    total_samples = len(results)
    total_exact = sum(1 for r in results if r["exact_match"])
    total_cat = sum(1 for r in results if r["category_match"])
    overall_exact_acc = round((total_exact / max(1, total_samples)) * 100.0, 1)
    overall_cat_acc = round((total_cat / max(1, total_samples)) * 100.0, 1)

    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY & PERFORMANCE BREAKDOWN")
    print("=" * 80)
    print(f"Total Evaluated Specimens : {total_samples}")
    print(f"Overall Category Accuracy : {overall_cat_acc}% ({total_cat}/{total_samples})")
    print(f"Overall Exact Class Acc   : {overall_exact_acc}% ({total_exact}/{total_samples})")
    print(f"Total Benchmark Runtime   : {total_time}s (avg {total_time/max(1, total_samples):.2f}s/image)")
    print("-" * 80)
    print(f"{'Group':25s} | {'Total':5s} | {'Category Acc':12s} | {'Exact Acc':10s}")
    print("-" * 80)
    for g, st in group_stats.items():
        n = st["total"]
        c_acc = round((st["correct_category"] / max(1, n)) * 100.0, 1) if n > 0 else 0.0
        e_acc = round((st["correct_class"] / max(1, n)) * 100.0, 1) if n > 0 else 0.0
        print(f"{g:25s} | {n:5d} | {c_acc:11.1f}% | {e_acc:9.1f}%")
    print("=" * 80)

    # Save CSV
    csv_path = REPORTS_DIR / "live_100_internet_stress_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "file", "group", "expected_class", "predicted_class",
            "predicted_condition", "super_state", "confidence", "exact_match",
            "category_match", "latency_ms", "species", "source"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"[+] Saved detailed CSV results to: {csv_path}")

    # Save JSON summary
    json_path = REPORTS_DIR / "live_100_internet_stress_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": total_samples,
            "overall_category_accuracy_pct": overall_cat_acc,
            "overall_exact_accuracy_pct": overall_exact_acc,
            "total_runtime_seconds": total_time,
            "group_breakdown": group_stats,
            "results": results
        }, f, indent=2)
    print(f"[+] Saved detailed JSON summary to: {json_path}")

    return overall_cat_acc, overall_exact_acc, group_stats, results

if __name__ == "__main__":
    run_benchmark()
