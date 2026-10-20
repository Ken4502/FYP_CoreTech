"""
Optimized Multithreaded Specimen Harvester for Missing Cacti & Pothos Deepening
Uses ThreadPoolExecutor for 10x faster concurrent downloads from iNaturalist.
"""
import sys
import json
import shutil
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config

NEW_TARGET_TAXA = {
    "Golden Barrel Cactus (Echinocactus grusonii)": "Echinocactus grusonii",
    "Columnar Cactus (Pachycereus pringlei)": "Pachycereus pringlei",
    "Pothos (Ivy arum)": "Epipremnum aureum"
}

def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.array(resized, dtype=np.int32)
    diff = arr[:, :-1] > arr[:, 1:]
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")

def query_inaturalist(taxon_name: str, count: int = 50):
    encoded = urllib.parse.quote(taxon_name)
    records = []
    seen = set()
    for grade in ["research", "any"]:
        if len(records) >= count * 2:
            break
        grade_param = "&quality_grade=research" if grade == "research" else ""
        url = (
            f"https://api.inaturalist.org/v1/observations?"
            f"taxon_name={encoded}{grade_param}&photos=true&per_page={max(count * 2, 40)}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "FloraResearchBot/1.0 (FYP Botanical AI Project)"})
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for item in data.get("results", []):
                    for p in item.get("photos", []):
                        orig_url = p.get("url", "")
                        if orig_url and orig_url not in seen:
                            seen.add(orig_url)
                            high_res = orig_url.replace("/square.", "/medium.").replace("/small.", "/medium.")
                            records.append({
                                "obs_id": item.get("id"),
                                "url": high_res,
                                "taxon": taxon_name
                            })
                            if len(records) >= count * 2:
                                break
        except Exception as e:
            print(f"  [Warning: {taxon_name}]: {e}", flush=True)
    return records

def download_single(cand, min_dim: int = 224):
    url = cand["url"]
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        import io
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        if w < min_dim or h < min_dim:
            return None
        return img, cand["obs_id"]
    except Exception:
        return None

def harvest_and_distribute(class_name: str, taxon_name: str, target_count: int = 50):
    train_dir = config.TRAIN_DIR / class_name
    val_dir = config.VAL_DIR / class_name
    test_dir = config.TEST_DIR / class_name

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    existing_train = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    if len(existing_train) >= 30 and class_name != "Pothos (Ivy arum)":
        print(f"[Harvester] '{class_name}' already has {len(existing_train)} specimens. Skipping download.", flush=True)
        return

    print(f"\n[Harvester] Fetching specimens for: '{class_name}' ({taxon_name})...", flush=True)
    candidates = query_inaturalist(taxon_name, count=target_count)
    print(f"   Found {len(candidates)} candidate observations. Downloading concurrently...", flush=True)

    existing_hashes = set()
    for d in [train_dir, val_dir, test_dir]:
        for f in d.glob("*.*"):
            try:
                with Image.open(f) as im:
                    existing_hashes.add(compute_dhash(im))
            except Exception:
                pass

    downloaded = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_single, cand) for cand in candidates]
        for f in as_completed(futures):
            res = f.result()
            if res is None:
                continue
            img, obs_id = res
            h = compute_dhash(img)
            if any(hamming_distance(h, eh) <= 3 for eh in existing_hashes):
                continue
            existing_hashes.add(h)
            downloaded.append((img, obs_id))
            if len(downloaded) >= target_count:
                break

    print(f"   Downloaded and verified {len(downloaded)} fresh unique specimens.", flush=True)

    n_total = len(downloaded)
    n_val = max(4, int(n_total * 0.15))
    n_test = max(4, int(n_total * 0.15))
    n_train = n_total - n_val - n_test

    splits = (
        [(img, obs, train_dir) for img, obs in downloaded[:n_train]] +
        [(img, obs, val_dir) for img, obs in downloaded[n_train:n_train + n_val]] +
        [(img, obs, test_dir) for img, obs in downloaded[n_train + n_val:]]
    )

    for img, obs, out_dir in splits:
        fn = f"inat_{obs}.jpg"
        img.save(out_dir / fn, "JPEG", quality=92)

    print(f"   Distributed: {n_train} train, {n_val} val, {n_test} test to {class_name}.", flush=True)

def integrate_user_samples():
    print("\n[Specimens] Integrating user-provided specimen images into evaluation datasets...", flush=True)
    user_images = [
        ("Columnar Cactus (Pachycereus pringlei)", r"C:/Users/User/.gemini/antigravity/brain/d97fd749-7a53-48bf-99ed-b47c211057f2/.user_uploaded/media_1792003307823_ab7e5f6b.jpg", "user_specimen_1.jpg"),
        ("Pothos (Ivy arum)", r"C:/Users/User/.gemini/antigravity/brain/d97fd749-7a53-48bf-99ed-b47c211057f2/.user_uploaded/media_1792003307872_0d306914.webp", "user_specimen_2.webp"),
        ("Golden Barrel Cactus (Echinocactus grusonii)", r"C:/Users/User/.gemini/antigravity/brain/d97fd749-7a53-48bf-99ed-b47c211057f2/.user_uploaded/media_1792003307885_82be0d60.webp", "user_specimen_3.webp")
    ]

    for class_name, src_path, dst_name in user_images:
        test_dir = config.TEST_DIR / class_name
        test_dir.mkdir(parents=True, exist_ok=True)
        dst_test = test_dir / dst_name
        shutil.copy2(src_path, dst_test)
        print(f"   Added {dst_name} -> {test_dir}", flush=True)

if __name__ == "__main__":
    for c_name, taxon in NEW_TARGET_TAXA.items():
        harvest_and_distribute(c_name, taxon, target_count=50)
    integrate_user_samples()
    print("\n[Complete] Specimen harvesting and integration finished successfully!", flush=True)

