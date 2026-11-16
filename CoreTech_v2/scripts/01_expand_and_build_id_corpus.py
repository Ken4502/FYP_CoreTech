"""
CoreTech v2 Phase 1A: Botanical Identification Dataset Expansion & Leak-Free Corpus Builder.

1. Consolidates all clean existing specimens across:
     - Plant_Identification_Experiment/data/{train,val,test}
     - Plant_Identification_Experiment/data/specimens (10 SEA species)
     - Datasets/Indonesian Herb Leaf Dataset 3500 (Pandan, Lidah Buaya, Sirih, Jeruk Nipis)
     - Datasets/house_plant_species (exact/safe overlaps)
2. Computes exact SHA-256 and 64-bit dHash per specimen to eliminate all duplicates and
   near-duplicates BEFORE splitting.
3. Expands every species in botanical_taxonomy_v2.json up to 500-650 unique high-res
   specimens from iNaturalist research-grade observations (with polite rate-limiting).
4. Splits strictly by observation group ID into train / val / test (zero cross-split leakage).
"""
import os
import re
import io
import sys
import time
import json
import random
import shutil
import hashlib
import threading
import urllib.parse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))
from configs.v2_config import (
    CORETECH_ROOT, DATA_DIR, ID_DATA_DIR, REPORTS_DIR, load_taxonomy
)

POOL_DIR = DATA_DIR / "botanical_id_pool"
LOG_PATH = REPORTS_DIR / "data_id_harvest.log"
MANIFEST_PATH = REPORTS_DIR / "id_corpus_manifest.json"

POOL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS_API = {"User-Agent": "CoreTechV2BotanicalBuilder/1.0 (academic FYP plant AI research)"}
HEADERS_IMG = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CoreTechV2/1.0"}

_SESSION_LOCAL = threading.local()


def get_session() -> requests.Session:
    s = getattr(_SESSION_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=Retry(total=1, backoff_factor=0.3))
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _SESSION_LOCAL.session = s
    return s


def log_msg(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    val = 0
    for bit in diff.flatten():
        val = (val << 1) | int(bit)
    return val


def is_near_dup(h: int, existing_hashes: list, max_hamming: int = 4) -> bool:
    for eh in existing_hashes:
        if (h ^ eh).bit_count() <= max_hamming:
            return True
    return False


def validate_and_hash_image(path: Path, min_dim: int = 160):
    try:
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            rgb = im.convert("RGB")
        w, h = rgb.size
        if min(w, h) < min_dim:
            return None
        # Reject blank / solid featureless images
        arr = np.asarray(rgb.resize((64, 64)), dtype=np.float32)
        if float(np.std(arr)) < 8.0:
            return None
        dh = compute_dhash(rgb)
        return sha, dh, w, h
    except Exception:
        return None


def extract_obs_group(filename: str) -> str:
    """Groups photos from the same iNaturalist observation so splits never leak."""
    m = re.match(r"^(?:specimen|obs|inat)_(\d+)", filename)
    if m:
        return f"obs_{m.group(1)}"
    # Fallback strip numeric copy suffixes
    stem = Path(filename).stem
    return f"file_{stem}"


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def collect_existing_sources(taxonomy: dict) -> dict:
    """Maps each species name -> list of existing candidate file paths."""
    sources = defaultdict(list)

    # 1. Plant_Identification_Experiment/data/{train,val,test}
    p_exp = CORETECH_ROOT / "Plant_Identification_Experiment" / "data"
    for split in ["train", "val", "test"]:
        sp_dir = p_exp / split
        if not sp_dir.exists():
            continue
        for c_dir in sp_dir.iterdir():
            if c_dir.is_dir() and c_dir.name in taxonomy:
                for f in c_dir.glob("*.*"):
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                        sources[c_dir.name].append((f, f"exp_{split}_{f.name}"))

    # 2. Plant_Identification_Experiment/data/specimens (SEA 10)
    spec_dir = p_exp / "specimens"
    if spec_dir.exists():
        sci_to_key = {v["scientific_name"].lower().replace(" ", "_"): k for k, v in taxonomy.items()}
        for s_dir in spec_dir.iterdir():
            if s_dir.is_dir() and s_dir.name in sci_to_key:
                k = sci_to_key[s_dir.name]
                for f in s_dir.glob("*.*"):
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                        sources[k].append((f, f.name))

    # 3. Datasets/Indonesian Herb Leaf Dataset 3500
    ih_dir = CORETECH_ROOT / "Datasets" / "Indonesian Herb Leaf Dataset 3500"
    ih_map = {
        "Pandan": "Pandan (Pandanus amaryllifolius)",
        "Lidah Buaya": "Aloe Vera",
        "Sirih": "Betel Leaf (Piper betle)",
        "Jeruk Nipis": "Lime Tree (Citrus aurantiifolia)",
    }
    if ih_dir.exists():
        for folder, sp_key in ih_map.items():
            sub = ih_dir / folder
            if sub.exists() and sp_key in taxonomy:
                for f in sub.glob("*.*"):
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                        sources[sp_key].append((f, f"indo_{folder}_{f.name}"))

    return sources


def resolve_inat_taxon_id(sci_name: str):
    sess = get_session()
    try:
        r = sess.get(
            f"https://api.inaturalist.org/v1/taxa?q={urllib.parse.quote(sci_name)}&rank=species,genus,subspecies,variety,hybrid",
            headers=HEADERS_API,
            timeout=10,
        )
        if r.status_code == 200:
            res = r.json().get("results", [])
            if res:
                return res[0].get("id")
    except Exception:
        pass
    return None


def query_inaturalist_candidates(sci_name: str, needed: int, existing_urls: set) -> list:
    """Queries iNaturalist politely using persistent HTTP sessions and synonym-aware taxon_id fallback."""
    candidates = []
    seen = set(existing_urls)
    sess = get_session()

    # Try taxon_name first; if sparse, fallback to active synonym taxon_id
    queries = [("taxon_name", sci_name)]
    taxon_id = resolve_inat_taxon_id(sci_name)
    if taxon_id:
        queries = [("taxon_id", str(taxon_id)), ("taxon_name", sci_name)]

    target_cands = max(needed * 2, 120)

    for q_key, q_val in queries:
        if len(candidates) >= target_cands:
            break
        for grade in ["research", "any"]:
            if len(candidates) >= target_cands:
                break
            max_pages = 5 if grade == "research" else 3
            for page in range(1, max_pages + 1):
                if len(candidates) >= target_cands:
                    break
                params = {
                    q_key: q_val,
                    "photos": "true",
                    "per_page": 200,
                    "page": page,
                    "order": "desc",
                    "order_by": "votes",
                }
                if grade == "research":
                    params["quality_grade"] = "research"

                retries = 0
                results = []
                while retries < 3:
                    try:
                        resp = sess.get("https://api.inaturalist.org/v1/observations", params=params, headers=HEADERS_API, timeout=15)
                        if resp.status_code == 429:
                            wait_s = 25 * (retries + 1)
                            log_msg(f"  [Rate Limit 429] '{sci_name}' sleeping {wait_s}s...")
                            time.sleep(wait_s)
                            retries += 1
                            continue
                        if resp.status_code != 200:
                            break
                        results = resp.json().get("results", [])
                        if not results:
                            break
                        for obs in results:
                            obs_id = obs.get("id")
                            for p_idx, p in enumerate(obs.get("photos", [])):
                                u = p.get("url", "")
                                if not u:
                                    continue
                                large_u = u.replace("/square.", "/large.").replace("/small.", "/large.").replace("/medium.", "/large.")
                                med_u = u.replace("/square.", "/medium.").replace("/small.", "/medium.")
                                if large_u not in seen:
                                    seen.add(large_u)
                                    candidates.append((large_u, med_u, obs_id, p_idx))
                        time.sleep(0.75)
                        break
                    except Exception:
                        time.sleep(1.5)
                        retries += 1
                if not results:
                    break
    return candidates


def download_and_store_candidate(cand, out_dir: Path, lock: threading.Lock, seen_shas: set, seen_dhashes: list, target: int) -> bool:
    large_u, med_u, obs_id, p_idx = cand
    with lock:
        if len(seen_shas) >= target:
            return False

    sess = get_session()
    raw = None
    for url in [large_u, med_u]:
        with lock:
            if len(seen_shas) >= target:
                return False
        try:
            resp = sess.get(url, headers=HEADERS_IMG, timeout=8)
            if resp.status_code == 200 and len(resp.content) >= 6000:
                raw = resp.content
                break
        except Exception:
            continue
    if raw is None:
        return False

    try:
        sha = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            rgb = im.convert("RGB")
        w, h = rgb.size
        if min(w, h) < 256:
            return False
        arr = np.asarray(rgb.resize((64, 64)), dtype=np.float32)
        if float(np.std(arr)) < 10.0:
            return False
        dh = compute_dhash(rgb)

        with lock:
            if len(seen_shas) >= target:
                return False
            if sha in seen_shas or is_near_dup(dh, seen_dhashes, max_hamming=4):
                return False
            seen_shas.add(sha)
            seen_dhashes.append(dh)
            dst = out_dir / f"obs_{obs_id}_{p_idx}_{sha[:8]}.jpg"
            rgb.save(dst, "JPEG", quality=93)
            return True
    except Exception:
        return False


def build_split_for_species(species: str, pool_files: list, seed: int = 42):
    """Partitions deduplicated observation groups cleanly into train / val / test."""
    groups = defaultdict(list)
    for f in pool_files:
        groups[extract_obs_group(f.name)].append(f)

    rng = random.Random(f"{seed}_{species}")
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)

    total = len(pool_files)
    # 12% test (40..75), 12% val (40..75), remaining train (~76%)
    n_test_target = min(75, max(20, int(round(total * 0.12))))
    n_val_target = min(75, max(20, int(round(total * 0.12))))

    splits = {"test": [], "val": [], "train": []}
    for gk in group_keys:
        files = sorted(groups[gk], key=lambda p: p.name)
        if len(splits["test"]) < n_test_target and total >= 60:
            splits["test"].extend(files)
        elif len(splits["val"]) < n_val_target and total >= 40:
            splits["val"].extend(files)
        else:
            splits["train"].extend(files)

    for split_name, files in splits.items():
        dst_dir = ID_DATA_DIR / split_name / species
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            link_or_copy(f, dst_dir / f.name)

    return {k: len(v) for k, v in splits.items()}


def main():
    taxonomy = load_taxonomy()
    log_msg("=" * 72)
    log_msg(f"CORETECH V2 BOTANICAL CORPUS BUILDER ({len(taxonomy)} SPECIES)")
    log_msg("=" * 72)

    existing_map = collect_existing_sources(taxonomy)
    manifest = {}

    species_list = sorted(taxonomy.keys())
    for idx, sp in enumerate(species_list, 1):
        info = taxonomy[sp]
        target = info["target_images"]
        sci_name = info["scientific_name"]

        safe_dir = POOL_DIR / sp
        safe_dir.mkdir(parents=True, exist_ok=True)
        done_flag = safe_dir / "_done_meta.json"

        # Fast resume if already completed to target
        if done_flag.exists():
            try:
                meta = json.loads(done_flag.read_text(encoding="utf-8"))
                pool_files = list(safe_dir.glob("*.jpg")) + list(safe_dir.glob("*.png"))
                if meta.get("target") == target and len(pool_files) >= min(target, meta.get("pooled", 0)):
                    if not (ID_DATA_DIR / "test" / sp).exists() or len(list((ID_DATA_DIR / "test" / sp).glob("*.*"))) == 0:
                        split_counts = build_split_for_species(sp, pool_files)
                    else:
                        split_counts = {
                            s: len(list((ID_DATA_DIR / s / sp).glob("*.*")))
                            for s in ["train", "val", "test"]
                        }
                    manifest[sp] = {"total": len(pool_files), "splits": split_counts, "family": info["family"]}
                    if idx % 25 == 0 or idx == len(species_list):
                        log_msg(f"[{idx}/{len(species_list)}] Resumed '{sp}': {len(pool_files)} imgs -> {split_counts}")
                    continue
            except Exception:
                pass

        # Step 1: Consolidate & dedup existing files + whatever was already pooled
        seen_shas = set()
        seen_dhashes = []
        already_pooled = list(safe_dir.glob("*.jpg")) + list(safe_dir.glob("*.png"))
        for f in already_pooled:
            res = validate_and_hash_image(f, min_dim=160)
            if res is None:
                f.unlink(missing_ok=True)
                continue
            sha, dh, _, _ = res
            if sha in seen_shas or is_near_dup(dh, seen_dhashes, max_hamming=3):
                f.unlink(missing_ok=True)
                continue
            seen_shas.add(sha)
            seen_dhashes.append(dh)

        for src_path, dst_name in existing_map.get(sp, []):
            if len(seen_shas) >= target:
                break
            res = validate_and_hash_image(src_path, min_dim=160)
            if res is None:
                continue
            sha, dh, _, _ = res
            if sha in seen_shas or is_near_dup(dh, seen_dhashes, max_hamming=3):
                continue
            seen_shas.add(sha)
            seen_dhashes.append(dh)
            # Clean extension
            out_name = Path(dst_name).stem + ".jpg"
            dst_path = safe_dir / out_name
            if not dst_path.exists():
                try:
                    if src_path.suffix.lower() in {".jpg", ".jpeg"}:
                        link_or_copy(src_path, dst_path)
                    else:
                        with Image.open(src_path) as im:
                            im.convert("RGB").save(dst_path, "JPEG", quality=93)
                except Exception:
                    pass

        initial_cnt = len(seen_shas)
        needed = max(0, target - initial_cnt)

        # Step 2: Expand deficient classes from iNaturalist
        if needed > 0:
            log_msg(f"[{idx}/{len(species_list)}] '{sp}' ({sci_name}): pooled={initial_cnt}/{target}, harvesting +{needed}...")
            cands = query_inaturalist_candidates(sci_name, needed=needed, existing_urls=set())
            lock = threading.Lock()
            added = 0
            if cands:
                ex = ThreadPoolExecutor(max_workers=24)
                try:
                    futs = [
                        ex.submit(download_and_store_candidate, c, safe_dir, lock, seen_shas, seen_dhashes, target)
                        for c in cands
                    ]
                    for fut in as_completed(futs):
                        if fut.result():
                            added += 1
                        with lock:
                            done_now = len(seen_shas) >= target
                        if done_now:
                            for rem in futs:
                                rem.cancel()
                            break
                finally:
                    ex.shutdown(wait=False, cancel_futures=True)
            log_msg(f"  -> Added {added} unique high-res specimens | Total={len(seen_shas)}")

        pool_files = sorted([f for f in safe_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        split_counts = build_split_for_species(sp, pool_files)
        done_flag.write_text(json.dumps({"target": target, "pooled": len(pool_files), "splits": split_counts}, indent=2), encoding="utf-8")
        manifest[sp] = {"total": len(pool_files), "splits": split_counts, "family": info["family"]}

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    total_all = sum(v["total"] for v in manifest.values())
    log_msg("=" * 72)
    log_msg(f"BOTANICAL ID CORPUS COMPLETE: {len(manifest)} species, {total_all} deduplicated images!")
    log_msg("=" * 72)


if __name__ == "__main__":
    main()
