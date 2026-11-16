"""
CoreTech v2 Phase 1B: Surface-True Plant Pathology, Health & Abiotic Stress Corpus Builder.

Assembles 19 unified surface classes across:
  1. Clean legacy PhytoCare data (stripping saved synthetic aug_* copies and cross-split duplicates)
  2. PlantWild v2 (EGOISTyrh/mvpdr-plant-disease - 11,488 cleaned in-the-wild disease images)
  3. PlantDoc Extended (MuzammilMax/Plant_Doc_Extended - wild field backgrounds)
  4. FGVC8 Plant Pathology 2021 (timm/plant-pathology-2021 - high-res surface foliage photos)
  5. PlantVillage balanced supplement (BrandonFors/Plant-Diseases-PlantVillage-Dataset, tagged pv_ for dynamic bg blending)
  6. PDSNet field disease images (Project-AgML/pdsnet_field_plant_disease_segmentation)
  7. Variegated/Spiny Healthy Hard Negatives from CoreTech v2 Botanical Pool
  8. Automated YOLOE Plant-Surface Hygiene Gate to purge non-plant/bug-on-concrete noise
"""
import io
import os
import sys
import json
import time
import shutil
import random
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl
from PIL import Image, ImageFile
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ImageFile.LOAD_TRUNCATED_IMAGES = True

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))
from configs.v2_config import (
    CORETECH_ROOT, DATA_DIR, PATH_DATA_DIR, REPORTS_DIR, SURFACE_PATHOLOGY_CLASSES
)

POOL_DIR = DATA_DIR / "pathology_pool"
HF_CACHE_STAGE = DATA_DIR / "_hf_staging"
LOG_PATH = REPORTS_DIR / "data_pathology_build.log"
MANIFEST_PATH = REPORTS_DIR / "pathology_corpus_manifest.json"

for d in [POOL_DIR, HF_CACHE_STAGE, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


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


def is_valid_surface_image(raw_or_img, min_dim: int = 160):
    try:
        if isinstance(raw_or_img, (bytes, bytearray)):
            sha = hashlib.sha256(raw_or_img).hexdigest()
            with Image.open(io.BytesIO(raw_or_img)) as im:
                im.load()
                rgb = im.convert("RGB")
        elif isinstance(raw_or_img, Path):
            raw = raw_or_img.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                rgb = im.convert("RGB")
        else:
            rgb = raw_or_img.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=92)
            sha = hashlib.sha256(buf.getvalue()).hexdigest()

        w, h = rgb.size
        if min(w, h) < min_dim:
            return None
        # Resize excessively huge >1280px images cleanly to max 1024px to save disk & speed IO
        if max(w, h) > 1024:
            scale = 1024.0 / max(w, h)
            rgb = rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        arr = np.asarray(rgb.resize((64, 64)), dtype=np.float32)
        if float(np.std(arr)) < 8.0:
            return None
        # Reject grayscale / historical scanned book pages
        hsv = rgb.resize((64, 64)).convert("HSV")
        sat = np.asarray(hsv, dtype=np.float32)[:, :, 1]
        if float(np.mean(sat)) < 10.0 and float(np.std(sat)) < 8.0:
            return None

        dh = compute_dhash(rgb)
        return rgb, sha, dh
    except Exception:
        return None


class SurfacePoolManager:
    def __init__(self):
        self.shas = {c: set() for c in SURFACE_PATHOLOGY_CLASSES}
        self.dhashes = {c: [] for c in SURFACE_PATHOLOGY_CLASSES}
        self.counts = {c: 0 for c in SURFACE_PATHOLOGY_CLASSES}
        for c in SURFACE_PATHOLOGY_CLASSES:
            (POOL_DIR / c).mkdir(parents=True, exist_ok=True)
            # Load existing if resumed
            for f in (POOL_DIR / c).glob("*.jpg"):
                try:
                    raw = f.read_bytes()
                    sha = hashlib.sha256(raw).hexdigest()
                    with Image.open(io.BytesIO(raw)) as im:
                        dh = compute_dhash(im.convert("RGB"))
                    self.shas[c].add(sha)
                    self.dhashes[c].append(dh)
                    self.counts[c] += 1
                except Exception:
                    f.unlink(missing_ok=True)
        log_msg(f"SurfacePoolManager ready (preloaded {sum(self.counts.values())} pooled files).")

    def try_add(self, cls_name: str, src, prefix: str, min_dim: int = 160, max_hamming: int = 4) -> bool:
        if cls_name not in self.shas:
            return False
        res = is_valid_surface_image(src, min_dim=min_dim)
        if res is None:
            return False
        rgb, sha, dh = res
        if sha in self.shas[cls_name]:
            return False
        for eh in self.dhashes[cls_name]:
            if (dh ^ eh).bit_count() <= max_hamming:
                return False
        self.shas[cls_name].add(sha)
        self.dhashes[cls_name].append(dh)
        self.counts[cls_name] += 1
        out_path = POOL_DIR / cls_name / f"{prefix}_{sha[:10]}.jpg"
        rgb.save(out_path, "JPEG", quality=93)
        return True


def map_legacy_disease_class(name: str) -> str:
    mapping = {
        "Aloe_Healthy": "Healthy_Foliage",
        "Cactus_Healthy": "Healthy_Foliage",
        "Money_Plant_Healthy": "Healthy_Foliage",
        "Snake_Plant_Healthy": "Healthy_Foliage",
        "Spider_Plant_Healthy": "Healthy_Foliage",
        "Aloe_Anthracnose": "Fungal_Leaf_Spot_Anthracnose",
        "Aloe_LeafSpot": "Fungal_Leaf_Spot_Anthracnose",
        "Snake_Plant_Anthracnose": "Fungal_Leaf_Spot_Anthracnose",
        "Spider_Plant_Fungal_leaf_spot": "Fungal_Leaf_Spot_Anthracnose",
        "Universal_Alternaria_Leaf_Spot": "Fungal_Leaf_Spot_Anthracnose",
        "Aloe_Rust": "Rust_Fungi",
        "Universal_Rust_Fungi": "Rust_Fungi",
        "Aloe_Sunburn": "Abiotic_Sunburn_Scald",
        "Cactus_Dactylopius_Opuntia": "Mealybugs_Cochineal",
        "Universal_Mealybugs": "Mealybugs_Cochineal",
        "Money_Plant_Bacterial_wilt_disease": "Bacterial_Wilt_Soft_Rot",
        "Money_Plant_Manganese_Toxicity": "Abiotic_Nutrient_Chlorosis",
        "Snake_Plant_Leaf_Withering": "Abiotic_Dehydration_Underwatering",
        "Spider_Plant_Leaf_Tip_Necrosis": "Abiotic_Dehydration_Underwatering",
        "Universal_Botrytis_Gray_Mold": "Botrytis_Gray_Mold",
        "Universal_Powdery_Mildew": "Powdery_Mildew",
        "Universal_Downy_Mildew": "Downy_Mildew",
        "Universal_Sooty_Mold": "Sooty_Mold",
        "Universal_Leaf_Miners": "Leaf_Miners",
        "Universal_Spider_Mites": "Spider_Mites",
        "Universal_Scale_Insects": "Scale_Insects",
        "Universal_Armored_Scales": "Scale_Insects",
        "Universal_Aphids": "Aphids",
        "Universal_Thrips": "Thrips_Damage",
        "Universal_Whiteflies": "Whiteflies",
    }
    return mapping.get(name)


def map_plantwild_folder(folder: str) -> str:
    s = folder.lower()
    if "powdery mildew" in s:
        return "Powdery_Mildew"
    if "downy mildew" in s or "blue mold" in s:
        return "Downy_Mildew"
    if "rust" in s:
        return "Rust_Fungi"
    if "gray mold" in s or "botrytis" in s or "brown rot" in s or "end rot" in s or "pocket disease" in s:
        return "Botrytis_Gray_Mold"
    if "bacterial" in s or "black rot" in s or "fire blight" in s or "canker" in s or "halo blight" in s or "soft rot" in s:
        return "Bacterial_Wilt_Soft_Rot"
    if "scorch" in s:
        return "Abiotic_Dehydration_Underwatering"
    if "greening" in s or "leafroll" in s:
        return "Abiotic_Nutrient_Chlorosis"
    if any(k in s for k in ["scab", "blight", "spot", "anthracnose", "mold", "streak", "blotch", "blast", "curl"]):
        return "Fungal_Leaf_Spot_Anthracnose"
    return None


def map_plantdoc_folder(folder: str) -> str:
    s = folder.lower()
    if "powdery_mildew" in s:
        return "Powdery_Mildew"
    if "rust" in s:
        return "Rust_Fungi"
    if "spider_mites" in s:
        return "Spider_Mites"
    if "bacterial_spot" in s:
        return "Bacterial_Wilt_Soft_Rot"
    if any(k in s for k in ["scab", "leaf_spot", "black_rot", "blight", "leaf_mold", "target_spot"]):
        return "Fungal_Leaf_Spot_Anthracnose"
    return None


def ingest_legacy_clean(pm: SurfacePoolManager):
    existing_leg = sum(len(list((POOL_DIR / c).glob("leg_*.jpg"))) for c in SURFACE_PATHOLOGY_CLASSES)
    if existing_leg >= 11500:
        log_msg(f"STEP 1: Legacy clean already pooled ({existing_leg} files) -> skipping cleanly.")
        return
    log_msg("STEP 1: Ingesting clean PhytoCare legacy data (stripping saved synthetic aug_* copies)...")
    pd_data = CORETECH_ROOT / "Plant_Disease_Experiment" / "data"
    added = 0
    # Aloe has 1000+ per class; cap Aloe_Healthy/etc. at 300 so Aloe doesn't skew surface pathology
    caps = defaultdict(lambda: 600)
    caps["Aloe_Healthy"] = 250
    caps["Aloe_Anthracnose"] = 300
    caps["Aloe_LeafSpot"] = 300
    caps["Aloe_Rust"] = 350
    caps["Aloe_Sunburn"] = 500
    per_legacy_cnt = defaultdict(int)

    for split in ["train", "val", "test", "internet_benchmark_500"]:
        sp = pd_data / split
        if not sp.exists():
            continue
        for c_dir in sorted(sp.iterdir()):
            if not c_dir.is_dir():
                continue
            target_cls = map_legacy_disease_class(c_dir.name)
            if not target_cls:
                continue
            files = sorted(c_dir.glob("*.*"))
            random.Random(42).shuffle(files)
            for f in files:
                if per_legacy_cnt[c_dir.name] >= caps[c_dir.name]:
                    break
                fn = f.name.lower()
                if fn.startswith("aug_") or "synth" in fn:
                    continue
                if pm.try_add(target_cls, f, prefix=f"leg_{c_dir.name[:12]}", min_dim=160):
                    added += 1
                    per_legacy_cnt[c_dir.name] += 1
    log_msg(f"  -> Legacy clean ingested: +{added} unique surface images.")


def ingest_plantwild_v2(pm: SurfacePoolManager):
    existing_pw = sum(len(list((POOL_DIR / c).glob("pw2_*.jpg"))) for c in SURFACE_PATHOLOGY_CLASSES)
    if existing_pw >= 1500:
        log_msg(f"STEP 2: PlantWild v2 already pooled ({existing_pw} files) -> skipping cleanly.")
        return
    log_msg("STEP 2: Downloading & ingesting PlantWild v2 (EGOISTyrh/mvpdr-plant-disease)...")
    pw_dir = HF_CACHE_STAGE / "plantwild_v2"
    snapshot_download(
        repo_id="EGOISTyrh/mvpdr-plant-disease",
        repo_type="dataset",
        local_dir=str(pw_dir),
        allow_patterns=["plantwild_v2/images/**"],
        max_workers=32,
    )
    img_root = pw_dir / "plantwild_v2" / "images"
    added = 0
    if img_root.exists():
        for cls_dir in sorted(img_root.iterdir()):
            if not cls_dir.is_dir():
                continue
            target_cls = map_plantwild_folder(cls_dir.name)
            if not target_cls:
                continue
            # Cap Fungal_Leaf_Spot_Anthracnose per folder so one crop doesn't dominate
            cap = 80 if target_cls == "Fungal_Leaf_Spot_Anthracnose" else 300
            cnt = 0
            for f in sorted(cls_dir.glob("*.*")):
                if cnt >= cap:
                    break
                if pm.try_add(target_cls, f, prefix="pw2", min_dim=180):
                    added += 1
                    cnt += 1
    log_msg(f"  -> PlantWild v2 ingested: +{added} wild surface images.")


def ingest_plantdoc_extended(pm: SurfacePoolManager):
    existing_pd = sum(len(list((POOL_DIR / c).glob("pdoc_*.jpg"))) for c in SURFACE_PATHOLOGY_CLASSES)
    if existing_pd >= 800:
        log_msg(f"STEP 3: PlantDoc already pooled ({existing_pd} files) -> skipping cleanly.")
        return
    log_msg("STEP 3: Ingesting PlantDoc wild leaf dataset from parquet (agyaatcoder/PlantDoc)...")
    shutil.rmtree(HF_CACHE_STAGE / "plantdoc_ext", ignore_errors=True)
    added = 0
    for sh in ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet", "data/test-00000-of-00001.parquet"]:
        local_pq = hf_hub_download("agyaatcoder/PlantDoc", sh, repo_type="dataset", local_dir=str(HF_CACHE_STAGE / "plantdoc_pq"))
        df = pl.read_parquet(local_pq)
        for row in df.iter_rows(named=True):
            cats = [str(c).lower() for c in ((row.get("objects") or {}).get("category") or [])]
            if not cats:
                continue
            mapped = []
            for cn in cats:
                tc = map_plantdoc_folder(cn.replace(" ", "_"))
                if not tc and "leaf" in cn and not any(k in cn for k in ["spot", "scab", "rust", "mildew", "blight", "rot", "mold", "virus", "mite", "mosaic"]):
                    tc = "Healthy_Foliage"
                if tc:
                    mapped.append(tc)
            if not mapped:
                continue
            # Use majority category of objects on leaf
            target_cls = max(set(mapped), key=mapped.count)
            if pm.try_add(target_cls, row["image"]["bytes"], prefix="pdoc", min_dim=180):
                added += 1
    log_msg(f"  -> PlantDoc parquet ingested: +{added} wild surface images.")


def ingest_plant_pathology_2021(pm: SurfacePoolManager):
    existing_pp = sum(len(list((POOL_DIR / c).glob("pp21_*.jpg"))) for c in SURFACE_PATHOLOGY_CLASSES)
    if existing_pp >= 1500:
        log_msg(f"STEP 4: FGVC8 PP2021 already pooled ({existing_pp} files) -> skipping cleanly.")
        return
    log_msg("STEP 4: Ingesting FGVC8 Plant Pathology 2021 (timm/plant-pathology-2021 validation + 6 train shards)...")
    api = HfApi()
    all_files = api.list_repo_files("timm/plant-pathology-2021", repo_type="dataset")
    shards = [f for f in all_files if "validation-" in f] + sorted([f for f in all_files if "train-" in f])[:6]

    added_per_cls = defaultdict(int)
    caps = {"Healthy_Foliage": 600, "Powdery_Mildew": 1000, "Rust_Fungi": 600, "Fungal_Leaf_Spot_Anthracnose": 500}

    for sh in shards:
        if all(added_per_cls[k] >= caps[k] for k in caps):
            break
        local_pq = hf_hub_download("timm/plant-pathology-2021", sh, repo_type="dataset", local_dir=str(HF_CACHE_STAGE / "pp2021"))
        df = pl.read_parquet(local_pq)
        for row in df.iter_rows(named=True):
            lbls = row.get("label_names") or []
            if len(lbls) != 1:
                continue  # Keep single-symptom clean examples
            lbl = lbls[0]
            if lbl == "healthy":
                target_cls = "Healthy_Foliage"
            elif lbl == "powdery_mildew":
                target_cls = "Powdery_Mildew"
            elif lbl == "rust":
                target_cls = "Rust_Fungi"
            elif lbl in {"frog_eye_leaf_spot", "scab"}:
                target_cls = "Fungal_Leaf_Spot_Anthracnose"
            else:
                continue
            if added_per_cls[target_cls] >= caps[target_cls]:
                continue
            img_bytes = row["image"]["bytes"]
            if pm.try_add(target_cls, img_bytes, prefix="pp21", min_dim=224):
                added_per_cls[target_cls] += 1
    log_msg(f"  -> FGVC8 PP2021 ingested: {dict(added_per_cls)}")


def ingest_plantvillage_selective(pm: SurfacePoolManager):
    existing_pv = sum(len(list((POOL_DIR / c).glob("pv_*.jpg"))) for c in SURFACE_PATHOLOGY_CLASSES)
    if existing_pv >= 2500:
        log_msg(f"STEP 5: PlantVillage already pooled ({existing_pv} files) -> skipping cleanly.")
        return
    log_msg("STEP 5: Selectively ingesting PlantVillage (tagged pv_ for dynamic indoor background compositing)...")
    # Label names verified from BrandonFors/Plant-Diseases-PlantVillage-Dataset README
    pv_map = {
        3: "Healthy_Foliage", 4: "Healthy_Foliage", 6: "Healthy_Foliage", 10: "Healthy_Foliage",
        14: "Healthy_Foliage", 17: "Healthy_Foliage", 19: "Healthy_Foliage", 22: "Healthy_Foliage",
        24: "Healthy_Foliage", 27: "Healthy_Foliage", 37: "Healthy_Foliage",
        5: "Powdery_Mildew", 25: "Powdery_Mildew",
        2: "Rust_Fungi", 8: "Rust_Fungi",
        16: "Bacterial_Wilt_Soft_Rot", 18: "Bacterial_Wilt_Soft_Rot", 28: "Bacterial_Wilt_Soft_Rot",
        33: "Spider_Mites",
        15: "Abiotic_Nutrient_Chlorosis",  # Citrus greening asymmetrical chlorosis
        0: "Fungal_Leaf_Spot_Anthracnose", 1: "Fungal_Leaf_Spot_Anthracnose", 7: "Fungal_Leaf_Spot_Anthracnose",
        11: "Fungal_Leaf_Spot_Anthracnose", 13: "Fungal_Leaf_Spot_Anthracnose", 20: "Fungal_Leaf_Spot_Anthracnose",
        26: "Fungal_Leaf_Spot_Anthracnose", 29: "Fungal_Leaf_Spot_Anthracnose", 31: "Fungal_Leaf_Spot_Anthracnose",
        32: "Fungal_Leaf_Spot_Anthracnose", 34: "Fungal_Leaf_Spot_Anthracnose",
    }
    caps = {
        "Healthy_Foliage": 500,
        "Powdery_Mildew": 600,
        "Rust_Fungi": 400,
        "Bacterial_Wilt_Soft_Rot": 500,
        "Spider_Mites": 750,
        "Abiotic_Nutrient_Chlorosis": 450,
        "Fungal_Leaf_Spot_Anthracnose": 450,
    }
    added_per_cls = defaultdict(int)
    for sh in ["data/test-00000-of-00001.parquet", "data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]:
        local_pq = hf_hub_download("BrandonFors/Plant-Diseases-PlantVillage-Dataset", sh, repo_type="dataset", local_dir=str(HF_CACHE_STAGE / "pv"))
        df = pl.read_parquet(local_pq)
        for row in df.iter_rows(named=True):
            lbl = row["label"]
            target_cls = pv_map.get(lbl)
            if not target_cls or added_per_cls[target_cls] >= caps[target_cls]:
                continue
            if pm.try_add(target_cls, row["image"]["bytes"], prefix="pv", min_dim=180):
                added_per_cls[target_cls] += 1
    log_msg(f"  -> PlantVillage ingested (tagged pv_): {dict(added_per_cls)}")


def ingest_healthy_hard_negatives(pm: SurfacePoolManager):
    """Pulls clean variegated, striped, and spiny indoor plants so healthy patterns never false-alarm."""
    existing_hh = len(list((POOL_DIR / "Healthy_Foliage").glob("hhneg_*.jpg")))
    if existing_hh >= 800:
        log_msg(f"STEP 6: Variegated/Spiny Healthy Hard Negatives already pooled ({existing_hh}) -> skipping cleanly.")
        return
    log_msg("STEP 6: Adding variegated & spiny Healthy Hard Negatives from Botanical ID corpus...")
    id_pool = DATA_DIR / "botanical_id_pool"
    if not id_pool.exists():
        id_pool = CORETECH_ROOT / "Plant_Identification_Experiment" / "data" / "train"

    targets = [
        "Calathea", "Calathea Medallion", "Calathea Zebrina", "Calathea Warscewiczii",
        "Croton", "Dumb Cane", "Nerve Plant", "Polka Dot Plant", "Watermelon Peperomia",
        "Golden Barrel Cactus", "Bunny Ears Cactus", "Columnar Cactus", "Pincushion Cactus",
        "Moon Cactus", "Snake plant", "Cylindrical Snake Plant", "Pothos", "Satin Pothos",
        "Spider Plant", "Aloe Vera", "Monstera Deliciosa", "Zebra Haworthia"
    ]
    added = 0
    for d in sorted(id_pool.iterdir()):
        if not d.is_dir():
            continue
        if any(t.lower() in d.name.lower() for t in targets):
            files = sorted(d.glob("*.jpg"))[:45]
            for f in files:
                if pm.try_add("Healthy_Foliage", f, prefix="hhneg", min_dim=200):
                    added += 1
    log_msg(f"  -> Variegated/Spiny Healthy Hard Negatives added: +{added}")


def harvest_thin_surface_classes(pm: SurfacePoolManager):
    """Expands under-populated surface pest & abiotic stress classes via verified iNaturalist + clean plant filter."""
    import requests
    log_msg("STEP 7: Expanding thin surface pest & abiotic stress classes from iNaturalist...")
    # Target taxa verified to produce real plant-surface damage / colonies
    INAT_SURFACE_EXPANSION = {
        "Leaf_Miners": [("48086", 550)],               # Agromyzidae leaf mines (predominantly mined leaves!)
        "Sooty_Mold": [("154563", 550), ("338190", 550)],
        "Scale_Insects": [("49456", 500), ("48093", 500)],
        "Mealybugs_Cochineal": [("52788", 550), ("202091", 550)],
        "Botrytis_Gray_Mold": [("324263", 550)],
        "Powdery_Mildew": [("55525", 700)],
        "Downy_Mildew": [("52598", 650)],
        "Thrips_Damage": [("83443", 500)],
        "Whiteflies": [("154784", 500)],
        "Aphids": [("52381", 550)],
    }

    # Prefer fast Stage 1B YOLO11m-Seg when available for foliage hygiene gate
    yoloe_model = None
    seg_pt = V2_ROOT / "artifacts" / "yolo_plant_seg.pt"
    yoloe_path = V2_ROOT / "artifacts" / "yoloe-11l-seg.pt"
    try:
        if seg_pt.exists():
            from ultralytics import YOLO
            yoloe_model = YOLO(str(seg_pt))
            yoloe_model.to("cuda")
            log_msg("  [Surface Filter] Stage 1B YOLO11m-Seg plant-foliage hygiene gate active!")
        elif yoloe_path.exists():
            from ultralytics import YOLOE
            yoloe_model = YOLOE(str(yoloe_path))
            prompts = ["plant foliage", "plant leaf", "cactus", "plant stem"]
            yoloe_model.set_classes(prompts, yoloe_model.get_text_pe(prompts))
            yoloe_model.to("cuda")
            log_msg("  [Surface Filter] YOLOE plant-foliage hygiene gate active!")
    except Exception as e:
        log_msg(f"  [Surface Filter Warning] {e}")

    sess = requests.Session()
    headers = {"User-Agent": "CoreTechV2PathologyHarvester/1.0"}

    for cls_name, taxon_specs in INAT_SURFACE_EXPANSION.items():
        target_cap = 500
        if pm.counts[cls_name] >= target_cap:
            log_msg(f"  [✓] {cls_name}: already {pm.counts[cls_name]}/{target_cap}")
            continue
        log_msg(f"  [*] Expanding '{cls_name}' (current={pm.counts[cls_name]}, target={target_cap})...")
        for tid, _ in taxon_specs:
            if pm.counts[cls_name] >= target_cap:
                break
            for page in range(1, 5):
                if pm.counts[cls_name] >= target_cap:
                    break
                try:
                    resp = sess.get(
                        "https://api.inaturalist.org/v1/observations",
                        params={"taxon_id": tid, "quality_grade": "research", "photos": "true", "per_page": 150, "page": page, "order_by": "id"},
                        headers=headers,
                        timeout=15,
                    )
                    if resp.status_code == 429:
                        time.sleep(25)
                        continue
                    if resp.status_code != 200:
                        break
                    obs_list = resp.json().get("results", [])
                    if not obs_list:
                        break
                    for obs in obs_list:
                        if pm.counts[cls_name] >= target_cap:
                            break
                        for p in obs.get("photos", [])[:1]:
                            u = p.get("url", "")
                            if not u:
                                continue
                            lu = u.replace("/square.", "/large.").replace("/small.", "/large.").replace("/medium.", "/large.")
                            try:
                                ir = sess.get(lu, headers=headers, timeout=8)
                                if ir.status_code != 200 or len(ir.content) < 8000:
                                    continue
                                if yoloe_model is not None:
                                    im_chk = Image.open(io.BytesIO(ir.content)).convert("RGB")
                                    res = yoloe_model.predict(im_chk, conf=0.14, imgsz=384, verbose=False)[0]
                                    if res.masks is None or len(res.masks.data) == 0:
                                        continue  # Reject photos with no visible plant leaf/stem surface!
                                pm.try_add(cls_name, ir.content, prefix=f"inat_{tid}", min_dim=200)
                            except Exception:
                                continue
                    time.sleep(0.8)
                except Exception:
                    break
        log_msg(f"  -> {cls_name} count after expansion: {pm.counts[cls_name]}")


def harvest_abiotic_stress_classes(pm: SurfacePoolManager):
    """Harvests real plant dehydration, overwatering/edema, chlorosis, and sunburn photos."""
    import requests
    log_msg("STEP 7B: Expanding Abiotic Surface Stress classes via iNaturalist Plantae + Openverse...")
    ABIOTIC_QUERIES = {
        "Abiotic_Dehydration_Underwatering": {
            "inat": ["underwatered", "wilting", "dehydrated", "drought"],
            "openverse": [],
            "target": 700,
        },
        "Abiotic_Overwatering_Edema": {
            "inat": ["overwatered", "edema", "oedema", "root rot", "waterlogged", "damping off", "soggy"],
            "openverse": [],
            "target": 500,
        },
        "Abiotic_Nutrient_Chlorosis": {
            "inat": ["chlorosis", "nutrient deficiency", "iron deficiency"],
            "openverse": [],
            "target": 700,
        },
        "Abiotic_Sunburn_Scald": {
            "inat": ["sunburn", "sunscald"],
            "openverse": [],
            "target": 650,
        },
    }

    # Prefer our freshly trained fast Stage 1B YOLO11m-Seg model when present!
    yoloe_model = None
    seg_pt = V2_ROOT / "artifacts" / "yolo_plant_seg.pt"
    yoloe_path = V2_ROOT / "artifacts" / "yoloe-11l-seg.pt"
    try:
        if seg_pt.exists():
            from ultralytics import YOLO
            yoloe_model = YOLO(str(seg_pt))
            yoloe_model.to("cuda")
        elif yoloe_path.exists():
            from ultralytics import YOLOE
            yoloe_model = YOLOE(str(yoloe_path))
            prompts = ["plant foliage", "plant leaf", "cactus", "succulent"]
            yoloe_model.set_classes(prompts, yoloe_model.get_text_pe(prompts))
            yoloe_model.to("cuda")
    except Exception:
        pass

    sess = requests.Session()
    headers = {"User-Agent": "CoreTechV2AbioticHarvester/1.0"}

    def accept_url(cls_name: str, url: str, prefix: str):
        try:
            ir = sess.get(url, headers=headers, timeout=8)
            if ir.status_code != 200 or len(ir.content) < 8000:
                return
            if yoloe_model is not None:
                im_chk = Image.open(io.BytesIO(ir.content)).convert("RGB")
                res = yoloe_model.predict(im_chk, conf=0.14, imgsz=384, verbose=False)[0]
                if res.masks is None or len(res.masks.data) == 0:
                    return
            pm.try_add(cls_name, ir.content, prefix=prefix, min_dim=180)
        except Exception:
            pass

    for cls_name, spec in ABIOTIC_QUERIES.items():
        target = spec["target"]
        if pm.counts[cls_name] >= target:
            log_msg(f"  [✓] {cls_name}: already {pm.counts[cls_name]}/{target}")
            continue
        log_msg(f"  [*] Expanding '{cls_name}' (current={pm.counts[cls_name]}, target={target})...")
        for q in spec["inat"]:
            if pm.counts[cls_name] >= target:
                break
            for page in range(1, 5):
                if pm.counts[cls_name] >= target:
                    break
                try:
                    r = sess.get(
                        "https://api.inaturalist.org/v1/observations",
                        params={"q": q, "taxon_id": "47126", "photos": "true", "per_page": 150, "page": page},
                        headers=headers,
                        timeout=15,
                    )
                    if r.status_code == 429:
                        time.sleep(25)
                        continue
                    if r.status_code != 200:
                        break
                    for obs in r.json().get("results", []):
                        if pm.counts[cls_name] >= target:
                            break
                        for p in obs.get("photos", [])[:1]:
                            u = p.get("url", "")
                            if u:
                                lu = u.replace("/square.", "/large.").replace("/small.", "/large.").replace("/medium.", "/large.")
                                accept_url(cls_name, lu, f"ab_inat_{q[:6]}")
                    time.sleep(0.8)
                except Exception:
                    break

        for oq in spec["openverse"]:
            if pm.counts[cls_name] >= target:
                break
            try:
                r = sess.get("https://api.openverse.org/v1/images/", params={"q": oq, "page_size": 40}, headers=headers, timeout=12)
                if r.status_code == 200:
                    for item in r.json().get("results", []):
                        if pm.counts[cls_name] >= target:
                            break
                        u = item.get("url")
                        if u:
                            accept_url(cls_name, u, f"ab_ov_{oq[:6].replace(' ', '_')}")
            except Exception:
                pass
        log_msg(f"  -> {cls_name} count after abiotic expansion: {pm.counts[cls_name]}")


def build_balanced_splits():
    log_msg("STEP 8: Building clean, leak-free train / val / test splits for all 19 surface classes...")
    manifest = {}
    for cls_name in SURFACE_PATHOLOGY_CLASSES:
        all_files = sorted((POOL_DIR / cls_name).glob("*.jpg"))
        rng = random.Random(f"path_v2_{cls_name}")
        # Prioritize authentic wild photos for test/val and cap train at 1800 so dominant classes don't drown rare classes
        wild_files = [f for f in all_files if not f.name.startswith("pv_")]
        pv_files = [f for f in all_files if f.name.startswith("pv_")]
        rng.shuffle(wild_files)
        rng.shuffle(pv_files)
        files = wild_files + pv_files
        total = len(files)
        n_test = min(100, max(20, int(round(total * 0.12)))) if total >= 60 else max(1, total // 6)
        n_val = min(100, max(20, int(round(total * 0.12)))) if total >= 60 else max(1, total // 6)

        test_files = files[:n_test]
        val_files = files[n_test:n_test + n_val]
        train_pool = files[n_test + n_val:]
        if len(train_pool) > 1800:
            rng.shuffle(train_pool)
            # Keep all non-pv up to 1500, plus pv mix up to 1800
            non_pv_tr = [f for f in train_pool if not f.name.startswith("pv_")][:1500]
            pv_tr = [f for f in train_pool if f.name.startswith("pv_")][:max(0, 1800 - len(non_pv_tr))]
            train_files = (non_pv_tr + pv_tr)[:1800]
        else:
            train_files = train_pool

        splits = {
            "test": test_files,
            "val": val_files,
            "train": train_files,
        }
        for sp_name, sp_files in splits.items():
            dst_dir = PATH_DATA_DIR / sp_name / cls_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in sp_files:
                dst = dst_dir / f.name
                if not dst.exists():
                    try:
                        os.link(f, dst)
                    except Exception:
                        shutil.copy2(f, dst)
        manifest[cls_name] = {k: len(v) for k, v in splits.items()}
        log_msg(f"  {cls_name:34s} | total={total:4d} -> {manifest[cls_name]}")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Clean staging parquet cache to reclaim space cleanly
    shutil.rmtree(HF_CACHE_STAGE, ignore_errors=True)
    log_msg("PATHOLOGY CORPUS BUILD COMPLETE!")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hf", "inat", "all"], default="all")
    args = ap.parse_args()

    pm = SurfacePoolManager()
    if args.stage in ("hf", "all"):
        ingest_legacy_clean(pm)
        ingest_plantwild_v2(pm)
        ingest_plantdoc_extended(pm)
        ingest_plant_pathology_2021(pm)
        ingest_plantvillage_selective(pm)
    if args.stage in ("inat", "all"):
        ingest_healthy_hard_negatives(pm)
        harvest_thin_surface_classes(pm)
        harvest_abiotic_stress_classes(pm)
        build_balanced_splits()


if __name__ == "__main__":
    main()
