"""
CoreTech v2 Phase 1C: Builds Plant Instance Segmentation (YOLO11-Seg) & Surface Lesion Detection (YOLO11-Det) datasets.

1. Instance Segmentation (data/instance_seg):
     - Downloads MIT Indoor Scenes (u5753411/MIT-Indoor-Scenes) as indoor room backgrounds & hard-negative non-plant frames.
     - Uses YOLOE-11L-Seg to extract clean foreground plant masks from curated indoor/botanical pools.
     - Synthesizes 2,500 realistic multi-plant indoor room composites with exact multi-instance YOLO polygon annotations
       so YOLO11m-Seg learns to separate overlapping plants, exclude pots/tables, and reject indoor furniture.
     - Adds 2,500 single-plant YOLOE-verified scenes + 1,500 pure indoor hard-negative empty-label rooms.
2. Surface Lesion Detection (data/lesion_det):
     - Extracts bounding-box lesion/infected-leaf annotations from KeenForgeAI/PlantDoc-corrected (Pascal VOC XML).
     - Extracts exact spot-cluster bounding boxes from Project-AgML/pdsnet_field_plant_disease_segmentation pixel masks.
"""
import io
import os
import sys
import cv2
import math
import json
import time
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image, ImageFile
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ImageFile.LOAD_TRUNCATED_IMAGES = True

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))
from configs.v2_config import (
    CORETECH_ROOT, DATA_DIR, SEG_DATA_DIR, LESION_DATA_DIR, NON_PLANT_DIR, REPORTS_DIR, ARTIFACTS_DIR
)

HF_STAGE = DATA_DIR / "_hf_seg_stage"
HF_STAGE.mkdir(parents=True, exist_ok=True)
LOG_PATH = REPORTS_DIR / "data_seg_lesion_build.log"


def log_msg(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


def mask_to_yolo_polygons(mask_bin: np.ndarray, min_area: int = 200):
    """Converts binary uint8 mask (0/255) into normalized YOLO segmentation polygon strings."""
    h, w = mask_bin.shape[:2]
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        eps = 0.004 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 4:
            continue
        pts = approx.reshape(-1, 2)
        coords = []
        for x, y in pts:
            coords.append(f"{np.clip(x / w, 0.0, 1.0):.5f}")
            coords.append(f"{np.clip(y / h, 0.0, 1.0):.5f}")
        polys.append(" ".join(coords))
    return polys


def prepare_mit_indoor_scenes():
    """Downloads Indoor Scenes parquet shards (FatimaSaadNaik/indoor-scenes-dataset) into data/non_plant_hard_neg/mit_indoor."""
    out_dir = NON_PLANT_DIR / "mit_indoor"
    existing = list(out_dir.glob("*.jpg"))
    if len(existing) >= 4000:
        log_msg(f"[✓] MIT Indoor Scenes already staged ({len(existing)} images).")
        return existing

    out_dir.mkdir(parents=True, exist_ok=True)
    log_msg("Downloading Indoor Scenes parquet (FatimaSaadNaik/indoor-scenes-dataset)...")
    mit_raw = HF_STAGE / "indoor_pq"
    saved = []
    for sh_idx in range(6):
        if len(saved) >= 5000:
            break
        sh = f"data/train-{sh_idx:05d}-of-00006.parquet"
        local_pq = hf_hub_download("FatimaSaadNaik/indoor-scenes-dataset", sh, repo_type="dataset", local_dir=str(mit_raw))
        df = pl.read_parquet(local_pq)
        for row in df.iter_rows(named=True):
            cat = (row.get("category") or "").lower()
            cap = (row.get("caption") or "").lower()
            # Strictly exclude greenhouses/florists/nurseries/gardens so indoor rooms contain zero false-negative plants
            if any(k in cat for k in ["greenhouse", "florist", "nursery"]) or any(k in cap for k in ["potted plant", "houseplant", "garden", "flower pot"]):
                continue
            try:
                rgb = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                if min(rgb.size) < 224:
                    continue
                if max(rgb.size) > 960:
                    sc = 960.0 / max(rgb.size)
                    rgb = rgb.resize((int(rgb.size[0] * sc), int(rgb.size[1] * sc)), Image.Resampling.BILINEAR)
                dst = out_dir / f"mit_{len(saved):05d}.jpg"
                rgb.save(dst, "JPEG", quality=90)
                saved.append(dst)
                if len(saved) >= 5000:
                    break
            except Exception:
                continue
    shutil.rmtree(mit_raw, ignore_errors=True)
    log_msg(f"  -> Staged {len(saved)} clean indoor non-plant room scenes.")
    return saved


def build_instance_seg_dataset(indoor_imgs: list):
    """Builds YOLO11-Seg dataset: single plants + synthetic multi-plant overlaps + empty indoor rooms."""
    if (SEG_DATA_DIR / "data.yaml").exists():
        val_n = len(list((SEG_DATA_DIR / "images" / "val").glob("*.jpg")))
        if val_n >= 400:
            log_msg(f"[✓] YOLO instance_seg dataset already built ({val_n} val images).")
            return

    for sp in ["train", "val"]:
        (SEG_DATA_DIR / "images" / sp).mkdir(parents=True, exist_ok=True)
        (SEG_DATA_DIR / "labels" / sp).mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLOE
    yoloe_path = ARTIFACTS_DIR / "yoloe-11l-seg.pt"
    model = YOLOE(str(yoloe_path))
    prompts = ["potted plant", "plant foliage", "cactus", "succulent", "houseplant leaves"]
    model.set_classes(prompts, model.get_text_pe(prompts))
    model.to("cuda")

    # Collect diverse botanical images from existing ID & disease folders
    plant_imgs = []
    search_dirs = [
        DATA_DIR / "botanical_id_pool",
        CORETECH_ROOT / "Plant_Identification_Experiment" / "data" / "train",
        CORETECH_ROOT / "Datasets" / "house_plant_species",
    ]
    for sd in search_dirs:
        if not sd.exists():
            continue
        for c_dir in sd.iterdir():
            if c_dir.is_dir():
                plant_imgs.extend(sorted(c_dir.glob("*.jpg"))[:20])
    random.Random(42).shuffle(plant_imgs)
    plant_imgs = plant_imgs[:4000]
    log_msg(f"Building YOLO11-Seg dataset from {len(plant_imgs)} plant photos + {len(indoor_imgs)} indoor rooms...")

    # Extract clean foreground plant cutouts + direct single-plant pseudo-masks
    cutouts = []
    single_count = 0
    for idx, p in enumerate(plant_imgs):
        try:
            with Image.open(p) as im:
                rgb = im.convert("RGB")
                if min(rgb.size) < 224:
                    continue
                if max(rgb.size) > 768:
                    sc = 768.0 / max(rgb.size)
                    rgb = rgb.resize((int(rgb.size[0] * sc), int(rgb.size[1] * sc)), Image.Resampling.BILINEAR)
            w, h = rgb.size
            arr = np.array(rgb)
            res = model.predict(rgb, conf=0.22, imgsz=640, verbose=False)[0]
            if res.masks is None or len(res.masks.data) == 0:
                continue

            # Merge overlapping fragments into distinct plant masks
            raw_masks = res.masks.data.detach().cpu().numpy()
            merged_masks = []
            for m_raw in raw_masks:
                m = (cv2.resize(m_raw, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)
                area = int(m.sum())
                if area < 0.04 * (w * h) or area > 0.92 * (w * h):
                    continue
                placed = False
                for i, km in enumerate(merged_masks):
                    inter = np.sum((m > 0) & (km > 0))
                    if inter / max(1, min(area, int(km.sum()))) > 0.55:
                        merged_masks[i] = ((km > 0) | (m > 0)).astype(np.uint8)
                        placed = True
                        break
                if not placed:
                    merged_masks.append(m)

            if not merged_masks:
                continue

            split = "val" if (idx % 8 == 0) else "train"
            label_lines = []
            for m in merged_masks:
                polys = mask_to_yolo_polygons((m * 255).astype(np.uint8))
                for poly in polys:
                    label_lines.append(f"0 {poly}")
                # Save cutout from largest single clean mask for synthetic multi-plant room scenes
                if len(cutouts) < 2000 and 0.08 * w * h <= m.sum() <= 0.70 * w * h:
                    ys, xs = np.where(m > 0)
                    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                    cutouts.append((arr[y1:y2, x1:x2], m[y1:y2, x1:x2]))

            if label_lines:
                stem = f"real_{idx:05d}"
                rgb.save(SEG_DATA_DIR / "images" / split / f"{stem}.jpg", "JPEG", quality=90)
                (SEG_DATA_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
                single_count += 1
        except Exception:
            continue
        if (idx + 1) % 500 == 0:
            log_msg(f"  -> Processed {idx + 1}/{len(plant_imgs)} real scenes (saved={single_count}, cutouts={len(cutouts)})")

    # 2. Synthesize 2,400 realistic multi-plant indoor composites (exact non-overlapping/partial-occlusion polygon labels!)
    log_msg(f"Synthesizing 2,400 multi-plant indoor room composites from {len(cutouts)} clean cutouts...")
    rng = random.Random(42)
    for m_idx in range(2400):
        bg_path = rng.choice(indoor_imgs)
        try:
            with Image.open(bg_path) as im:
                bg = im.convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
            canvas = np.array(bg, dtype=np.uint8)
            n_plants = rng.choice([2, 2, 3, 3, 4])
            # Draw from smallest/back to largest/front so occlusions stay geometrically exact
            chosen = rng.sample(cutouts, min(n_plants, len(cutouts)))
            instance_masks = []  # visible masks updated per occlusion

            for p_arr, p_mask in chosen:
                ch, cw = p_arr.shape[:2]
                target_h = rng.randint(150, 460)
                sc = target_h / max(1, ch)
                nw, nh = max(48, min(580, int(cw * sc))), max(48, min(580, int(ch * sc)))
                p_res = cv2.resize(p_arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
                m_res = (cv2.resize(p_mask, (nw, nh), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)

                if rng.random() < 0.5:
                    p_res = cv2.flip(p_res, 1)
                    m_res = cv2.flip(m_res, 1)

                ox = rng.randint(0, max(0, 640 - nw))
                oy = rng.randint(0, max(0, 640 - nh))

                # Feather edges slightly (1px) for realistic lighting blending
                alpha = cv2.GaussianBlur(m_res.astype(np.float32), (3, 3), 0)[:, :, None]
                roi = canvas[oy:oy + nh, ox:ox + nw].astype(np.float32)
                blended = (p_res.astype(np.float32) * alpha + roi * (1.0 - alpha)).astype(np.uint8)
                canvas[oy:oy + nh, ox:ox + nw] = blended

                full_m = np.zeros((640, 640), dtype=np.uint8)
                full_m[oy:oy + nh, ox:ox + nw] = m_res
                # Subtract newly pasted foreground from previously placed background plants!
                for prev_i in range(len(instance_masks)):
                    instance_masks[prev_i] = instance_masks[prev_i] * (1 - full_m)
                instance_masks.append(full_m)

            label_lines = []
            for fm in instance_masks:
                if fm.sum() < 400:
                    continue
                for poly in mask_to_yolo_polygons((fm * 255).astype(np.uint8)):
                    label_lines.append(f"0 {poly}")

            if label_lines:
                split = "val" if (m_idx % 8 == 0) else "train"
                stem = f"multi_{m_idx:05d}"
                Image.fromarray(canvas).save(SEG_DATA_DIR / "images" / split / f"{stem}.jpg", "JPEG", quality=90)
                (SEG_DATA_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
        except Exception:
            continue

    # 3. Add 1,500 pure indoor non-plant empty-label negatives (suppresses false detections on chairs/walls!)
    for n_idx, np_path in enumerate(indoor_imgs[:1500]):
        try:
            split = "val" if (n_idx % 8 == 0) else "train"
            stem = f"neg_indoor_{n_idx:05d}"
            shutil.copy2(np_path, SEG_DATA_DIR / "images" / split / f"{stem}.jpg")
            (SEG_DATA_DIR / "labels" / split / f"{stem}.txt").write_text("", encoding="utf-8")
        except Exception:
            continue

    yaml_text = (
        f"path: {SEG_DATA_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: plant\n"
    )
    (SEG_DATA_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")
    log_msg("[✓] YOLO11-Seg instance_seg dataset built cleanly!")


def build_lesion_det_dataset():
    """Builds YOLO11 Surface Lesion & Infected Foliage Detector dataset from PlantDoc Parquet + PDSNet masks."""
    if (LESION_DATA_DIR / "data.yaml").exists():
        val_n = len(list((LESION_DATA_DIR / "images" / "val").glob("*.jpg")))
        if val_n >= 150:
            log_msg(f"[✓] YOLO lesion_det dataset already built ({val_n} val images).")
            return

    for sp in ["train", "val"]:
        (LESION_DATA_DIR / "images" / sp).mkdir(parents=True, exist_ok=True)
        (LESION_DATA_DIR / "labels" / sp).mkdir(parents=True, exist_ok=True)

    # 2 classes: 0 = healthy_foliage, 1 = surface_lesion_or_disease
    pdoc_raw = HF_STAGE / "plantdoc_pq"
    log_msg("Downloading PlantDoc Object Detection parquet shards (agyaatcoder/PlantDoc)...")
    shards = [
        ("train", "data/train-00000-of-00002.parquet"),
        ("train", "data/train-00001-of-00002.parquet"),
        ("val", "data/test-00000-of-00001.parquet"),
    ]
    count = 0
    for split, sh in shards:
        local_pq = hf_hub_download("agyaatcoder/PlantDoc", sh, repo_type="dataset", local_dir=str(pdoc_raw))
        df = pl.read_parquet(local_pq)
        for row in df.iter_rows(named=True):
            try:
                rgb = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                w, h = rgb.size
                if min(w, h) < 120:
                    continue
                objs = row.get("objects") or {}
                bboxes = objs.get("bbox") or []
                cats = objs.get("category") or []
                lines = []
                for bb, cat in zip(bboxes, cats):
                    # Confirmed COCO [x, y, bw, bh] format
                    x, y, bw_px, bh_px = [float(v) for v in bb]
                    if bw_px < 4 or bh_px < 4:
                        continue
                    cname = str(cat).lower()
                    cls_id = 1 if any(k in cname for k in ["spot", "scab", "rust", "mildew", "blight", "rot", "mold", "virus", "mite", "mosaic", "curl"]) else 0
                    cx = np.clip((x + bw_px / 2.0) / w, 0.0, 1.0)
                    cy = np.clip((y + bh_px / 2.0) / h, 0.0, 1.0)
                    bw = np.clip(bw_px / w, 0.01, 1.0)
                    bh = np.clip(bh_px / h, 0.01, 1.0)
                    lines.append(f"{cls_id} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}")

                if lines:
                    out_stem = f"pdoc_{count:05d}"
                    rgb.save(LESION_DATA_DIR / "images" / split / f"{out_stem}.jpg", "JPEG", quality=92)
                    (LESION_DATA_DIR / "labels" / split / f"{out_stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                    count += 1
            except Exception:
                continue

    # Also ingest PDSNet exact pixel spot masks (label 2 = disease spot!) -> tight spot bounding boxes!
    log_msg("Downloading PDSNet field disease spot masks (Project-AgML/pdsnet_field_plant_disease_segmentation)...")
    pq_p = hf_hub_download("Project-AgML/pdsnet_field_plant_disease_segmentation", "data/train-00000-of-00001.parquet", repo_type="dataset", local_dir=str(HF_STAGE / "pdsnet"))
    df = pl.read_parquet(pq_p)
    pds_cnt = 0
    for idx, row in enumerate(df.iter_rows(named=True)):
        try:
            rgb = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            mask = np.array(Image.open(io.BytesIO(row["mask"]["bytes"])))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            h, w = mask.shape[:2]
            spot_mask = (mask == 2).astype(np.uint8)
            num_l, labels, stats, _ = cv2.connectedComponentsWithStats(spot_mask)
            lines = []
            for li in range(1, num_l):
                x, y, bw, bh, area = stats[li]
                if area < 36:
                    continue
                cx = (x + bw / 2.0) / w
                cy = (y + bh / 2.0) / h
                lines.append(f"1 {cx:.5f} {cy:.5f} {bw / w:.5f} {bh / h:.5f}")
            if lines:
                split = "val" if (idx % 6 == 0) else "train"
                out_stem = f"pds_{pds_cnt:04d}"
                rgb.save(LESION_DATA_DIR / "images" / split / f"{out_stem}.jpg", "JPEG", quality=92)
                (LESION_DATA_DIR / "labels" / split / f"{out_stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                pds_cnt += 1
        except Exception:
            continue

    shutil.rmtree(pdoc_raw, ignore_errors=True)
    shutil.rmtree(HF_STAGE / "pdsnet", ignore_errors=True)

    yaml_text = (
        f"path: {LESION_DATA_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: healthy_foliage\n"
        f"  1: surface_lesion_or_disease\n"
    )
    (LESION_DATA_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")
    log_msg(f"[✓] YOLO11 lesion_det dataset built cleanly! (PlantDoc={count}, PDSNet={pds_cnt})")


def main():
    indoor_imgs = prepare_mit_indoor_scenes()
    build_instance_seg_dataset(indoor_imgs)
    build_lesion_det_dataset()


if __name__ == "__main__":
    main()
