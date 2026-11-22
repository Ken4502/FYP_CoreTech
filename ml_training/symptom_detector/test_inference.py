"""
UrbanGreen CoreTech: Standalone Single-Image Symptom Inference Tester.
Validates 6-class contract compliance, prints raw detection details,
and saves annotated preview images.

Usage:
  python test_inference.py --weights <path_to_best.pt> --image <path_to_photo> [--conf 0.45]
"""
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TARGET_CLASSES = {
    0: "yellowing_chlorosis",
    1: "crispy_brown_edges",
    2: "dark_spots_necrosis",
    3: "white_powdery_residue",
    4: "chewed_holes_stippling",
    5: "healthy_leaf",
}

CLASS_COLORS = {
    "yellowing_chlorosis": (234, 179, 8),    # Yellow
    "crispy_brown_edges": (217, 119, 6),     # Brown/Amber
    "dark_spots_necrosis": (220, 38, 38),    # Crimson Red
    "white_powdery_residue": (6, 182, 212),  # Cyan
    "chewed_holes_stippling": (147, 51, 234),# Purple
    "healthy_leaf": (16, 185, 129),          # Emerald Green
}


def main():
    parser = argparse.ArgumentParser(description="UrbanGreen CoreTech: Standalone Symptom Inference Tester")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained YOLO11 symptom weights (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to test plant image")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold (default: 0.50)")
    parser.add_argument("--output", type=str, default="", help="Path to save annotated image (optional)")
    args = parser.parse_args()

    weights_p = Path(args.weights).resolve()
    img_p = Path(args.image).resolve()

    if not weights_p.exists():
        print(f"[FATAL] Weights file does not exist: {weights_p}")
        sys.exit(1)
    if not img_p.exists():
        print(f"[FATAL] Image file does not exist: {img_p}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[FATAL] 'ultralytics' library not installed. Install via: pip install ultralytics")
        sys.exit(1)

    print("=" * 70)
    print("  UrbanGreen PhytoCare CoreTech: Standalone Symptom Detector Test")
    print(f"  Model Weights : {weights_p.name} ({weights_p})")
    print(f"  Input Image   : {img_p.name}")
    print(f"  Confidence    : {args.conf}")
    print("=" * 70)

    # 1. Load Model & Validate 6-Class Contract
    model = YOLO(str(weights_p))
    model_names = model.names
    print(f"[*] Verifying model class contract ({len(model_names)} classes registered):")
    contract_ok = True
    for cid, exp_name in TARGET_CLASSES.items():
        actual_name = model_names.get(cid, None)
        status = "MATCH" if actual_name == exp_name else f"MISMATCH (got '{actual_name}')"
        print(f"    [{cid}] {exp_name:<24} : {status}")
        if actual_name != exp_name:
            contract_ok = False

    if not contract_ok:
        print("[!] WARNING: Model class names do not match the expected 6-class contract!")
    else:
        print("[✓] Model class names strictly adhere to 6-class contract.")

    # 2. Run Inference
    results = model.predict(source=str(img_p), conf=args.conf, iou=args.iou, verbose=False)
    raw_img = Image.open(img_p).convert("RGB")
    annotated = raw_img.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = raw_img.size

    detections = []
    if results and len(results) > 0:
        r = results[0]
        for box in r.boxes:
            cid = int(box.cls.item())
            conf_val = float(box.conf.item())
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            cname = model_names.get(cid, TARGET_CLASSES.get(cid, f"class_{cid}"))
            detections.append({
                "class_id": cid,
                "class_name": cname,
                "confidence": conf_val,
                "bbox": xyxy
            })

    print(f"\n[*] RAW DETECTIONS ({len(detections)} found):")
    if not detections:
        print("    [None] No symptoms detected above confidence threshold.")
    else:
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d["bbox"]
            bw, bh = x2 - x1, y2 - y1
            area_pct = (bw * bh) / (w * h) * 100.0
            print(f"    [{i+1}] {d['class_name']:<24} conf={d['confidence']:.3f} | bbox=[{x1}, {y1}, {x2}, {y2}] ({area_pct:.1f}% img area)")

            col = CLASS_COLORS.get(d["class_name"], (220, 38, 38))
            draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
            lbl = f"{d['class_name']} ({int(d['confidence']*100)}%)"
            draw.rectangle([x1, max(0, y1 - 20), x1 + len(lbl) * 8, y1], fill=col)
            draw.text((x1 + 4, max(0, y1 - 18)), lbl, fill=(255, 255, 255))

    # 3. Save Annotated Preview
    out_path = Path(args.output).resolve() if args.output else img_p.parent / f"test_annotated_{img_p.stem}.jpg"
    annotated.save(out_path, format="JPEG", quality=90)
    print(f"\n[✓] Saved annotated preview image to:\n    {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
