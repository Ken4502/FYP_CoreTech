"""
Interactive and batch testing script for FloraNet-100 model.
Allows testing single images or benchmarking random test samples across 100 classes.
"""

import sys
import argparse
import random
from pathlib import Path

# Configure utf-8 encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_image, load_inference_model

def test_random_samples(num_samples: int = 12):
    test_dir = PROJECT_ROOT / "data" / "test"
    all_classes = [d.name for d in test_dir.iterdir() if d.is_dir()]
    
    # Pick a balanced mix of original classes and newly introduced classes
    selected_classes = random.sample(all_classes, min(num_samples, len(all_classes)))
    
    print("=" * 70)
    print(f"🌿 FLORANET-100 INFERENCE TEST ({len(selected_classes)} Diverse Test Samples)")
    print("=" * 70)
    
    correct = 0
    for i, cls in enumerate(selected_classes, 1):
        cls_folder = test_dir / cls
        images = list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.png")) + list(cls_folder.glob("*.jpeg"))
        if not images:
            continue
        sample_img = random.choice(images)
        
        predictions = predict_image(sample_img, top_k=5)
        top1 = predictions[0]
        is_match = (top1["species"] == cls)
        in_top5 = any(p["species"] == cls for p in predictions)
        
        if is_match:
            correct += 1
            mark = "✅ MATCH"
        elif in_top5:
            mark = "🟡 IN TOP-5"
        else:
            mark = "❌ MISS"
            
        print(f"\n[{i}/{len(selected_classes)}] {mark}")
        print(f"  • Ground Truth: {cls}")
        print(f"  • Sample File:  {sample_img.name}")
        print(f"  • Top-1 Pred:   {top1['species']} ({top1['confidence']}%)")
        print(f"  • Top-5 Ranked:")
        for p in predictions:
            check = "👉" if p["species"] == cls else "  "
            print(f"      {check} [{p['rank']}] {p['species']}: {p['confidence']}%")
            
    accuracy = (correct / len(selected_classes)) * 100
    print("\n" + "=" * 70)
    print(f"🏆 TEST RESULT: {correct}/{len(selected_classes)} Top-1 Matches ({accuracy:.1f}%)")
    print("=" * 70)

def test_single_image(image_path: str):
    p = Path(image_path)
    if not p.exists():
        print(f"Error: Image file not found at '{image_path}'")
        return
        
    print("=" * 70)
    print(f"🔍 TESTING SINGLE IMAGE: {p.name}")
    print("=" * 70)
    
    predictions = predict_image(p, top_k=5)
    print(f"\nTop-5 Predictions:")
    for res in predictions:
        print(f"  [{res['rank']}] {res['species']}: {res['confidence']}%")
def test_targeted():
    test_dir = PROJECT_ROOT / "data" / "test"
    target_classes = [
        "Watermelon Peperomia (Peperomia argyreia)",
        "Wax Plant (Hoya carnosa)",
        "Calathea",
        "Pothos (Ivy arum)",
        "Snake plant (Sanseviera)",
        "Cast Iron Plant (Aspidistra elatior)",
        "Fiddle Leaf Fig (Ficus lyrata)",
        "Croton (Codiaeum variegatum)",
        "Calathea Medallion (Goeppertia roseopicta)",
        "Monstera Deliciosa (Monstera deliciosa)"
    ]
    
    print("=" * 70)
    print("🎯 TARGETED VERIFICATION: KEY EXPANSION & TURNAROUND SPECIES")
    print("=" * 70)
    
    correct = 0
    for i, cls in enumerate(target_classes, 1):
        cls_folder = test_dir / cls
        images = list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.png")) + list(cls_folder.glob("*.jpeg"))
        if not images:
            print(f"[{i}] Missing test images for '{cls}'")
            continue
        sample_img = images[0]
        
        predictions = predict_image(sample_img, top_k=5)
        top1 = predictions[0]
        is_match = (top1["species"] == cls)
        in_top5 = any(p["species"] == cls for p in predictions)
        
        if is_match:
            correct += 1
            mark = "✅ MATCH"
        elif in_top5:
            mark = "🟡 IN TOP-5"
        else:
            mark = "❌ MISS"
            
        print(f"\n[{i}/{len(target_classes)}] {mark}")
        print(f"  • Ground Truth: {cls}")
        print(f"  • Top-1 Pred:   {top1['species']} ({top1['confidence']}%)")
        if not is_match:
            print(f"  • Top-3:")
            for p in predictions[:3]:
                print(f"      [{p['rank']}] {p['species']}: {p['confidence']}%")
                
    accuracy = (correct / len(target_classes)) * 100
    print("\n" + "=" * 70)
    print(f"🏆 TARGETED ACCURACY: {correct}/{len(target_classes)} ({accuracy:.1f}%)")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Test FloraNet-100 Botanical Classifier")
    parser.add_argument("--image", type=str, default=None, help="Path to single image file to predict")
    parser.add_argument("--num", type=int, default=12, help="Number of random test samples to benchmark")
    parser.add_argument("--targeted", action="store_true", help="Run targeted test on key turnaround & new species")
    args = parser.parse_args()
    
    if args.image:
        test_single_image(args.image)
    elif args.targeted:
        test_targeted()
    else:
        test_random_samples(num_samples=args.num)

if __name__ == "__main__":
    main()
