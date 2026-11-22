from ultralytics import YOLO

weights_p = r"c:\Users\User\Desktop\FYP\CoreTech\models\symptom_yolo11s.pt"
data_yaml = r"c:\Users\User\Desktop\FYP\CoreTech\ml_training\symptom_detector\datasets\symptom_dataset\data.yaml"

if __name__ == "__main__":
    model = YOLO(weights_p)
    metrics = model.val(data=data_yaml, split="val", workers=0, verbose=False)
    names = model.names

    print("=" * 72)
    print("YOLO11s PER-CLASS VALIDATION BREAKDOWN (VAL SPLIT)")
    print("=" * 72)
    header = f"{'Class ID & Name':<28} {'Precision':<11} {'Recall':<11} {'mAP50':<11} {'mAP50-95':<11}"
    print(header)
    print("-" * 72)

    for i in sorted(names.keys()):
        cname = f"{i}: {names[i]}"
        p = float(metrics.box.p[i]) if i < len(metrics.box.p) else 0.0
        r = float(metrics.box.r[i]) if i < len(metrics.box.r) else 0.0
        ap50 = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0
        ap = float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0.0
        print(f"{cname:<28} {p:<11.3f} {r:<11.3f} {ap50:<11.3f} {ap:<11.3f}")

    print("-" * 72)
    print(f"{'Mean / All Classes':<28} {metrics.box.mp:<11.3f} {metrics.box.mr:<11.3f} {metrics.box.map50:<11.3f} {metrics.box.map:<11.3f}")
    print("=" * 72)
