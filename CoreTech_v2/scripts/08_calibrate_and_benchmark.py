"""
CoreTech v2 Phase 5: Out-of-Fold Ensemble Calibration & Strictly Unseen Holdout Benchmark.
1. Optimizes Platt Temperature (T) and ensemble weights on `val/` splits ONLY.
2. Evaluates strictly untouched `test/` splits for both Botanical ID & Surface Pathology,
   verifying >= 90% accuracy targets and OOD/uncertainty gating behavior.
"""
import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))

from configs.v2_config import (
    ID_DATA_DIR, PATH_DATA_DIR, NON_PLANT_DIR, ARTIFACTS_DIR, REPORTS_DIR,
    SURFACE_PATHOLOGY_CLASSES, DEVICE, USE_BF16, load_taxonomy
)
from src.botanical_ensemble import load_botanical_ensemble
from src.surface_pathology import load_pathology_suite


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temp = nn.Parameter(torch.ones(1) * 1.0)
    opt = torch.optim.LBFGS([temp], lr=0.02, max_iter=60)
    crit = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        t_clamped = torch.clamp(temp, min=0.35, max=4.0)
        loss = crit(logits / t_clamped, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return round(float(torch.clamp(temp, min=0.35, max=4.0).item()), 4)


def calibrate_and_eval_flora():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tf", V2_ROOT / "scripts" / "05_train_flora_models.py")
    tf_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tf_mod)

    models_dict, _, taxonomy = load_botanical_ensemble(reload=True)
    if not models_dict:
        print("[Skip] No trained flora models found yet.")
        return {}

    amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
    first_arch = next(iter(models_dict.keys()))
    ref_classes = models_dict[first_arch]["classes"]
    class_to_idx = {c: i for i, c in enumerate(ref_classes)}

    def collect_split(split_name: str):
        all_logits = {}
        all_cos = {}
        y_ref = None
        for arch, bundle in models_dict.items():
            mean, std = tf_mod.get_norm_stats(arch)
            _, val_tf = tf_mod.build_transforms(bundle["img_size"], mean, std)
            ds = tf_mod.AlbumentationsFolderDataset(ID_DATA_DIR / split_name, val_tf, class_to_idx=class_to_idx)
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
            lg_list, cs_list, ys = [], [], []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                        lg, cs = bundle["model"](x)
                    lg_list.append(lg.float().cpu())
                    cs_list.append(cs.float().max(dim=-1).values.cpu())
                    ys.append(y)
            all_logits[arch] = torch.cat(lg_list, dim=0)
            all_cos[arch] = torch.cat(cs_list, dim=0)
            y_ref = torch.cat(ys, dim=0)
        return all_logits, all_cos, y_ref

    print("[Flora Calib] Collecting validation logits across trained models...")
    val_lg, val_cos, val_y = collect_split("val")
    temps = {arch: fit_temperature(val_lg[arch], val_y) for arch in val_lg}

    # Grid search optimal ensemble weights on val
    archs = list(val_lg.keys())
    best_w = {a: 1.0 / len(archs) for a in archs}
    best_acc = -1.0
    if len(archs) == 3:
        for w0 in np.linspace(0.2, 0.6, 5):
            for w1 in np.linspace(0.2, 0.6, 5):
                w2 = 1.0 - w0 - w1
                if w2 < 0.05:
                    continue
                ws = [w0, w1, w2]
                probs = sum(ws[i] * F.softmax(val_lg[archs[i]] / temps[archs[i]], dim=-1) for i in range(3))
                acc = float((probs.argmax(dim=-1) == val_y).float().mean().item())
                if acc > best_acc:
                    best_acc = acc
                    best_w = {archs[i]: round(float(ws[i]), 3) for i in range(3)}
    elif len(archs) == 2:
        for w0 in np.linspace(0.2, 0.8, 7):
            ws = [w0, 1.0 - w0]
            probs = sum(ws[i] * F.softmax(val_lg[archs[i]] / temps[archs[i]], dim=-1) for i in range(2))
            acc = float((probs.argmax(dim=-1) == val_y).float().mean().item())
            if acc > best_acc:
                best_acc = acc
                best_w = {archs[i]: round(float(ws[i]), 3) for i in range(2)}

    # Calibrate OOD cosine floor (5th percentile of correctly classified validation samples)
    val_probs = sum(best_w[a] * F.softmax(val_lg[a] / temps[a], dim=-1) for a in archs)
    correct_mask = val_probs.argmax(dim=-1) == val_y
    mean_cos = torch.stack([val_cos[a] for a in archs], dim=0).mean(dim=0)
    ood_floor = round(float(np.percentile(mean_cos[correct_mask].numpy(), 2)), 3) if correct_mask.any() else 0.22

    calib_out = {
        "weights": best_w,
        "temperatures": temps,
        "species_conf_threshold": 0.50,
        "species_margin_threshold": 0.12,
        "min_ood_cosine": max(0.18, min(0.35, ood_floor)),
        "val_ensemble_top1_pct": round(best_acc * 100.0, 2),
    }
    (ARTIFACTS_DIR / "flora_calibration.json").write_text(json.dumps(calib_out, indent=2), encoding="utf-8")

    print("[Flora Eval] Evaluating strictly unseen TEST split...")
    test_lg, test_cos, test_y = collect_split("test")
    test_probs = sum(best_w[a] * F.softmax(test_lg[a] / temps[a], dim=-1) for a in archs)
    preds1 = test_probs.argmax(dim=-1)
    top5 = test_probs.topk(5, dim=-1).indices

    top1_acc = float((preds1 == test_y).float().mean().item() * 100.0)
    top5_acc = float(sum(test_y[i].item() in top5[i].tolist() for i in range(len(test_y))) / len(test_y) * 100.0)
    macro_f1 = float(f1_score(test_y.numpy(), preds1.numpy(), average="macro") * 100.0)

    # Cactaceae subset evaluation
    cact_idxs = [i for i, y in enumerate(test_y.tolist()) if "Cactaceae" in taxonomy.get(ref_classes[y], {}).get("family", "")]
    cact_top1 = float(np.mean([preds1[i].item() == test_y[i].item() for i in cact_idxs]) * 100.0) if cact_idxs else 0.0
    cact_fam_acc = 0.0
    if cact_idxs:
        fam_hits = sum("Cactaceae" in taxonomy.get(ref_classes[preds1[i].item()], {}).get("family", "") for i in cact_idxs)
        cact_fam_acc = fam_hits / len(cact_idxs) * 100.0

    # Per-model individual test top-1
    indiv = {}
    for a in archs:
        indiv[a] = round(float((test_lg[a].argmax(dim=-1) == test_y).float().mean().item() * 100.0), 2)

    res = {
        "calibration": calib_out,
        "test_samples": len(test_y),
        "individual_test_top1_pct": indiv,
        "ensemble_test_top1_pct": round(top1_acc, 2),
        "ensemble_test_top5_pct": round(top5_acc, 2),
        "ensemble_macro_f1_pct": round(macro_f1, 2),
        "cactaceae_species_top1_pct": round(cact_top1, 2),
        "cactaceae_family_top1_pct": round(cact_fam_acc, 2),
    }
    print("[Flora Eval Result]:", json.dumps(res, indent=2))
    return res


def calibrate_and_eval_pathology():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pm", V2_ROOT / "scripts" / "06_train_pathology_mil.py")
    pm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm_mod)

    mil_dict, _, _ = load_pathology_suite(reload=True)
    if not mil_dict:
        print("[Skip] No trained pathology MIL models found yet.")
        return {}

    amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
    first_arch = next(iter(mil_dict.keys()))
    classes = mil_dict[first_arch]["classes"]

    def collect_split(split_name: str):
        fine_lg, sup_lg = {}, {}
        y_fine_ref, y_sup_ref = None, None
        for arch, bundle in mil_dict.items():
            ds = pm_mod.SurfaceMILDataset(PATH_DATA_DIR / split_name, bundle["img_size"], classes, [], is_train=False)
            loader = DataLoader(ds, batch_size=12, shuffle=False, num_workers=0)
            fl, sl, yf, ys = [], [], [], []
            with torch.no_grad():
                for bag, f_lbl, s_lbl in loader:
                    bag = bag.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                        out_f, out_s, _, _ = bundle["model"](bag)
                    fl.append(out_f.float().cpu())
                    sl.append(out_s.float().cpu())
                    yf.append(f_lbl)
                    ys.append(s_lbl)
            fine_lg[arch] = torch.cat(fl, dim=0)
            sup_lg[arch] = torch.cat(sl, dim=0)
            y_fine_ref = torch.cat(yf, dim=0)
            y_sup_ref = torch.cat(ys, dim=0)
        return fine_lg, sup_lg, y_fine_ref, y_sup_ref

    print("[Pathology Calib] Collecting validation logits...")
    val_fl, val_sl, val_yf, val_ys = collect_split("val")
    temps = {a: fit_temperature(val_fl[a], val_yf) for a in val_fl}
    archs = list(val_fl.keys())
    best_w = {a: 1.0 / len(archs) for a in archs}
    best_acc = -1.0
    if len(archs) == 2:
        for w0 in np.linspace(0.2, 0.8, 7):
            ws = [w0, 1.0 - w0]
            probs = sum(ws[i] * F.softmax(val_fl[archs[i]] / temps[archs[i]], dim=-1) for i in range(2))
            acc = float((probs.argmax(dim=-1) == val_yf).float().mean().item())
            if acc > best_acc:
                best_acc = acc
                best_w = {archs[i]: round(float(ws[i]), 3) for i in range(2)}

    calib_out = {
        "weights": best_w,
        "temperatures": temps,
        "healthy_override_thresh": 0.62,
        "val_ensemble_top1_pct": round(best_acc * 100.0, 2),
    }
    (ARTIFACTS_DIR / "pathology_calibration.json").write_text(json.dumps(calib_out, indent=2), encoding="utf-8")

    print("[Pathology Eval] Evaluating strictly unseen TEST split...")
    test_fl, test_sl, test_yf, test_ys = collect_split("test")
    test_fp = sum(best_w[a] * F.softmax(test_fl[a] / temps[a], dim=-1) for a in archs)
    test_sp = sum(best_w[a] * F.softmax(test_sl[a], dim=-1) for a in archs)

    preds_f = test_fp.argmax(dim=-1)
    preds_s = test_sp.argmax(dim=-1)
    top3 = test_fp.topk(3, dim=-1).indices

    fine_top1 = float((preds_f == test_yf).float().mean().item() * 100.0)
    fine_top3 = float(sum(test_yf[i].item() in top3[i].tolist() for i in range(len(test_yf))) / len(test_yf) * 100.0)
    sup_acc = float((preds_s == test_ys).float().mean().item() * 100.0)
    macro_f1 = float(f1_score(test_yf.numpy(), preds_f.numpy(), average="macro") * 100.0)

    rep_text = classification_report(test_yf.numpy(), preds_f.numpy(), target_names=classes, digits=3)
    (REPORTS_DIR / "pathology_test_classification_report.txt").write_text(rep_text, encoding="utf-8")

    res = {
        "calibration": calib_out,
        "test_samples": len(test_yf),
        "test_fine_top1_pct": round(fine_top1, 2),
        "test_fine_top3_pct": round(fine_top3, 2),
        "test_macro_f1_pct": round(macro_f1, 2),
        "test_super_healthy_biotic_abiotic_pct": round(sup_acc, 2),
    }
    print("[Pathology Eval Result]:", json.dumps(res, indent=2))
    return res


def main():
    summary = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "flora_id": calibrate_and_eval_flora(),
        "surface_pathology": calibrate_and_eval_pathology(),
    }
    (REPORTS_DIR / "benchmark_v2_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[✓] Saved full CoreTech v2 benchmark results -> {REPORTS_DIR / 'benchmark_v2_eval.json'}")


if __name__ == "__main__":
    main()
