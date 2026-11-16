"""
Stage 2: Foundation Botanical Identification Ensemble & Calibrated Uncertainty Gate.
Combines heterogeneous cosine-normalized models (BioCLIP-2 + EVA-02-448 + ConvNeXt-Base-384)
with multi-view TTA, calibrated ensemble temperatures, OOD cosine similarity gating,
and hierarchical botanical Family/Genus fallback + actionable angle recommendations.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import sys
V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))
from configs.v2_config import (
    ARTIFACTS_DIR, DEVICE, USE_BF16, SPECIES_CONF_THRESHOLD, SPECIES_MARGIN_THRESHOLD, load_taxonomy
)
try:
    from .cosine_models import BioCLIP2Classifier, TimmCosineClassifier
except (ImportError, ValueError):
    from src.cosine_models import BioCLIP2Classifier, TimmCosineClassifier

_cached_models = None
_cached_calib = None
_cached_taxonomy = None


def _eval_views_tensor(pil_img: Image.Image, img_size: int, mean, std) -> torch.Tensor:
    """Builds 3 complementary views (Full Aspect-Padded, Tight Center, Horizontal Flip) for robust TTA."""
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    # View 1: Aspect-preserving pad to square (never chops cactus spines/tips!)
    max_s = max(w, h)
    pad_l = (max_s - w) // 2
    pad_t = (max_s - h) // 2
    sq = Image.new("RGB", (max_s, max_s), (128, 128, 128))
    sq.paste(rgb, (pad_l, pad_t))
    v1 = sq.resize((img_size, img_size), Image.Resampling.BILINEAR)

    # View 2: Horizontal mirror of v1
    v2 = v1.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # View 3: Center 82% zoom crop
    min_s = min(w, h)
    c_sz = max(64, int(min_s * 0.85))
    cx, cy = w // 2, h // 2
    x0, y0 = max(0, cx - c_sz // 2), max(0, cy - c_sz // 2)
    v3 = rgb.crop((x0, y0, min(w, x0 + c_sz), min(h, y0 + c_sz))).resize((img_size, img_size), Image.Resampling.BILINEAR)

    tensors = []
    for v in (v1, v2, v3):
        t = TF.to_tensor(v)
        t = TF.normalize(t, mean=mean, std=std)
        tensors.append(t)
    return torch.stack(tensors, dim=0)


def load_botanical_ensemble(reload: bool = False):
    global _cached_models, _cached_calib, _cached_taxonomy
    if _cached_models is not None and not reload:
        return _cached_models, _cached_calib, _cached_taxonomy

    _cached_taxonomy = load_taxonomy()
    calib_path = ARTIFACTS_DIR / "flora_calibration.json"
    if calib_path.exists():
        _cached_calib = json.loads(calib_path.read_text(encoding="utf-8"))
    else:
        _cached_calib = {
            "weights": {"bioclip2": 0.40, "eva02": 0.35, "convnext": 0.25},
            "temperatures": {"bioclip2": 1.0, "eva02": 1.0, "convnext": 1.0},
            "species_conf_threshold": SPECIES_CONF_THRESHOLD,
            "species_margin_threshold": SPECIES_MARGIN_THRESHOLD,
            "min_ood_cosine": 0.22,
        }

    models_dict = {}
    specs = [
        ("bioclip2", "flora_bioclip2.pth", (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ("eva02", "flora_eva02.pth", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ("convnext", "flora_convnext.pth", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]

    for arch, ckpt_name, mean, std in specs:
        p = ARTIFACTS_DIR / ckpt_name
        if not p.exists():
            continue
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        classes = ckpt["classes"]
        img_size = ckpt.get("img_size", 384)
        if arch == "bioclip2":
            m = BioCLIP2Classifier(num_classes=len(classes), unfreeze_last_blocks=0)
        elif arch == "eva02":
            m = TimmCosineClassifier("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", num_classes=len(classes), pretrained=False)
        else:
            m = TimmCosineClassifier("convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384", num_classes=len(classes), pretrained=False)
        m.load_state_dict(ckpt["model_state_dict"], strict=True)
        m = m.to(DEVICE).eval()
        models_dict[arch] = {
            "model": m,
            "classes": classes,
            "img_size": img_size,
            "mean": mean,
            "std": std,
        }

    _cached_models = models_dict
    return _cached_models, _cached_calib, _cached_taxonomy


def identify_dominant_plant(pil_crop: Image.Image, top_k: int = 5) -> Dict[str, Any]:
    """
    Identifies the Stage 1B isolated plant specimen using the calibrated foundation ensemble.
    Provides graceful Genus/Family fallback and capture advice when confidence/margin is low.
    """
    models_dict, calib, taxonomy = load_botanical_ensemble()
    if not models_dict:
        return {
            "status": "NO_MODEL",
            "species": "Unidentified Botanical Specimen",
            "family": "Unknown",
            "confidence": 0.0,
            "is_confident_species": False,
            "candidates": [],
            "recommendations": ["Botanical identification models are not yet trained or loaded."],
        }

    weights_cfg = calib.get("weights", {})
    temps_cfg = calib.get("temperatures", {})
    amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16

    blended_probs = None
    max_cos_scores = []
    ref_classes = None
    total_w = 0.0

    with torch.no_grad():
        for arch, bundle in models_dict.items():
            w = float(weights_cfg.get(arch, 1.0))
            T = max(0.3, float(temps_cfg.get(arch, 1.0)))
            x = _eval_views_tensor(pil_crop, bundle["img_size"], bundle["mean"], bundle["std"]).to(DEVICE)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits, cos_sim = bundle["model"](x)
            probs = F.softmax(logits.float() / T, dim=-1).mean(dim=0)
            top_cos = float(cos_sim.float().mean(dim=0).max().item())
            max_cos_scores.append(top_cos)

            if blended_probs is None:
                blended_probs = w * probs
                ref_classes = bundle["classes"]
            else:
                blended_probs = blended_probs + w * probs
            total_w += w

    blended_probs = blended_probs / max(1e-6, total_w)
    top_p, top_i = blended_probs.topk(min(top_k, len(ref_classes)))

    candidates = []
    for r, (p, idx) in enumerate(zip(top_p, top_i), 1):
        cname = ref_classes[idx.item()]
        fam = taxonomy.get(cname, {}).get("family", "Other Botanical Taxa")
        candidates.append({
            "rank": r,
            "species": cname,
            "family": fam,
            "confidence": round(p.item() * 100.0, 2),
        })

    # Family-level probability aggregation for hierarchical fallback
    fam_probs = {}
    for idx in range(len(ref_classes)):
        cname = ref_classes[idx]
        fam = taxonomy.get(cname, {}).get("family", f"Other ({cname})")
        fam_probs[fam] = fam_probs.get(fam, 0.0) + float(blended_probs[idx].item())

    top_fam, top_fam_p = max(fam_probs.items(), key=lambda x: x[1])
    top1_prob = float(top_p[0].item())
    top2_prob = float(top_p[1].item()) if len(top_p) > 1 else 0.0
    margin = top1_prob - top2_prob
    mean_ood_cos = float(np.mean(max_cos_scores)) if max_cos_scores else 0.0

    conf_thresh = float(calib.get("species_conf_threshold", SPECIES_CONF_THRESHOLD))
    margin_thresh = float(calib.get("species_margin_threshold", SPECIES_MARGIN_THRESHOLD))
    ood_thresh = float(calib.get("min_ood_cosine", 0.22))

    is_ood = mean_ood_cos < ood_thresh
    is_confident = (not is_ood) and (top1_prob >= conf_thresh or (top1_prob >= 0.45 and margin >= margin_thresh))

    top_species = candidates[0]["species"]
    top_species_fam = candidates[0]["family"]

    if is_confident:
        status = "CONFIDENT_SPECIES"
        display_name = top_species
        display_conf = round(top1_prob * 100.0, 2)
        recommendations = []
    elif not is_ood and top_fam_p >= 0.65 and not top_fam.startswith("Other"):
        status = "GENUS_FAMILY_FALLBACK"
        fam_short = top_fam.split(" (")[0]
        display_name = f"{fam_short} (Likely {top_species})"
        display_conf = round(top_fam_p * 100.0, 2)
        recommendations = [
            f"Confirmed plant family ({top_fam}, {display_conf:.0f}%), though exact cultivar is closely split among {candidates[0]['species']} and {candidates[1]['species']}.",
            "To pinpoint exact species, take a clearer side-profile showing leaf stem attachment and overall plant structure."
        ]
    else:
        status = "LOW_CONFIDENCE_SPECIES"
        display_name = f"Unlisted or Ambiguous Plant (Closest: {top_species})"
        display_conf = round(top1_prob * 100.0, 2)
        recommendations = [
            "Could not confidently match this exact species against our curated botanical library.",
            "Try capturing both a full-plant silhouette and a crisp close-up of a single mature leaf/stem in bright diffuse daylight."
        ]

    return {
        "status": status,
        "species": display_name,
        "raw_top_species": top_species,
        "family": top_fam if status == "GENUS_FAMILY_FALLBACK" else top_species_fam,
        "family_confidence": round(top_fam_p * 100.0, 2),
        "confidence": display_conf,
        "species_confidence": round(top1_prob * 100.0, 2),
        "margin_pct": round(margin * 100.0, 2),
        "ood_cosine": round(mean_ood_cos, 3),
        "is_confident_species": is_confident,
        "candidates": candidates,
        "recommendations": recommendations,
    }
