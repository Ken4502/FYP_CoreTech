"""
Stage 2: Foundation Botanical Identification Ensemble & Calibrated Uncertainty Gate.
Combines heterogeneous cosine-normalized models (BioCLIP-2 + EVA-02-448 + ConvNeXt-Base-384)
with multi-view TTA, calibrated ensemble temperatures, OOD cosine similarity gating,
and hierarchical botanical Family/Genus fallback + actionable angle recommendations.
"""
import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import sys
V2_ROOT = Path(__file__).resolve().parent.parent
CORETECH_ROOT = V2_ROOT.parent
if str(CORETECH_ROOT) not in sys.path:
    sys.path.append(str(CORETECH_ROOT))
if str(V2_ROOT) not in sys.path:
    sys.path.append(str(V2_ROOT))
try:
    from configs.v2_config import (
        ARTIFACTS_DIR, DEVICE, USE_BF16, SPECIES_CONF_THRESHOLD, SPECIES_MARGIN_THRESHOLD, load_taxonomy
    )
except (ImportError, ValueError):
    from CoreTech_v2.configs.v2_config import (
        ARTIFACTS_DIR, DEVICE, USE_BF16, SPECIES_CONF_THRESHOLD, SPECIES_MARGIN_THRESHOLD, load_taxonomy
    )

try:
    from .cosine_models import BioCLIP2Classifier, TimmCosineClassifier
except (ImportError, ValueError):
    try:
        from CoreTech_v2.src.cosine_models import BioCLIP2Classifier, TimmCosineClassifier
    except (ImportError, ValueError):
        from src.cosine_models import BioCLIP2Classifier, TimmCosineClassifier

_cached_models = None
_cached_calib = None
_cached_taxonomy = None
_cached_floranet_261 = None

SPECIES_VISUAL_PRIOR_MATRIX = {
    # Variegated taxa (distinct patterns/patches of white, cream, yellow, pink)
    "variegated_taxa": [
        "pothos", "epipremnum", "snake plant", "sansevieria", "calathea", "aglaonema", "chinese evergreen",
        "coleus", "croton", "codiaeum", "begonia", "ctenanthe", "tradescantia", "stromanthe", "cordyline",
        "syngonium", "maranta", "prayer plant", "fittonia", "nerve plant", "hypoestes", "polka dot",
        "philodendron pink princess", "philodendron birkin", "dieffenbachia", "spider plant", "chlorophytum"
    ],
    # Fenestrated taxa (natural leaf splits, holes, perforations)
    "fenestrated_taxa": [
        "monstera", "monstera deliciosa", "monstera adansonii", "swiss cheese", "rhaphidophora tetrasperma"
    ],
    # Glossy / waxy cuticle taxa (thick reflective epidermal layer)
    "glossy_taxa": [
        "ficus", "rubber plant", "rubber tree", "fiddle leaf", "lyrata", "elastica", "benjamina",
        "zz plant", "zamioculcas", "peace lily", "spathiphyllum", "anthurium", "citrus", "croton",
        "coffee", "coffea", "hoya", "schefflera", "dracaena"
    ],
    # Succulent taxa (thick fleshy water-storage leaves with farina / waxy coating)
    "succulent_taxa": [
        "aloe", "aloe vera", "echeveria", "cactus", "cacti", "cactaceae", "crassula", "crassulaceae",
        "jade plant", "kalanchoe", "sedum", "haworthia", "sansevieria", "snake plant", "senecio",
        "string of pearls", "agave", "euphorbia", "desert rose", "adenium"
    ]
}


def get_species_visual_priors(species_name: str, family: str = "") -> Dict[str, bool]:
    tax_str = f"{species_name} {family}".lower()
    return {
        "is_variegated": any(k in tax_str for k in SPECIES_VISUAL_PRIOR_MATRIX["variegated_taxa"]),
        "is_fenestrated": any(k in tax_str for k in SPECIES_VISUAL_PRIOR_MATRIX["fenestrated_taxa"]),
        "is_glossy": any(k in tax_str for k in SPECIES_VISUAL_PRIOR_MATRIX["glossy_taxa"]),
        "is_succulent": any(k in tax_str for k in SPECIES_VISUAL_PRIOR_MATRIX["succulent_taxa"]),
    }


def extract_genus_from_species(species_name: str) -> str:
    """Extracts the botanical genus from a species name or common name."""
    if not species_name or not isinstance(species_name, str):
        return "Unknown"
    s = species_name.strip()
    # Format: 'Common Name (Genus species)' e.g. 'Snake Plant (Sansevieria trifasciata)'
    if "(" in s and ")" in s:
        inner = s[s.find("(") + 1 : s.find(")")].strip()
        parts = inner.replace("_", " ").split()
        if parts:
            return parts[0].strip().capitalize()
    s_clean = s.replace("___", " ").replace("__", " ").replace("_", " ")
    parts = s_clean.split()
    if parts:
        return parts[0].strip().capitalize()
    return "Unknown"


def load_floranet_261_model():
    global _cached_floranet_261
    if _cached_floranet_261 is not None:
        return _cached_floranet_261

    ckpt_paths = [
        CORETECH_ROOT / "Plant_Identification_Experiment" / "artifacts" / "best_model.pth",
        V2_ROOT.parent.parent / "Plant_Identification_Experiment" / "artifacts" / "best_model.pth",
    ]
    ckpt_path = next((p for p in ckpt_paths if p.exists()), None)
    if not ckpt_path:
        return None, None, 256

    try:
        import torchvision.models as tv_models
        import torch.nn as nn

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        classes = ckpt["classes"]
        resolution = ckpt.get("resolution", 256)

        m = tv_models.convnext_small(weights=None)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_f, len(classes))
        )
        m.load_state_dict(ckpt["model_state_dict"])
        m = m.to(DEVICE).eval()
        _cached_floranet_261 = (m, classes, resolution)
        return _cached_floranet_261
    except Exception as e:
        print(f"[Warning] Failed to load FloraNet-261 backbone: {e}")
        return None, None, 256


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


def identify_dominant_plant(pil_crop: Union[Image.Image, List[Image.Image]], top_k: int = 5) -> Dict[str, Any]:
    """
    Identifies the Stage 1B isolated plant specimen using the calibrated foundation ensemble.
    Supports single or multi-photo inputs (e.g. whole plant + close-up) with pre-softmax logit averaging.
    Provides Top-5 candidate ranking, genus-level grouping, and visual priors.
    """
    if isinstance(pil_crop, (list, tuple)):
        crops_list = [c for c in pil_crop if c is not None]
        if not crops_list:
            raise ValueError("At least one valid PIL Image crop must be provided.")
    else:
        crops_list = [pil_crop]

    models_dict, calib, taxonomy = load_botanical_ensemble()
    if not models_dict:
        fn_model, fn_classes, fn_res = load_floranet_261_model()
        if fn_model is not None:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            crop_logits_list = []
            crop_cos_list = []

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    for crop in crops_list:
                        x = _eval_views_tensor(crop, fn_res, mean, std).to(DEVICE)
                        feats = fn_model.features(x)
                        feats = fn_model.avgpool(feats)
                        feats = fn_model.classifier[0](feats)
                        feats = fn_model.classifier[1](feats)

                        feats_norm = F.normalize(feats, p=2, dim=-1)
                        W = fn_model.classifier[2][1].weight
                        W_norm = F.normalize(W, p=2, dim=-1)

                        cos_sim = F.linear(feats_norm, W_norm)
                        crop_cos_list.append(cos_sim.mean(dim=0))
                        # Temperature-scaled cosine logits (s = 16.0) averaged over TTA views
                        crop_logits_list.append((cos_sim * 16.0).mean(dim=0))

                # Multi-photo logit averaging: z_final = (z_1 + ... + z_N) / N
                avg_logits = torch.stack(crop_logits_list, dim=0).mean(dim=0)
                mean_cos = torch.stack(crop_cos_list, dim=0).mean(dim=0)
                max_cos_val = float(mean_cos.max().item())
                probs = F.softmax(avg_logits.float(), dim=-1)

            top_p, top_i = probs.topk(min(top_k * 2, len(fn_classes)))
            candidates = []
            for r, (p, idx) in enumerate(zip(top_p, top_i), 1):
                cname = fn_classes[idx.item()]
                fam = taxonomy.get(cname, {}).get("family", "Other Botanical Taxa")
                genus = extract_genus_from_species(cname)
                candidates.append({
                    "rank": r,
                    "species": cname,
                    "species_name": cname,
                    "genus": genus,
                    "family": fam,
                    "confidence": round(p.item() * 100.0, 2),
                    "cosine": round(float(mean_cos[idx.item()].item()), 4),
                    "visual_priors": get_species_visual_priors(cname, fam),
                })

            top1_prob = float(top_p[0].item())
            top2_prob = float(top_p[1].item()) if len(top_p) > 1 else 0.0
            margin = top1_prob - top2_prob
            top_species = candidates[0]["species_name"]
            top_genus = candidates[0]["genus"]
            top_species_fam = candidates[0]["family"]
            top1_conf = round(top1_prob * 100.0, 2)

            fam_probs = {}
            for idx in range(len(fn_classes)):
                cname = fn_classes[idx]
                fam = taxonomy.get(cname, {}).get("family", f"Other ({cname})")
                fam_probs[fam] = fam_probs.get(fam, 0.0) + float(probs[idx].item())
            top_fam, top_fam_p = max(fam_probs.items(), key=lambda item: item[1])

            # Out-Of-Distribution (OOD) Non-Plant Rejection Gate
            if max_cos_val < 0.35 and top1_prob < 0.20:
                priors = get_species_visual_priors("No Plant Detected", "Non-Plant")
                return {
                    "status": "NON_PLANT_OOD",
                    "species": "No Plant Detected",
                    "raw_top_species": "Non-Plant Subject",
                    "genus": "Non-Plant",
                    "family": "Non-Plant",
                    "family_confidence": 0.0,
                    "confidence": 0.0,
                    "species_confidence": 0.0,
                    "margin_pct": 0.0,
                    "ood_cosine": round(max_cos_val, 4),
                    "is_confident_species": False,
                    "candidates": [],
                    "top_candidates": [],
                    "visual_priors": priors,
                    "recommendations": [
                        "Point your camera directly at a living plant, leaf cluster, cactus, or flower.",
                        "Ensure the plant fills at least 35-60% of the frame.",
                        "Avoid photographing bedroom furniture, beds, walls, clothing, or electronics."
                    ],
                }

            top3_genera = [c["genus"] for c in candidates[:3] if c.get("genus") and c["genus"] != "Unknown"]
            gen_counts = Counter(top3_genera)
            shares_genus = any(cnt >= 2 for cnt in gen_counts.values())

            # Genus-Level Fallback / Grouping
            if top1_prob < 0.45:
                status = "GENUS_FAMILY_FALLBACK"
                final_species = f"{top_genus} ({top_species_fam})" if top_genus != "Unknown" else (top_species if not top_fam.startswith("Other") else "Botanical Specimen")
                final_fam = top_fam
                final_conf = round(max(top1_prob * 100.0, top_fam_p * 100.0), 2)
                recs = [
                    f"Specimen classified at genus/family level: {final_species}.",
                    "Capture an additional photo showing clear leaf venation and stem attachment."
                ]
            elif top1_prob < 0.75 or shares_genus:
                status = "GENUS_GROUPING"
                final_species = top_species if top_genus.lower() == top_species.lower() else f"{top_genus} (likely {top_species})"
                final_fam = top_species_fam
                final_conf = top1_conf
                recs = [
                    f"Classified under genus {top_genus}, most closely matching {top_species}.",
                    "Consider taking a close-up photo of leaf venation to distinguish cultivar varieties."
                ]
            else:
                status = "CONFIDENT_SPECIES"
                final_species = top_species
                final_fam = top_species_fam
                final_conf = top1_conf
                recs = []

            visual_priors = get_species_visual_priors(top_species, final_fam)

            return {
                "status": status,
                "species": final_species,
                "raw_top_species": top_species,
                "genus": top_genus,
                "family": final_fam,
                "family_confidence": round(top_fam_p * 100.0, 2),
                "confidence": final_conf,
                "species_confidence": top1_conf,
                "margin_pct": round(margin * 100.0, 2),
                "ood_cosine": round(max_cos_val, 4),
                "is_confident_species": (status == "CONFIDENT_SPECIES"),
                "candidates": candidates[:top_k],
                "top_candidates": candidates[:top_k],
                "visual_priors": visual_priors,
                "recommendations": recs,
            }
        return {
            "status": "NO_MODEL",
            "species": "Unidentified Botanical Specimen",
            "genus": "Unknown",
            "family": "Unknown",
            "confidence": 0.0,
            "is_confident_species": False,
            "candidates": [],
            "top_candidates": [],
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

            crop_logits = []
            crop_cos = []
            for crop in crops_list:
                x = _eval_views_tensor(crop, bundle["img_size"], bundle["mean"], bundle["std"]).to(DEVICE)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                    logits, cos_sim = bundle["model"](x)
                crop_logits.append(logits.float().mean(dim=0))
                crop_cos.append(cos_sim.float().mean(dim=0))

            # Pre-softmax multi-photo logit averaging: z_final = (z_1 + z_2) / 2
            avg_logits = torch.stack(crop_logits, dim=0).mean(dim=0)
            avg_cos = torch.stack(crop_cos, dim=0).mean(dim=0)

            probs = F.softmax(avg_logits / T, dim=-1)
            top_cos = float(avg_cos.max().item())
            max_cos_scores.append(top_cos)

            if blended_probs is None:
                blended_probs = w * probs
                ref_classes = bundle["classes"]
            else:
                blended_probs = blended_probs + w * probs
            total_w += w

    blended_probs = blended_probs / max(1e-6, total_w)
    top_p, top_i = blended_probs.topk(min(top_k * 2, len(ref_classes)))

    candidates = []
    for r, (p, idx) in enumerate(zip(top_p, top_i), 1):
        cname = ref_classes[idx.item()]
        fam = taxonomy.get(cname, {}).get("family", "Other Botanical Taxa")
        genus = extract_genus_from_species(cname)
        candidates.append({
            "rank": r,
            "species": cname,
            "species_name": cname,
            "genus": genus,
            "family": fam,
            "confidence": round(p.item() * 100.0, 2),
            "visual_priors": get_species_visual_priors(cname, fam),
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
    top_species = candidates[0]["species_name"]
    top_genus = candidates[0]["genus"]
    top_species_fam = candidates[0]["family"]
    top1_conf = round(top1_prob * 100.0, 2)

    top3_genera = [c["genus"] for c in candidates[:3] if c.get("genus") and c["genus"] != "Unknown"]
    gen_counts = Counter(top3_genera)
    shares_genus = any(cnt >= 2 for cnt in gen_counts.values())

    if is_ood or top1_prob < 0.45:
        status = "GENUS_FAMILY_FALLBACK"
        display_name = f"{top_genus} ({top_species_fam})" if top_genus != "Unknown" else top_species
        display_conf = top1_conf
        recommendations = [
            f"Specimen classified at genus/family level: {display_name}.",
            "To maximize confidence, capture an additional photo showing clear leaf venation and stem attachment."
        ]
    elif top1_prob < 0.75 or shares_genus:
        status = "GENUS_GROUPING"
        display_name = top_species if top_genus.lower() == top_species.lower() else f"{top_genus} (likely {top_species})"
        display_conf = top1_conf
        recommendations = [
            f"Classified under genus {top_genus}, most closely matching {top_species}.",
            "For exact cultivar confirmation, consider capturing a second macro shot of foliage detail."
        ]
    else:
        status = "CONFIDENT_SPECIES"
        display_name = top_species
        display_conf = top1_conf
        recommendations = []

    visual_priors = get_species_visual_priors(top_species, top_fam if status == "GENUS_FAMILY_FALLBACK" else top_species_fam)

    return {
        "status": status,
        "species": display_name,
        "raw_top_species": top_species,
        "genus": top_genus,
        "family": top_fam if status == "GENUS_FAMILY_FALLBACK" else top_species_fam,
        "family_confidence": round(top_fam_p * 100.0, 2),
        "confidence": display_conf,
        "species_confidence": top1_conf,
        "margin_pct": round(margin * 100.0, 2),
        "ood_cosine": round(mean_ood_cos, 3),
        "is_confident_species": (status == "CONFIDENT_SPECIES"),
        "candidates": candidates[:top_k],
        "top_candidates": candidates[:top_k],
        "visual_priors": visual_priors,
        "recommendations": recommendations,
    }
