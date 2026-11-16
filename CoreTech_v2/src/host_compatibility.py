"""
Biological Host-Pathogen Compatibility Matrix (Bayesian Prior).
Applies botanical family constraints to eliminate biologically impossible diagnoses
(e.g., obligate foliar Downy Mildew or Leaf Miners cannot infect leafless desert cacti;
Rust Fungi does not infect indoor Araceae such as Monstera or Pothos).
"""
from typing import Any, Dict, List, Tuple
import torch


# Compatibility Multipliers: 1.0 = normal/compatible, < 0.10 = biologically impossible / severe penalty, > 1.0 = high susceptibility
HOST_COMPATIBILITY_MATRIX: Dict[str, Dict[str, float]] = {
    "cactaceae": {
        "Downy_Mildew": 0.01,
        "Powdery_Mildew": 0.01,
        "Rust_Fungi": 0.01,
        "Leaf_Miners": 0.01,
        "Whiteflies": 0.05,
        "Aphids": 0.10,
        "Botrytis_Gray_Mold": 0.05,
        "Fungal_Leaf_Spot_Anthracnose": 0.15,
        "Mealybugs_Cochineal": 1.30,
        "Scale_Insects": 1.20,
        "Spider_Mites": 1.10,
        "Bacterial_Wilt_Soft_Rot": 1.15,
    },
    "crassulaceae": {
        "Downy_Mildew": 0.02,
        "Powdery_Mildew": 0.05,
        "Rust_Fungi": 0.02,
        "Leaf_Miners": 0.02,
        "Mealybugs_Cochineal": 1.30,
        "Scale_Insects": 1.15,
        "Bacterial_Wilt_Soft_Rot": 1.15,
    },
    "asphodelaceae": {
        # Aloe, Haworthia
        "Downy_Mildew": 0.02,
        "Powdery_Mildew": 0.02,
        "Leaf_Miners": 0.02,
        "Rust_Fungi": 0.90,  # Aloe rust is real (Uromyces aloes)
        "Bacterial_Wilt_Soft_Rot": 1.20,
        "Scale_Insects": 1.15,
        "Mealybugs_Cochineal": 1.15,
    },
    "asparagaceae": {
        # Sansevieria (Snake Plant), Spider Plant, Dracaena
        "Downy_Mildew": 0.02,
        "Powdery_Mildew": 0.04,
        "Rust_Fungi": 0.02,
        "Leaf_Miners": 0.05,
        "Fungal_Leaf_Spot_Anthracnose": 1.15,
        "Bacterial_Wilt_Soft_Rot": 1.15,
        "Spider_Mites": 1.05,
        "Scale_Insects": 1.10,
    },
    "araceae": {
        # Monstera, Philodendron, Epipremnum (Money Plant), Pothos, Aglaonema, ZZ Plant
        "Rust_Fungi": 0.02,  # Rust does not infect indoor Araceae foliage
        "Downy_Mildew": 0.05,
        "Fungal_Leaf_Spot_Anthracnose": 1.25,
        "Bacterial_Wilt_Soft_Rot": 1.20,
        "Thrips_Damage": 1.25,
        "Spider_Mites": 1.20,
        "Scale_Insects": 1.10,
        "Mealybugs_Cochineal": 1.10,
    },
    "orchidaceae": {
        "Downy_Mildew": 0.02,
        "Leaf_Miners": 0.05,
        "Fungal_Leaf_Spot_Anthracnose": 1.20,
        "Bacterial_Wilt_Soft_Rot": 1.25,
        "Botrytis_Gray_Mold": 1.25,
        "Mealybugs_Cochineal": 1.20,
        "Scale_Insects": 1.20,
    },
    "arecaceae": {
        # Palms
        "Downy_Mildew": 0.05,
        "Powdery_Mildew": 0.05,
        "Leaf_Miners": 0.10,
        "Scale_Insects": 1.30,
        "Spider_Mites": 1.20,
        "Fungal_Leaf_Spot_Anthracnose": 1.15,
        "Sooty_Mold": 1.20,
    },
    "rosaceae": {
        # Rose, Apple, Strawberry
        "Powdery_Mildew": 1.30,
        "Rust_Fungi": 1.35,
        "Botrytis_Gray_Mold": 1.25,
        "Fungal_Leaf_Spot_Anthracnose": 1.20,
        "Aphids": 1.25,
        "Spider_Mites": 1.20,
    },
    "begoniaceae": {
        "Powdery_Mildew": 1.30,
        "Botrytis_Gray_Mold": 1.30,
        "Bacterial_Wilt_Soft_Rot": 1.20,
    },
}


def _match_family_key(plant_family: str) -> str:
    """Matches free-form plant family string to normalized matrix key."""
    if not plant_family:
        return ""
    fl = plant_family.lower()
    for key in HOST_COMPATIBILITY_MATRIX.keys():
        if key in fl:
            return key
    # Extra common synonym mappings
    if "cactus" in fl or "cacti" in fl:
        return "cactaceae"
    if "aloe" in fl or "haworthia" in fl or "xanthorrhoea" in fl:
        return "asphodelaceae"
    if "sansevieria" in fl or "snake" in fl or "spider" in fl or "dracaena" in fl:
        return "asparagaceae"
    if "pothos" in fl or "monstera" in fl or "aroid" in fl or "epipremnum" in fl:
        return "araceae"
    if "orchid" in fl:
        return "orchidaceae"
    if "palm" in fl or "cycad" in fl:
        return "arecaceae"
    if "rose" in fl:
        return "rosaceae"
    if "begonia" in fl:
        return "begoniaceae"
    if "succulent" in fl or "crassula" in fl or "echeveria" in fl or "kalanchoe" in fl:
        return "crassulaceae"
    return ""


def apply_biological_host_prior(
    fine_probs: torch.Tensor,
    class_names: List[str],
    plant_family: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    r"""
    Applies Bayesian host compatibility prior:
      P_adjusted(C_i) \propto P_model(C_i) * M(C_i, HostFamily)
    Suppresses biologically impossible diseases for the given plant family.
    """
    mat_key = _match_family_key(plant_family)
    if not mat_key:
        return fine_probs, {
            "applied": False,
            "matched_family": "Universal / Unconstrained",
            "suppressed_diseases": [],
        }

    weights_map = HOST_COMPATIBILITY_MATRIX[mat_key]
    adjusted = fine_probs.clone()
    suppressed = []

    for idx, cname in enumerate(class_names):
        w = float(weights_map.get(cname, 1.0))
        raw_p = float(fine_probs[idx].item())
        if w < 0.10 and raw_p >= 0.02:
            suppressed.append({
                "disease": cname,
                "raw_confidence": round(raw_p * 100.0, 1),
                "penalty_factor": w,
                "reason": f"Biologically incompatible with host family {plant_family}.",
            })
        adjusted[idx] = adjusted[idx] * w

    # Re-normalize probability simplex
    s = adjusted.sum().clamp(min=1e-8)
    adjusted = adjusted / s

    return adjusted, {
        "applied": True,
        "matched_family": mat_key.capitalize(),
        "suppressed_diseases": suppressed,
    }
