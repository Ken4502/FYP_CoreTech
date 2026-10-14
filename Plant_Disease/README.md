# PhytoCare AI: Indoor Plant Disease Diagnosis & Treatment Advisor

An end-to-end deep learning computer vision pipeline and interactive web advisory prototype for diagnosing plant pathologies in indoor houseplants using **MobileNetV3** (`FloraNet-V3`) with NVIDIA CUDA GPU acceleration.

---

## 🌟 Key Features

- **5 Diagnosable Plant Species (16 Conditions):**
  - **Aloe Vera:** Healthy, Anthracnose, Leaf Spot, Rust, Sunburn
  - **Cactus (Opuntia):** Healthy, Cochineal Scale Insect Infestation (*Dactylopius opuntiae*)
  - **Money Plant (Pothos):** Healthy, Bacterial Wilt Disease, Manganese Toxicity
  - **Snake Plant (Sansevieria):** Healthy, Anthracnose, Leaf Withering & Desiccation
  - **Spider Plant:** Healthy, Fungal Leaf Spot, Leaf Tip Necrosis / Tip Burn
- **Disease-Optimized Preprocessing & Augmentations:**
  - *Lesion-Preserving Cropping:* `RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.85, 1.15))` prevents cropping out small focal disease spots while maintaining scale invariance.
  - *Dual-Axis Flips:* `RandomHorizontalFlip(p=0.5)` and `RandomVerticalFlip(p=0.5)` to model the radial symmetry of lesions.
  - *Orientation Invariance:* `RandomRotation(degrees=30)` and `RandomAffine` for handheld phone photography.
  - *Diagnostic Color Preservation:* Subtle `ColorJitter(hue=0.03)` prevents confusing critical disease color signatures (e.g. orange rust vs yellow chlorosis).
- **Class-Imbalance Strategy:**
  - Addresses the severe dataset skew (~87% Aloe Vera vs. minority species) using inverse-frequency class-weighted `CrossEntropyLoss` with label smoothing (`0.05`).
- **Botanical Care & Treatment Recommendation Engine:**
  - Integrated knowledge base for all 16 conditions.
  - Provides **Immediate First-Aid Actions**, **Organic & Cultural Remedies**, **Chemical Fungicides/Bactericides/Pesticides**, and **Long-Term Prevention & Environmental Rules**.
- **Model Deployment & Verification:**
  - Multi-platform exports to **TorchScript** (`artifacts/model.pt`) and **ONNX** (`artifacts/model.onnx`).
  - Automated evaluation generating 16x16 confusion matrix heatmaps and per-class precision/recall/F1 metrics.
- **Interactive Web Prototype:**
  - Built with **FastAPI** + responsive single-page frontend.
  - Live photo upload, diagnosis verdict banner, tabbed treatment guides, and clickable confidence breakdown.

---

## 🪴 Available Plants & Diagnosable Conditions (16 Classes)

| Plant Species | Condition Class Name | Type | Severity | Primary Symptoms |
| :--- | :--- | :--- | :--- | :--- |
| **Aloe Vera** | `Aloe_Healthy` | Healthy Baseline | None | Firm, plump, erect green succulent leaves |
| **Aloe Vera** | `Aloe_Anthracnose` | Fungal (*Colletotrichum*) | Moderate | Sunken circular dark brown/black lesions |
| **Aloe Vera** | `Aloe_LeafSpot` | Fungal/Bacterial (*Alternaria*) | Mild–Mod | Scattered dark blemishes with yellow chlorotic halos |
| **Aloe Vera** | `Aloe_Rust` | Fungal (*Uromyces aloes*) | Moderate | Elevated reddish-brown pustules & circular depressions |
| **Aloe Vera** | `Aloe_Sunburn` | Abiotic Photodamage | Mild–Mod | Scorched bronze/reddish-brown papery bleached patches |
| **Cactus** | `Cactus_Healthy` | Healthy Baseline | None | Rigid green pads/stem, intact areoles and spines |
| **Cactus** | `Cactus_Dactylopius_Opuntia` | Pest Infestation | Severe | White cottony waxy clusters hiding sap-sucking scales |
| **Money Plant** | `Money_Plant_Healthy` | Healthy Baseline | None | Glossy, vibrant heart-shaped trailing vines |
| **Money Plant** | `Money_Plant_Bacterial_wilt_disease` | Bacterial (*Ralstonia*) | Severe | Rapid vine drooping, water-soaked slimy rotting stems |
| **Money Plant** | `Money_Plant_Manganese_Toxicity` | Nutritional Disorder | Moderate | Interveinal chlorosis, necrotic speckles, curling margins |
| **Snake Plant** | `Snake_Plant_Healthy` | Healthy Baseline | None | Rigid upright leathery sword blades with vivid banding |
| **Snake Plant** | `Snake_Plant_Anthracnose` | Fungal (*Colletotrichum*) | Moderate | Sunken red-bordered lesions with tan centers tearing blades |
| **Snake Plant** | `Snake_Plant_Leaf_Withering` | Root/Water Stress | Moderate | Limp wrinkled puckered blades (underwatering vs root rot) |
| **Spider Plant** | `Spider_Plant_Healthy` | Healthy Baseline | None | Arching variegated ribbon leaves producing healthy stolons |
| **Spider Plant** | `Spider_Plant_Fungal_leaf_spot` | Fungal Spotting | Moderate | Dark spots with chlorotic yellow halo borders on ribbons |
| **Spider Plant** | `Spider_Plant_Leaf_Tip_Necrosis` | Abiotic / Toxicity | Mild–Mod | Crisp brittle brown/black tips (fluoride/tap water/dry air) |

---

## 📁 Project Structure

```text
Plant_Disease/
├── data/                          # Reorganized dataset
│   ├── train/                     # 17,008 images across 16 classes
│   ├── val/                       # 2,078 images across 16 classes
│   └── test/                      # 2,011 images across 16 classes
├── artifacts/
│   ├── class_indices.json         # 16-class index mapping
│   ├── disease_knowledge_base.json# Curated treatment & care recommendations
│   ├── best_model.pth             # Best model weights checkpoint
│   ├── last_model.pth             # Final training checkpoint
│   ├── model.onnx                 # Cross-platform ONNX export
│   ├── model.pt                   # TorchScript traced model
│   └── metrics.json               # Test evaluation metrics
├── reports/
│   ├── training_history.png       # Loss & accuracy curves
│   ├── confusion_matrix.png       # 16x16 Confusion matrix heatmap
│   └── classification_report.txt  # Detailed per-class metrics
├── src/
│   ├── __init__.py
│   ├── config.py                  # Hyperparameters and path configurations
│   ├── dataset.py                 # Disease transforms, class weights, DataLoaders
│   ├── model.py                   # FloraNet-V3 (MobileNetV3 architectural blueprint)
│   ├── disease_knowledge.py       # Knowledge base & treatment engine
│   ├── train.py                   # Class-weighted training loop with AMP
│   ├── evaluate.py                # Test set evaluation & confusion matrix
│   ├── predict.py                 # CLI & callable diagnosis and advisory utility
│   └── export.py                  # ONNX & TorchScript exporter
├── app/
│   ├── main.py                    # FastAPI web server
│   └── static/                    # Frontend prototype (HTML, CSS, JS)
├── organize_dataset.py            # Dataset validator & fast hardlinker
├── run_pipeline.py                # Master orchestrator script
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Run the Entire Pipeline (End-to-End)
```powershell
& "..\Plant_Identification\.venv\Scripts\python.exe" run_pipeline.py --serve
```

### 2. Run Individual Steps

#### Organize Dataset Only
```powershell
& "..\Plant_Identification\.venv\Scripts\python.exe" organize_dataset.py
```

#### Train the Model
```powershell
& "..\Plant_Identification\.venv\Scripts\python.exe" src/train.py --epochs 14
```

#### Evaluate on Test Set
```powershell
& "..\Plant_Identification\.venv\Scripts\python.exe" src/evaluate.py
```

#### Export to ONNX & TorchScript
```powershell
& "..\Plant_Identification\.venv\Scripts\python.exe" src/export.py
```

#### Diagnose a Plant Image via CLI
```powershell
& "..\Plant_Identification\.venv\Scripts\python.exe" src/predict.py --image path/to/leaf.jpg --top_k 3
```

#### Launch Local Web Prototype
```powershell
& "..\Plant_Identification\.venv\Scripts\python.exe" app/main.py
```
Open your browser at: **http://127.0.0.1:8001**

