# FYP_CoreTech: Intelligent Houseplant Identification & Pathology Advisory System

A dual-module deep learning and computer vision framework for indoor botanical care, comprising:
1. **FloraID (`Plant_Identification`):** 47-class houseplant species identification using MobileNetV3.
2. **PhytoCare AI (`Plant_Disease`):** 16-condition pathology diagnosis and treatment advisory engine for indoor houseplants.

---

## 🌟 Modules Overview

| Module | Core Architecture | Classes | Port | Description |
| :--- | :--- | :---: | :---: | :--- |
| **Plant_Identification** | MobileNetV3-Large | 47 Species | `8000` | Classifies indoor plant species from leaf/plant photos with top-5 confidence ranking. |
| **Plant_Disease** | FloraNet-V3 (MobileNetV3) | 16 Conditions | `8001` | Identifies plant pathologies (fungal, bacterial, pest, abiotic) and provides actionable treatment plans. |

---

## 📁 Repository Structure

```text
FYP_CoreTech/
├── Plant_Identification_Experiment/  # Flagship FloraNet-250 (ConvNeXt + Swin Ensemble, 250 species, Port 8000)
│   ├── web/                          # Web UI, visual benchmark, and API server
│   ├── artifacts/                    # Trained model weights (best_model.pth, Swin), class indices
│   ├── src/                          # Model architecture, 5-crop TTA, ensemble prediction
│   ├── reports/                      # 500-specimen tournament benchmarks and logs
│   └── run_web.py                    # Entrypoint runner for FloraNet-250 (Port 8000)
├── Plant_Disease_Experiment/         # Flagship PhytoCare Ultra v5.0 (31 classes, Health Gate, Port 8001)
│   ├── web/                          # Clinical diagnosis UI, 500-test live arena, API server
│   ├── artifacts/                    # Model weights (best_model.pth, health_gate.pth), knowledge base
│   ├── src/                          # 5-crop fovea TTA, spot localization, treatment advisory
│   ├── reports/                      # 500-specimen Internet benchmark reports & CSV
│   └── run_web.py                    # Entrypoint runner for PhytoCare Ultra (Port 8001)
├── Plant_Identification/             # Baseline species identification module (47 classes)
├── Plant_Disease/                    # Baseline disease diagnosis module (16 classes)
├── preprocessing/                    # Adaptive color normalization & foliage segmentation
├── requirements.txt                  # Unified Python dependencies
├── .gitattributes                    # Git LFS tracking for deep learning model checkpoints
├── .gitignore                        # Excludes raw datasets & temporary checkpoints
└── README.md                         # Documentation
```

---

## 🚀 Quick Start (Cloning & Running)

### ⚠️ Prerequisite: Git Large File Storage (Git LFS)
The trained model weights (`.pth` files) are tracked via **Git LFS**. Before cloning, ensure Git LFS is installed:

```bash
# 1. Install Git LFS (one-time setup)
git lfs install

# 2. Clone the repository
git clone https://github.com/Ken4502/FYP_CoreTech.git
cd FYP_CoreTech

# 3. If you already cloned without LFS, pull the binary weights:
git lfs pull
```

---

### Environment Setup (Python 3.10+)

```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies (CPU or GPU)
pip install -r requirements.txt
```

> **GPU Acceleration (Optional, recommended):**
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

### Running the Applications

#### 🌟 1. Flagship FloraNet-250 (Species Identification — Port 8000)
```bash
cd Plant_Identification_Experiment
python run_web.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

#### 🌿 2. Flagship PhytoCare Ultra v5.0 (Disease Diagnosis & 500-Test Arena — Port 8001)
```bash
cd Plant_Disease_Experiment
python run_web.py
```
* **Clinical Diagnostic UI:** **[http://localhost:8001](http://localhost:8001)**
* **500-Specimen Benchmark Arena:** **[http://localhost:8001/benchmark](http://localhost:8001/benchmark)**

---

#### 📌 3. Baseline Applications (MobileNetV3)
* **Plant Identification Baseline (Port 8000):** `cd Plant_Identification && python app/main.py`
* **Plant Disease Baseline (Port 8001):** `cd Plant_Disease && python app/main.py`

---

## 📊 Datasets & Models
- Trained model checkpoints (`best_model.pth`, `health_gate.pth`) and knowledge bases are included via Git LFS for immediate out-of-the-box inference.
- Raw training datasets (tens of gigabytes) are excluded from Git to follow repository best practices. All inference, validation, and live iNaturalist benchmarking run directly without requiring training data.

