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

## 🚀 Quick Start (Plug & Play)

### ⚡ 1. One-Click Launch (Easiest Method)
Once you clone the repository, you can start both AI engines simultaneously with a single click:

* **On Windows**: Simply double-click **`start.bat`** (or run `start.bat` in Command Prompt / PowerShell).
* **On macOS / Linux**: Run:
  ```bash
  chmod +x start.sh
  ./start.sh
  ```
* **Using Python directly**:
  ```bash
  python launch.py
  ```

> **What `start.bat` / `launch.py` handles automatically:**
> 1. ✅ Verifies model weights and automatically runs `git lfs pull` if weights are missing.
> 2. ✅ Creates `.venv` and installs dependencies on first run.
> 3. ✅ Boots **FloraNet-250 (Port 8000)** and **PhytoCare Ultra (Port 8001)** simultaneously.
> 4. ✅ Automatically opens your web browser to both portals.
> 5. ✅ Handles clean single-key (`Ctrl+C`) shutdown for both servers.

---

### 🛠️ 2. Manual Setup (If preferred)

#### Prerequisites: Git LFS (for model weights)
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

#### Environment Setup (Python 3.10+)
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

#### Running Individual Services Manually
* **FloraNet-250 (Species Identification — Port 8000)**:
  ```bash
  cd Plant_Identification_Experiment && python run_web.py
  ```
* **PhytoCare Ultra v5.0 (Disease Diagnosis & 500-Test Arena — Port 8001)**:
  ```bash
  cd Plant_Disease_Experiment && python run_web.py
  ```
* **Baseline Applications (MobileNetV3)**:
  - `cd Plant_Identification && python app/main.py` (Port 8000)
  - `cd Plant_Disease && python app/main.py` (Port 8001)

---

## 📊 Datasets & Models
- Trained model checkpoints (`best_model.pth`, `health_gate.pth`) and knowledge bases are included via Git LFS for immediate out-of-the-box inference.
- Raw training datasets (tens of gigabytes) are excluded from Git to follow repository best practices. All inference, validation, and live iNaturalist benchmarking run directly without requiring training data.

