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
├── Plant_Identification/          # Species identification module
│   ├── app/                       # FastAPI web prototype (port 8000) & static UI
│   ├── artifacts/                 # Trained model weights (best_model.pth), class indices
│   ├── src/                       # PyTorch model architecture, dataset loaders, inference logic
│   ├── reports/                   # Training history & confusion matrix graphs
│   ├── Dockerfile                 # Standalone container build for FloraID
│   └── requirements.txt           # Python dependencies
├── Plant_Disease/                 # Disease diagnosis and treatment advisory module
│   ├── app/                       # FastAPI web prototype (port 8001) & static UI
│   ├── artifacts/                 # Trained model weights (best_model.pth), knowledge base
│   ├── src/                       # Diagnostics engine, treatment advisor, evaluation
│   ├── reports/                   # Confusion matrix & performance reports
│   ├── Dockerfile                 # Standalone container build for PhytoCare
│   └── requirements.txt           # Python dependencies
├── .dockerignore                  # Docker ignore rules (keeps builds lightweight)
├── .gitignore                     # Git ignore rules (protects repo from large binaries)
├── docker-compose.yml             # Single-command orchestrator for both services
└── README.md                      # Project documentation
```

---

## 🚀 Quick Start (Plug & Play)

### Option 1: Run with Docker (Recommended for instant plug-and-play)
No Python or CUDA configuration required. With Docker installed on any device:

```bash
# Build and launch both FloraID and PhytoCare AI
docker compose up -d
```
Access the applications:
* **FloraID (Species Identification):** [http://localhost:8000](http://localhost:8000)
* **PhytoCare AI (Disease Diagnosis):** [http://localhost:8001](http://localhost:8001)

To stop the services:
```bash
docker compose down
```

---

### Option 2: Run Locally with Python

#### 1. Plant Identification (FloraID)
```bash
cd Plant_Identification
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python app/main.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

#### 2. Plant Disease Diagnosis (PhytoCare AI)
```bash
cd Plant_Disease
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python app/main.py
```
Open [http://localhost:8001](http://localhost:8001) in your browser.

---

## 📊 Datasets & Models
- Pretrained model checkpoints (`best_model.pth`) and class mappings are prepackaged in each module's `artifacts/` directory for immediate inference.
- Full training datasets (14,000+ images) are excluded from this repository to adhere to version control best practices.
