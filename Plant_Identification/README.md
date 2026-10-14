# FloraID: House Plant Species Identification

An end-to-end computer vision pipeline and local web prototype for identifying 47 house plant species using **MobileNetV3-Large** and PyTorch with NVIDIA CUDA GPU acceleration.

---

## 🌟 Key Features

- **Dataset Reorganization (80/10/10):** Stratified train (80%), validation (10%), and test (10%) splits across 47 plant classes (14,790+ images).
- **Automated Image Sanitization:** Verifies image decodability, handles multiple formats (`.jpg`, `.jpeg`, `.png`, `.webp`), and sanitizes corrupt files.
- **Botanical Data Augmentation:**
  - `RandomResizedCrop(224)`
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomRotation(degrees=15)`
  - `ColorJitter` (brightness, contrast, saturation, hue) to mimic varied natural indoor lighting
  - ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Two-Stage Transfer Learning:**
  - **Stage 1 (Classifier Warm-up):** Frozen backbone, high learning rate (`1e-3`) for classifier head convergence.
  - **Stage 2 (Fine-tuning):** Unfrozen backbone with differential learning rates (`1e-4` backbone, `5e-4` head) and `CosineAnnealingLR`.
  - **Hardware Acceleration:** Native PyTorch Mixed Precision (`torch.amp.autocast`) on NVIDIA GeForce RTX 3060.
- **Evaluation & Diagnostics:**
  - Top-1 and Top-5 test set accuracy.
  - Per-class precision, recall, and F1-scores.
  - High-resolution 47-class confusion matrix (`reports/confusion_matrix.png`).
  - Loss and accuracy training curves (`reports/training_history.png`).
- **Model Deployment:**
  - Exported to **ONNX** (`artifacts/model.onnx`) with dynamic batching.
  - Exported to **TorchScript** (`artifacts/model.pt`).
- **Local Web Prototype:**
  - Built with **FastAPI** + clean HTML/CSS/JS frontend.
  - Drag-and-drop file upload, live preview, inference latency tracker, and top-5 confidence breakdown.

---

## 📁 Project Structure

```text
Plant_Identification/
├── .venv/                         # Project virtual environment (CUDA enabled)
├── data/                          # Stratified dataset
│   ├── train/                     # 80% per class
│   ├── val/                       # 10% per class
│   └── test/                      # 10% per class
├── artifacts/
│   ├── class_indices.json         # 47 plant species index-to-name mapping
│   ├── best_model.pth             # Best model checkpoint (by validation accuracy)
│   ├── last_model.pth             # Final training checkpoint
│   ├── model.onnx                 # Cross-platform ONNX export
│   ├── model.pt                   # TorchScript traced model
│   └── metrics.json               # Test evaluation metrics
├── reports/
│   ├── training_history.png       # Loss and accuracy curves
│   ├── confusion_matrix.png       # Confusion matrix visualization
│   └── classification_report.txt  # Detailed per-class metrics
├── src/
│   ├── __init__.py
│   ├── config.py                  # Hyperparameters and directory paths
│   ├── dataset.py                 # Augmentations and DataLoaders
│   ├── model.py                   # MobileNetV3 architecture definition
│   ├── train.py                   # Two-stage training and validation loop
│   ├── evaluate.py                # Test set evaluation
│   ├── predict.py                 # Single/batch inference CLI and API utility
│   └── export.py                  # ONNX & TorchScript exporter
├── app/
│   ├── main.py                    # FastAPI web server
│   └── static/                    # Single-page web prototype (HTML, CSS, JS)
├── organize_dataset.py            # Dataset verification & 80/10/10 splitter
├── run_pipeline.py                # Master orchestrator script
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Run the Entire Pipeline (End-to-End)
To organize data, train the model, evaluate on test data, export models, and launch the web server:
```powershell
.\.venv\Scripts\python.exe run_pipeline.py --serve
```

### 2. Run Individual Steps

#### Organize Dataset Only
```powershell
.\.venv\Scripts\python.exe organize_dataset.py
```

#### Train the Model Only
```powershell
.\.venv\Scripts\python.exe src/train.py
```

#### Evaluate on Test Set Only
```powershell
.\.venv\Scripts\python.exe src/evaluate.py
```

#### Export to ONNX and TorchScript
```powershell
.\.venv\Scripts\python.exe src/export.py
```

#### Predict a Single Plant Image (CLI)
```powershell
.\.venv\Scripts\python.exe src/predict.py --image path/to/leaf.jpg --top_k 5
```

#### Launch Local Web Prototype
```powershell
.\.venv\Scripts\python.exe app/main.py
```
Open your browser at: **http://127.0.0.1:8000**

---

## 🌿 Supported House Plant Species (47 Classes)

1. African Violet (*Saintpaulia ionantha*)
2. Aloe Vera
3. Anthurium (*Anthurium andraeanum*)
4. Areca Palm (*Dypsis lutescens*)
5. Asparagus Fern (*Asparagus setaceus*)
6. Begonia (*Begonia spp.*)
7. Bird of Paradise (*Strelitzia reginae*)
8. Birds Nest Fern (*Asplenium nidus*)
9. Boston Fern (*Nephrolepis exaltata*)
10. Calathea
11. Cast Iron Plant (*Aspidistra elatior*)
12. Chinese Money Plant (*Pilea peperomioides*)
13. Chinese evergreen (*Aglaonema*)
14. Christmas Cactus (*Schlumbergera bridgesii*)
15. Chrysanthemum
16. Ctenanthe
17. Daffodils (*Narcissus spp.*)
18. Dracaena
19. Dumb Cane (*Dieffenbachia spp.*)
20. Elephant Ear (*Alocasia spp.*)
21. English Ivy (*Hedera helix*)
22. Hyacinth (*Hyacinthus orientalis*)
23. Iron Cross begonia (*Begonia masoniana*)
24. Jade plant (*Crassula ovata*)
25. Kalanchoe
26. Lilium (*Hemerocallis*)
27. Lily of the valley (*Convallaria majalis*)
28. Money Tree (*Pachira aquatica*)
29. Monstera Deliciosa (*Monstera deliciosa*)
30. Orchid
31. Parlor Palm (*Chamaedorea elegans*)
32. Peace lily
33. Poinsettia (*Euphorbia pulcherrima*)
34. Polka Dot Plant (*Hypoestes phyllostachya*)
35. Ponytail Palm (*Beaucarnea recurvata*)
36. Pothos (*Ivy arum*)
37. Prayer Plant (*Maranta leuconeura*)
38. Rattlesnake Plant (*Calathea lancifolia*)
39. Rubber Plant (*Ficus elastica*)
40. Sago Palm (*Cycas revoluta*)
41. Schefflera
42. Snake plant (*Sanseviera*)
43. Tradescantia
44. Tulip
45. Venus Flytrap
46. Yucca
47. ZZ Plant (*Zamioculcas zamiifolia*)

