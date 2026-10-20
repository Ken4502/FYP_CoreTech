# Plant Identification Experimentation (Option C)

This folder contains the **Enhanced Plant Identification Experimentation Workspace**.
It is completely isolated from the original `Plant_Identification` codebase, ensuring that your baseline models, datasets, reports, and web servers remain 100% untouched.

---

## 🎯 What Makes This Version Stronger?

1. **Targeted Botanical Data Harvesting (`tools/harvester.py`)**:
   - Queries **iNaturalist API v1** for peer-reviewed, **Research-Grade** botanical observations mapped to scientific Latin names (`Yucca elephantipes`, `Epipremnum aureum`, etc.).
   - Automatic duplicate rejection using **Perceptual Difference Hashing (dHash)**.
   - Automatic non-plant noise filtering using **Preprocessing Method 1**: ensures images have sufficient plant foliage coverage before accepting them.
   - Isolated staging in `data/staging/` with safe 85% Train / 15% Val merging.
2. **Transfer Learning Backbones (`src/model.py`)**:
   - Upgrades from training from scratch to **MobileNetV3 Large** initialized with ImageNet-1k pre-trained weights.
   - Provides an immediate accuracy jump without catastrophic forgetting.
3. **Advanced Regularization & Augmentation (`src/dataset.py`)**:
   - **RandAugment** (`num_ops=2, magnitude=9`): Simulates diverse lighting, sharpness, and real-world indoor camera noise.
   - **RandomErasing** (`p=0.2`): Forces the network to focus on subtle leaf venation and serrations rather than global silhouette.
4. **Class-Balanced Loss (`src/train.py`)**:
   - Penalizes errors on scarce minority classes (like `Yucca` and `Begonia`) inversely proportional to their sample count, eliminating class collapse.
5. **Ablation Benchmarking (`src/evaluate.py`)**:
   - Directly calculates the performance $\Delta$ against your original baseline model (`CoreTech/Plant_Identification/artifacts/best_model.pth`).

---

## 🚀 Quick Start Guide

### Step 1: Initialize Experimental Data
To link the base dataset (0 MB extra disk space using hardlinks):
```bash
python setup_data.py
```

### Step 2: Harvest Verified Images for Weak Classes
To harvest 20 verified botanical images for the 8 under-represented classes (`Yucca`, `Calathea`, `Pothos`, `Begonia`, `Dracaena`, etc.):
```bash
python tools/harvester.py --species weak --count 20
```

Or harvest for a specific plant:
```bash
python tools/harvester.py --species "Yucca" --count 30
```

### Step 3: Merge Staged Images into Training Data
After verifying the images in `data/staging/`:
```bash
python tools/harvester.py --merge
```

### Step 4: Run Training
Train the enhanced model (Transfer Learning + RandAugment + Class Weighting):
```bash
python src/train.py
```

### Step 5: Evaluate & Compare with Baseline
```bash
python src/evaluate.py
```

---

## ⚡ End-to-End One-Command Execution
You can also run any step via `run_experiment.py`:
```bash
# Harvest weak classes
python run_experiment.py --harvest --count 15

# Merge staged data
python run_experiment.py --merge

# Train for 15 epochs
python run_experiment.py --train --epochs 15

# Evaluate on test set
python run_experiment.py --evaluate
```

