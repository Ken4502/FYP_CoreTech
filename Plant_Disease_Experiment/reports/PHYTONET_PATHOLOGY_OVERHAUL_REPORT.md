# PhytoCare Ultra: Deep Botanical Transfer & Class-Balanced Pathology Diagnosis Report

**Author / Project:** Final Year Project (FYP) — Autonomous Botanical & Phytopathological Intelligence System  
**Hardware Platform:** NVIDIA GeForce RTX 3060 (12 GB VRAM)  
**Evaluation Dataset:** 2,011 Unseen Multi-Species Test Specimens across 16 Houseplant Pathology Taxa  
**Evaluation Protocol:** 5-Crop Test-Time Augmentation (TTA) with Platt Softmax Temperature Calibration ($T = 0.81$)  
**Date:** October 2026  

---

## 1. Executive Summary

The baseline plant pathology classifier suffered from severe data skew (Aloe Vera comprising >70% of all frames) and generic ImageNet initialization, leading to catastrophic failure on minority houseplant diseases (overall test accuracy **57.43%**, Macro F1 **0.4807**, and complete failure on *Spider Plant* pathologies with **0.00% F1**).

To elevate the plant disease system to tournament-grade clinical performance matching our 250-class botanical identifier, we architected **PhytoNet Ultra**:
1. **Botanical Prior Knowledge Transfer:** Initialized the 50.2M-parameter ConvNeXt-Small backbone directly with visual priors learned by our FloraNet-250 botanical classification model (`Plant_Identification_Experiment/artifacts/best_model.pth`), imparting pre-existing leaf venation, serration, and chlorophyll texture awareness.
2. **Dataset Equalization & Targeted Minority Harvesting:** Pruned 10,000+ redundant video frames from the Aloe classes and harvested verified research-grade pathology specimens from iNaturalist, followed by lesion-preserving geometric augmentations to achieve a balanced 10,500-image training corpus.
3. **Progressive Multi-Stage Acuity:** Trained at $256 \times 256$ (Stage 1) followed by high-resolution fine-tuning at $384 \times 384$ (Stage 2) using Class-Balanced Focal Loss ($\gamma = 2.0$, label smoothing = 0.05).
4. **Test-Time Augmentation (TTA) & Temperature Scaling:** Deployed 5-crop spatial TTA and Platt scaling ($T = 0.81$) to eliminate overconfidence and calibrate diagnostic probabilities.
5. **Explainable AI (XAI) Spot Detection:** Integrated Grad-CAM lesion activation mapping to produce bounding boxes and damage severity percentages directly on input leaves.

```
========================================================================================
METRIC                   BASELINE RESNET-50        PHYTONET ULTRA (CONVNEXT-384)   DELTA
========================================================================================
Overall Test Accuracy         57.43%                          89.01%             +31.58%
Macro Average F1-Score        0.4807                          0.9279             +0.4472 (+93.0%)
Spider Plant Fungal Spot       0.00%                         100.00%            +100.00% (FIXED)
Spider Plant Tip Necrosis      0.00%                         100.00%            +100.00% (FIXED)
Aloe Rust F1-Score            20.00%                          82.95%             +62.95%
Money Plant Bacterial Wilt    50.00%                         100.00%             +50.00%
Money Plant Mn Toxicity       33.33%                         100.00%             +66.67%
Snake Plant Anthracnose       37.50%                          71.43%             +33.93%
Cactus Scale Insect           89.47%                          97.14%              +7.67%
========================================================================================
```

---

## 2. Experimental Timeline & Computational Budget

All experiments were executed locally on an **NVIDIA GeForce RTX 3060**:

| Pipeline Stage | Scope & Operations | Resolution / Batch | Duration | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 0: Data Harvesting & Equalization** | Multi-threaded iNaturalist download, Aloe pruning, lesion augmentations | N/A | **15.2 min** | Completed |
| **Stage 1: Botanical Prior Transfer** | 10 Epochs ConvNeXt-Small training with Focal Loss | $256 \times 256$, B=32 | **30.2 min** | Completed (89.61% Val) |
| **Stage 2: High-Acuity Fine-Tuning** | 8 Epochs 384px fine-tuning with Cosine Annealing | $384 \times 384$, B=16 | **34.7 min** | Completed (91.29% Val) |
| **Stage 3: Platt Temperature Calibration** | NLL minimization across validation logits | N/A | **12.4 sec** | Completed ($T = 0.81$) |
| **Stage 4: Full Test Benchmark (2,011 imgs)** | 5-Crop TTA inference + confusion matrix generation | $384 \times 384$, 5 crops | **5.5 min** | Completed (89.01% Acc) |
| **Stage 5 (Optional): Swin-384 Dual Ensemble** | Fine-tuning Swin-T 384px for architectural ensemble | $384 \times 384$, B=16 | *~48 min* | Ready on demand |

*Total time elapsed to reach the current 89.01% production model: **~86 minutes**.*

---

## 3. Detailed Per-Class Benchmark Analysis

The evaluation was performed across all **2,011 genuine test images** in `CoreTech/Plant_Disease_Experiment/data/test`. Below is the comprehensive classification breakdown:

| Biological Pathology Taxon | Test Support | Baseline Precision | Baseline Recall | Baseline F1 | PhytoNet Precision | PhytoNet Recall | PhytoNet F1 | F1 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aloe Anthracnose** | 377 | 0.6172 | 0.4960 | 0.5500 | **0.9043** | **0.9019** | **0.9031** | **+35.31%** |
| **Aloe Healthy** | 537 | 0.6000 | 0.9218 | 0.7269 | **0.9000** | **0.9385** | **0.9189** | **+19.20%** |
| **Aloe Leaf Spot** | 378 | 0.5401 | 0.3915 | 0.4540 | **0.8641** | **0.8413** | **0.8525** | **+39.85%** |
| **Aloe Rust** | 195 | 0.2345 | 0.1744 | 0.2000 | **0.7531** | **0.9231** | **0.8295** | **+62.95%** |
| **Aloe Sunburn** | 383 | 0.6176 | 0.5144 | 0.5613 | **0.9572** | **0.8172** | **0.8817** | **+32.04%** |
| **Cactus Dactylopius (Scale)** | 35 | 0.8293 | 0.9714 | 0.8947 | **0.9714** | **0.9714** | **0.9714** | **+7.67%** |
| **Cactus Healthy** | 29 | 0.9583 | 0.7931 | 0.8679 | **0.9655** | **0.9655** | **0.9655** | **+9.76%** |
| **Money Plant Bacterial Wilt** | 4 | 0.5000 | 0.5000 | 0.5000 | **1.0000** | **1.0000** | **1.0000** | **+50.00%** |
| **Money Plant Healthy** | 12 | 0.5789 | 0.9167 | 0.7097 | **1.0000** | **1.0000** | **1.0000** | **+29.03%** |
| **Money Plant Manganese Toxicity** | 9 | 0.6667 | 0.2222 | 0.3333 | **1.0000** | **1.0000** | **1.0000** | **+66.67%** |
| **Snake Plant Anthracnose** | 6 | 0.3000 | 0.5000 | 0.3750 | **0.6250** | **0.8333** | **0.7143** | **+33.93%** |
| **Snake Plant Healthy** | 10 | 0.7143 | 1.0000 | 0.8333 | **0.9000** | **0.9000** | **0.9000** | **+6.67%** |
| **Snake Plant Leaf Withering** | 12 | 1.0000 | 0.1667 | 0.2857 | **1.0000** | **0.8333** | **0.9091** | **+62.34%** |
| **Spider Plant Fungal Leaf Spot** | 9 | 0.0000 | 0.0000 | **0.0000** | **1.0000** | **1.0000** | **1.0000** | **+100.00%** |
| **Spider Plant Healthy** | 7 | 0.2500 | 1.0000 | 0.4000 | **1.0000** | **1.0000** | **1.0000** | **+60.00%** |
| **Spider Plant Leaf Tip Necrosis**| 8 | 0.0000 | 0.0000 | **0.0000** | **1.0000** | **1.0000** | **1.0000** | **+100.00%** |
| **Macro Average** | **2,011** | **0.5254** | **0.5355** | **0.4807** | **0.9275** | **0.9328** | **0.9279** | **+0.4472** |
| **Weighted Average** | **2,011** | **0.5648** | **0.5743** | **0.5520** | **0.8951** | **0.8901** | **0.8905** | **+31.58%** |

---

## 4. Key Architectural Discoveries

### 4.1 Cross-Domain Botanical Weight Transfer
Standard transfer learning initializes medical or agricultural backbones from ImageNet-1K (which is heavily biased toward dogs, vehicles, and everyday objects). By transferring pre-trained weights from our **FloraNet-250** botanical identification model (`Plant_Identification_Experiment/artifacts/best_model.pth`), the shallow and intermediate convolutional blocks already possessed high-frequency filters tuned specifically to:
- Secondary leaf venation and parallel monocot veins.
- Leaf margin serrations and succulents' cuticle wax reflection.
- Chlorophyll reflectance distributions under variable lighting.

This enabled the network to achieve **75.06% accuracy on Epoch 1** and **89.61% by Epoch 10**, skipping dozens of epochs of foundational feature learning.

### 4.2 Resolution Scaling ($256 \to 384\text{px}$)
Pathological lesions (e.g. *Aloe rust* telia pustules and *Money plant* manganese micro-specks) occupy only 1–3% of total image pixels. Scaling resolution from 256px to 384px yielded a direct **+1.68% boost in validation accuracy** (from 89.61% to 91.29%), accompanied by a marked reduction in false positives between *Aloe Leaf Spot* and *Aloe Sunburn*.

### 4.3 Resolving the Class Imbalance Gradient Starvation
In the baseline model, Aloe Vera accounted for 94% of training gradient updates. The loss gradients from 9 Spider Plant samples were mathematically dwarfed by thousands of Aloe samples, resulting in 0.00% recall. Equalizing the corpus to 1,000 images per Aloe condition and synthesizing/harvesting up to 500 images per minority condition provided sufficient gradient signal for the optimizer, resulting in **100% precision and recall on Spider Plant diseases**.

---

## 5. Clinical Web Server Integration & Explainability

The production web server is running locally on port **8001** (`http://127.0.0.1:8001/`):
- **Grad-CAM Lesion Heatmap:** Automatically highlights fungal pustules, necrotic spots, and scale insect clusters in red/amber contours with bounding boxes.
- **Severity Score:** Computes the ratio of diseased leaf tissue to healthy tissue.
- **Curated Knowledge Base:** Automatically issues tailored first-aid steps, copper/neem oil treatment, isolation recommendations, and watering adjustments.

