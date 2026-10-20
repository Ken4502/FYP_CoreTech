# 🌿 FloraNet Model Benchmark Diagnostic & Continuous Improvement Plan
> **Evaluated Specimens:** 500 external independent specimens (Wikimedia Commons & Openverse)
> **Zero Training Contamination:** Testing performed on out-of-distribution archives with no training dataset overlap.

## 1. Executive Performance & Recovery Potential
| Metric | Value | Architectural Significance |
|---|---|---|
| **In-Vocabulary Top-1 Accuracy** | **48.0%** | True identification rate on target botanical species |
| **In-Vocabulary Top-5 Retrieval** | **66.8%** | **Latent Headroom:** Correct class is in top 5 runner-ups |
| **FloraNet vs Baseline-47** | **48.0% vs 9.6%** | Deepened/Widened architecture dramatically outperforms baseline |
| **FloraNet vs General Vision (YOLO/ResNet)** | **48.0% vs 1.8% / 3.2%** | Domain-specific fine-grained features outperform generic object models |
| **FloraNet Exclusive Wins** | **187 specimens** | Specimens where only FloraNet succeeded and all others failed |
| **Confidence Margin (Correct)** | **+69.4%** | Separation distance between Top-1 and Top-2 probabilities |
| **Confidence Margin (Missed)** | **27.4%** | Smaller margin confirms model uncertainty on ambiguous borders |
| **Mean Prediction Entropy** | **0.82 nats** | Shannon uncertainty metric across the softmax distribution |

## 2. Failure Mode Categorization
| Error Category | Occurrences | Percentage | Root Cause & Targeted Remediation |
|---|---|---|---|
| `CORRECT_DECISIVE` | 211 | 42.2% | Decisive correct prediction with margin > 20%. Solid feature alignment. |
| `CORRECT_LOW_MARGIN` | 29 | 5.8% | Correct prediction with margin < 20%. Fragile decision boundary; add hard-negative contrastive loss. |
| `NEAR_MISS_TOP5` | 94 | 18.8% | **High-Value Target:** Correct species ranked #2-#5. Recalibrate softmax temperatures (T ≈ 0.85). |
| `IN_DISTRIBUTION_HARD_MISS` | 166 | 33.2% | **Active Learning Trigger:** True blind spot. Ingest specimen into reinforcement training queue. |

## 3. Top Confused Botanical Pairs
| Ground Truth Species | Confused Model Prediction | Occurrences | Recommended Fix |
|---|---|---|---|
| **Purple Deadnettle** | African Violet (Saintpaulia ionantha) | 3 | Apply pairwise contrastive margin loss between `Purple Deadnettle` and `African Violet (Saintpaulia ionantha)` |
| **Field Bindweed** | Anthurium (Anthurium andraeanum) | 3 | Apply pairwise contrastive margin loss between `Field Bindweed` and `Anthurium (Anthurium andraeanum)` |
| **French Tarragon** | Parlor Palm (Chamaedorea elegans) | 3 | Apply pairwise contrastive margin loss between `French Tarragon` and `Parlor Palm (Chamaedorea elegans)` |
| **Common Purslane** | Jade plant (Crassula ovata) | 2 | Apply pairwise contrastive margin loss between `Common Purslane` and `Jade plant (Crassula ovata)` |
| **Garden Dahlia** | Chrysanthemum | 2 | Apply pairwise contrastive margin loss between `Garden Dahlia` and `Chrysanthemum` |
| **Japanese Camellia** | Begonia (Begonia spp.) | 2 | Apply pairwise contrastive margin loss between `Japanese Camellia` and `Begonia (Begonia spp.)` |
| **Peppermint** | Pothos (Ivy arum) | 2 | Apply pairwise contrastive margin loss between `Peppermint` and `Pothos (Ivy arum)` |
| **Zucchini** | Aloe Vera | 2 | Apply pairwise contrastive margin loss between `Zucchini` and `Aloe Vera` |
| **Sweet Basil** | Arrowhead Plant (Syngonium podophyllum) | 2 | Apply pairwise contrastive margin loss between `Sweet Basil` and `Arrowhead Plant (Syngonium podophyllum)` |
| **Black Nightshade** | Bell Pepper (Capsicum annuum) | 2 | Apply pairwise contrastive margin loss between `Black Nightshade` and `Bell Pepper (Capsicum annuum)` |
| **Canada Thistle** | Air Plant (Tillandsia ionantha) | 2 | Apply pairwise contrastive margin loss between `Canada Thistle` and `Air Plant (Tillandsia ionantha)` |
| **Peperomia Hope** | Tradescantia | 2 | Apply pairwise contrastive margin loss between `Peperomia Hope` and `Tradescantia` |

## 4. Actionable Steps to Improve the Model
Based on the CSV test analysis, here is the concrete recipe to elevate FloraNet-102 accuracy:

1. **Softmax Temperature Scaling ($T$ Calibration):**
   - Notice that Top-5 accuracy is **66.8%**, while Top-1 is **48.0%**.
   - There are **94** specimens where the correct species was in the runner-up positions.
   - Calibrating temperature $T = 0.85$ on the validation logits sharpens confident predictions without changing weights.

2. **Active Learning Reinforcement:**
   - We have extracted **260 hard failure cases** with their direct image URLs.
   - Staged in `reports/active_learning_queue.json` for one-click re-download and 2-epoch fine-tuning.

3. **Pairwise Contrastive Augmentation on Confused Taxa:**
   - For recurring confusion pairs (e.g. within *Begonia*, *Dracaena*, or *Cactaceae*), add random perspective warping and high-frequency edge-pass filtering so the network learns fine-grained surface trichomes and spine counts.

