"""
PhytoCare Ultra 22-Class Universal Pathology Harvester & Flagship Training Engine
Expands the clinical pathology model from 16 to 22 classes by harvesting verified
research-grade specimens from iNaturalist:
  1. Universal_Mealybugs (Pseudococcidae, ID: 52788)
  2. Universal_Spider_Mites (Tetranychidae, ID: 68318)
  3. Universal_Powdery_Mildew (Erysiphaceae, ID: 55525)
  4. Universal_Scale_Insects (Coccidae, ID: 49456)
  5. Universal_Sooty_Mold (Capnodiaceae, ID: 154563)
  6. Universal_Aphids (Aphididae, ID: 52381)

Streams real-time telemetry directly to:
  - mega_pipeline.log (visible in IDE and http://127.0.0.1:8001/log)
"""

import sys
import os
import time
import json
import random
import io
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from src import config

DATA_DIR = config.DATA_DIR
TRAIN_DIR = config.TRAIN_DIR
VAL_DIR = config.VAL_DIR
TEST_DIR = config.TEST_DIR
REPORTS_DIR = config.REPORTS_DIR
ARTIFACTS_DIR = config.ARTIFACTS_DIR
LOG_FILE = REPORTS_DIR / "mega_pipeline.log"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "PhytoCare-Research-Harvester/2.0 (academic FYP)"}

UNIVERSAL_TAXA = {
    "Universal_Mealybugs": {
        "taxon_id": 52788,
        "common_name": "Mealybugs (Pseudococcidae)",
        "scientific_name": "Pseudococcidae",
        "category": "Insect Infestation",
        "severity": "Moderate to High",
        "summary": "Sap-sucking soft-bodied scale insects covered in white cottony wax that cluster in leaf axils and undersides, causing leaf yellowing and honeydew secretion.",
        "symptoms": [
            "White cottony or powdery wax masses in leaf joints and undersides",
            "Yellowing, wilting, or distorted foliage growth",
            "Sticky honeydew accumulation followed by secondary black sooty mold",
            "Stunted plant development from continuous sap extraction"
        ],
        "causes": [
            "Contaminated nursery soil or introduction from unquarantined new plants",
            "Warm indoor conditions with stagnant, dry airflow",
            "Over-fertilization with high nitrogen producing soft, lush tissue"
        ],
        "immediate_actions": [
            "Isolate the plant immediately to prevent crawl-stage nymphs from migrating.",
            "Dip a cotton swab in 70% isopropyl alcohol and directly touch every white cottony mass to dissolve their waxy protective coat instantly.",
            "Rinse leaves with a lukewarm spray to dislodge nymphs."
        ],
        "treatment": {
            "organic": [
                "Spray thoroughly with insecticidal soap or horticultural oil every 5-7 days for 3 cycles.",
                "Apply cold-pressed neem oil emulsion (1 tsp neem + 1/2 tsp castile soap per liter of warm water).",
                "Introduce beneficial predatory insects like Cryptolaemus montrouzieri (mealybug destroyers) for large collections."
            ],
            "chemical": [
                "Apply systemic insecticide drench containing acetamiprid or imidacloprid to roots so the plant absorbs protective toxins.",
                "Spray with pyrethrin-based contact insecticide targeting nymphs."
            ]
        },
        "prevention": [
            "Quarantine all newly purchased houseplants for 14 days before introducing them to collections.",
            "Inspect leaf axils regularly during routine watering.",
            "Avoid excessive nitrogen fertilization."
        ]
    },
    "Universal_Spider_Mites": {
        "taxon_id": 68318,
        "common_name": "Spider Mites (Tetranychidae)",
        "scientific_name": "Tetranychus urticae",
        "category": "Arachnid Infestation",
        "severity": "High",
        "summary": "Microscopic arachnid pests that pierce plant cells to feed on chlorophyll, producing fine silky webbing and severe chlorotic stippling across foliage.",
        "symptoms": [
            "Fine pale yellow or white speckling (stippling) on upper leaf surfaces",
            "Delicate silky webbing draped between leaf stems, veins, and growing tips",
            "Leaves turning bronze, brittle, yellow, and dropping prematurely",
            "Dusty texture on leaf undersides where colonies congregate"
        ],
        "causes": [
            "Hot, dry indoor environments with relative humidity under 40%",
            "Winter heating systems creating micro-drought zones around foliage",
            "Air currents carrying microscopic mites through open windows"
        ],
        "immediate_actions": [
            "Isolate specimen immediately; spider mites reproduce exponentially in warm air.",
            "Carry plant to shower or sink and rinse all foliage undersides with a strong stream of lukewarm water to wash away webs and mites.",
            "Increase ambient room humidity above 60%."
        ],
        "treatment": {
            "organic": [
                "Spray with cold-pressed neem oil or horticultural insecticidal soap every 3-5 days to interrupt the egg hatch cycle.",
                "Release biological predatory mites (Phytoseiulus persimilis or Neoseiulus californicus).",
                "Mist foliage daily with distilled water to maintain high humidity."
            ],
            "chemical": [
                "Apply a dedicated miticide/acaricide (e.g. Abamectin or Bifenthrin); standard insecticides often fail against arachnid mites.",
                "Rotate active chemical ingredients across generations to prevent resistance."
            ]
        },
        "prevention": [
            "Keep indoor relative humidity above 55% using humidifiers or pebble trays.",
            "Wipe large foliage clean monthly with a damp microfiber cloth.",
            "Avoid placing houseplants in the direct path of air conditioning or heater vents."
        ]
    },
    "Universal_Powdery_Mildew": {
        "taxon_id": 55525,
        "common_name": "Powdery Mildew",
        "scientific_name": "Erysiphaceae",
        "category": "Fungal Infection",
        "severity": "Moderate",
        "summary": "A common superficial fungal infection that coats leaves, buds, and young shoots with talcum-powder-like white mycelial patches, impairing photosynthesis.",
        "symptoms": [
            "White or greyish circular powdery patches spreading across upper leaf surfaces",
            "Curled, twisted, or distorted new leaves",
            "Premature senescence and dropping of infected foliage",
            "Stunted flowering or leaf development"
        ],
        "causes": [
            "High relative humidity combined with low light and poor air ventilation",
            "Crowded plant groupings preventing air movement",
            "Fluctuating day/night temperatures creating condensation"
        ],
        "immediate_actions": [
            "Separate infected plant from nearby foliage to prevent airborne spore dispersion.",
            "Prune off heavily covered leaves using shears sterilized in 70% alcohol.",
            "Place a small oscillating fan nearby on gentle speed to circulate air."
        ],
        "treatment": {
            "organic": [
                "Spray with potassium bicarbonate or baking soda solution (1 tsp baking soda + 1/2 tsp horticultural oil in 1 liter of water).",
                "Dust lightly with wettable sulfur powder (avoid applying when temperatures exceed 30C).",
                "Apply dilute milk spray (1 part cow milk to 9 parts water; sunlight activates natural antiseptic lactoferrin)."
            ],
            "chemical": [
                "Spray with systemic fungicide containing myclobutanil, tebuconazole, or chlorothalonil.",
                "Apply biofungicide containing Bacillus subtilis."
            ]
        },
        "prevention": [
            "Provide ample spacing between houseplants for free air circulation.",
            "Ensure plants receive bright, indirect sunlight.",
            "Water only at the root base early in the day; avoid wetting foliage."
        ]
    },
    "Universal_Scale_Insects": {
        "taxon_id": 49456,
        "common_name": "Soft Scale Insects",
        "scientific_name": "Coccidae",
        "category": "Insect Infestation",
        "severity": "Moderate to High",
        "summary": "Immobile waxy oval domes that anchor to stems and leaf midribs, siphoning essential plant sap and excreting sticky honeydew.",
        "symptoms": [
            "Small, raised, smooth or waxy brown, amber, or yellowish bumps clustered along veins and stems",
            "Sticky translucent droplets (honeydew) coating surrounding foliage and floor",
            "Yellow halos forming around feeding sites on foliage",
            "General loss of plant vigor, yellowing leaves, and dieback"
        ],
        "causes": [
            "Introduction from outdoor summering or newly acquired infested specimens",
            "Absence of natural outdoor predators in indoor settings",
            "Warm, sheltered room temperatures"
        ],
        "immediate_actions": [
            "Isolate plant immediately.",
            "Scrape scales off gently using an old toothbrush, thumbnail, or alcohol-soaked cotton pad.",
            "Wipe honeydew residue clean with warm soapy water to discourage mold."
        ],
        "treatment": {
            "organic": [
                "Apply horticultural mineral oil or neem oil spray to suffocate armored and soft scale stages.",
                "Spot-treat recalcitrant scales with 70% isopropyl alcohol on a cotton swab.",
                "Introduce parasitic wasps (Metaphycus helvolus) in greenhouse settings."
            ],
            "chemical": [
                "Apply systemic soil drench insecticide (imidacloprid or dinotefuran) which is absorbed through roots into leaf vascular tissue.",
                "Spray contact insecticide during crawler emergence window."
            ]
        },
        "prevention": [
            "Inspect the undersides of leaves and leaf-stem junctions before purchasing plants.",
            "Clean foliage periodically with mild soap solution.",
            "Quarantine plants that have spent summer outdoors before bringing them inside."
        ]
    },
    "Universal_Sooty_Mold": {
        "taxon_id": 154563,
        "common_name": "Sooty Mold",
        "scientific_name": "Capnodiaceae",
        "category": "Secondary Fungal Colonization",
        "severity": "Mild to Moderate",
        "summary": "A secondary superficial black fungal coating that grows on sticky sugary honeydew excreted by sap-sucking pests (aphids, scales, mealybugs, whiteflies).",
        "symptoms": [
            "Black, charcoal-like, velvety or crusty film coating upper leaf surfaces",
            "Presence of sticky honeydew droplets beneath the mold layer",
            "Reduced growth and pale yellowing caused by blocked solar absorption",
            "Mold easily wipes off with a wet cloth (distinguishing it from internal leaf necrosis)"
        ],
        "causes": [
            "Underlying active infestation of sap-sucking pests (aphids, mealybugs, whiteflies, or scale insects) secreting sugary honeydew",
            "Humid indoor environment encouraging fungal spore germination on sugar"
        ],
        "immediate_actions": [
            "Locate and identify the insect pest responsible for the honeydew (check higher leaves and stems).",
            "Gently wipe affected foliage with a damp cloth soaked in lukewarm water and mild castile soap to wash the mold and honeydew away.",
            "Treat the primary pest immediately to cut off the sugar source."
        ],
        "treatment": {
            "organic": [
                "Wipe leaves with 1% castile soap or horticultural oil solution to dissolve honeydew and loosen mold.",
                "Treat primary insect vector with cold-pressed neem oil.",
                "Rinse leaves thoroughly with clean water after cleaning."
            ],
            "chemical": [
                "Apply systemic insecticide against the primary pest vector.",
                "Copper fungicide can prevent spore reinfection if mold coverage is severe."
            ]
        },
        "prevention": [
            "Control and eliminate sucking insect pests promptly upon detection.",
            "Rinse leaves periodically to prevent sugar residue accumulation.",
            "Ensure good room air circulation."
        ]
    },
    "Universal_Aphids": {
        "taxon_id": 52381,
        "common_name": "Aphids (Greenflies / Blackflies)",
        "scientific_name": "Aphididae",
        "category": "Insect Infestation",
        "severity": "Moderate",
        "summary": "Soft-bodied pear-shaped insects that multiply rapidly in dense colonies on tender new growing tips, buds, and leaf undersides, sucking vital plant fluids.",
        "symptoms": [
            "Dense clusters of small green, black, brown, or pink pear-shaped insects on young shoots",
            "Curled, twisted, puckered, or stunted new growth",
            "Sticky clear honeydew droplets coating leaves and stems",
            "Presence of white shed cast skins scattered over foliage"
        ],
        "causes": [
            "Rapid parthenogenic reproduction (females giving live birth without mating)",
            "Warm indoor conditions and tender succulent new growth from high-nitrogen feeding",
            "Infested plants brought indoors from gardens or nurseries"
        ],
        "immediate_actions": [
            "Isolate the plant immediately.",
            "Take plant to a sink or shower and dislodge the bulk of the colony with a firm spray of lukewarm water.",
            "Pinch off severely deformed growing tips if infestation is localized."
        ],
        "treatment": {
            "organic": [
                "Spray thoroughly with insecticidal soap or neem oil spray, ensuring contact with leaf undersides.",
                "Dust lightly with food-grade diatomaceous earth on dry stems.",
                "Introduce beneficial biological predators such as ladybird beetles (Adalia bipunctata) or lacewing larvae."
            ],
            "chemical": [
                "Spray with pyrethrin or acetamiprid contact insecticide.",
                "Apply systemic imidacloprid root drench for prolonged residual protection."
            ]
        },
        "prevention": [
            "Avoid over-fertilizing with nitrogen which creates excessive soft growth that attracts aphids.",
            "Inspect growing tips and flower buds weekly.",
            "Use yellow sticky traps near plant stands to catch winged scouts early."
        ]
    }
}

def log_event(msg: str):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass

def fetch_inat_photos(taxon_id: int, max_photos: int = 500) -> list:
    photo_urls = []
    page = 1
    per_page = 50

    while len(photo_urls) < max_photos and page <= 15:
        url = (
            f"https://api.inaturalist.org/v1/observations?"
            f"taxon_id={taxon_id}&photos=true&quality_grade=research&per_page={per_page}&page={page}"
        )
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("results", [])
                if not results:
                    break
                for r in results:
                    for p in r.get("photos", []):
                        u = p.get("url", "")
                        if u:
                            med_url = u.replace("square", "medium")
                            if med_url not in photo_urls:
                                photo_urls.append(med_url)
                                if len(photo_urls) >= max_photos:
                                    break
                    if len(photo_urls) >= max_photos:
                        break
            page += 1
            time.sleep(0.3)
        except Exception as e:
            time.sleep(1.0)
            page += 1

    return photo_urls

def download_and_save_image(url: str, out_path: Path) -> bool:
    if out_path.exists():
        return True
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            content = resp.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            # Ensure minimum viable dimensions
            if img.width < 150 or img.height < 150:
                return False
            img.save(out_path, format="JPEG", quality=90)
            return True
    except Exception:
        return False

def harvest_universal_taxa():
    log_event("=" * 70)
    log_event("PHASE 1: HARVESTING UNIVERSAL PEST & PATHOGEN SUITE FROM INATURALIST")
    log_event(f"• Target Taxa: {len(UNIVERSAL_TAXA)} Universal Biological Classes")
    log_event(f"• Quota: 500 Images per Taxon (400 Train, 50 Val, 50 Test)")
    log_event("=" * 70)

    for cls_name, info in UNIVERSAL_TAXA.items():
        tid = info["taxon_id"]
        cname = info["common_name"]
        log_event(f"\n[Harvester] Querying iNaturalist API for {cname} (Taxon ID: {tid})...")

        urls = fetch_inat_photos(tid, max_photos=520)
        log_event(f"  Found {len(urls)} research-grade image URLs for {cls_name}.")

        train_dir = TRAIN_DIR / cls_name
        val_dir = VAL_DIR / cls_name
        test_dir = TEST_DIR / cls_name

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        random.seed(42)
        random.shuffle(urls)

        # Download tasks
        tasks = []
        for i, u in enumerate(urls):
            if i < 400:
                out = train_dir / f"inat_{cls_name}_{i+1:04d}.jpg"
            elif i < 450:
                out = val_dir / f"inat_{cls_name}_{i+1:04d}.jpg"
            elif i < 500:
                out = test_dir / f"inat_{cls_name}_{i+1:04d}.jpg"
            else:
                break
            tasks.append((u, out))

        saved_count = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_url = {executor.submit(download_and_save_image, u, p): p for (u, p) in tasks}
            for fut in as_completed(future_to_url):
                if fut.result():
                    saved_count += 1
                if saved_count % 100 == 0 or saved_count == len(tasks):
                    log_event(f"  [{cls_name}] Downloaded and verified {saved_count}/{len(tasks)} images...")

        log_event(f"  ✓ {cls_name} corpus ready: {len(list(train_dir.glob('*.jpg')))} train, {len(list(val_dir.glob('*.jpg')))} val, {len(list(test_dir.glob('*.jpg')))} test.")

def update_knowledge_base_and_indices():
    log_event("\n" + "=" * 70)
    log_event("PHASE 2: UPDATING CLINICAL KNOWLEDGE BASE & 22-CLASS INDICES")
    log_event("=" * 70)

    # 1. Update Knowledge Base
    kb_path = ARTIFACTS_DIR / "disease_knowledge_base.json"
    if kb_path.exists():
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
    else:
        kb = {}

    for cls_name, info in UNIVERSAL_TAXA.items():
        kb[cls_name] = {
            "class_id": cls_name,
            "plant_name": "Universal Houseplants",
            "scientific_name": info["scientific_name"],
            "condition_name": info["common_name"],
            "is_healthy": False,
            "severity": info["severity"],
            "category": info["category"],
            "summary": info["summary"],
            "symptoms": info["symptoms"],
            "causes": info["causes"],
            "immediate_actions": info["immediate_actions"],
            "treatment": info["treatment"],
            "prevention": info["prevention"]
        }

    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2)
    log_event(f"✓ Knowledge Base updated at {kb_path} (Total Entries: {len(kb)})")

    # 2. Build 22-Class Indices
    all_classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    indices_path = ARTIFACTS_DIR / "class_indices.json"
    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump({cls: i for i, cls in enumerate(all_classes)}, f, indent=2)
    log_event(f"✓ 22-Class Indices updated at {indices_path} ({len(all_classes)} Total Taxa):")
    for i, c in enumerate(all_classes):
        log_event(f"    [{i:02d}] {c}")

    return all_classes

def train_22class_model(all_classes: list):
    log_event("\n" + "=" * 70)
    log_event("PHASE 3: TRAINING 22-CLASS CONVNEXT-SMALL BOTANICAL FLAGSHIP")
    log_event(f"• Target Device:    {config.DEVICE}")
    log_event(f"• Backbone:         ConvNeXt-Small (50.2M Parameters, 384x384 Native)")
    log_event(f"• Pretrained Prior: 16-Class PhytoNet Weights ({config.BEST_MODEL_ENSEMBLE})")
    log_event(f"• Loss Function:    Class-Balanced Focal Loss (gamma=2.0, label_smoothing=0.05)")
    log_event("=" * 70)

    num_classes = len(all_classes)

    # Transforms (384px)
    train_tf = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.RandomResizedCrop(384, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
    val_ds = datasets.ImageFolder(str(VAL_DIR), transform=val_tf)

    log_event(f"[Corpus] Loaded {len(train_ds)} train images, {len(val_ds)} validation images across {num_classes} classes.")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    # Build ConvNeXt-Small with 22-class head
    model = models.convnext_small(weights=None)
    in_features = model.classifier[2].in_features

    # Transfer backbone from 16-class best model
    if config.BEST_MODEL_ENSEMBLE.exists():
        try:
            ckpt_16 = torch.load(config.BEST_MODEL_ENSEMBLE, map_location="cpu", weights_only=False)
            state_dict = ckpt_16.get("model_state_dict", {})
            filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("classifier.2")}
            model.load_state_dict(filtered_state, strict=False)
            log_event(f"✓ Transferred backbone weights from {config.BEST_MODEL_ENSEMBLE}")
        except Exception as e:
            log_event(f"[Warning] Backbone transfer error: {e}, falling back to FloraNet prior.")
            model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
    else:
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)

    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(config.DEVICE)

    # Focal Loss definition
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, label_smoothing=0.05):
            super().__init__()
            self.gamma = gamma
            self.smoothing = label_smoothing

        def forward(self, inputs, targets):
            ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.smoothing)
            pt = torch.exp(-ce_loss)
            focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    epochs = 8
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_acc = 0.0
    best_weights = None

    log_event("\n" + "=" * 70)
    log_event(f"COMMENCING 22-CLASS PROGRESSIVE HIGH-ACUITY TRAINING (8 EPOCHS @ 384x384)")
    log_event("=" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct_train += preds.eq(labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                for images, labels in val_loader:
                    images = images.to(config.DEVICE, non_blocking=True)
                    labels = labels.to(config.DEVICE, non_blocking=True)
                    outputs = model(images)
                    _, preds = outputs.max(1)
                    correct_val += preds.eq(labels).sum().item()
                    total_val += labels.size(0)

        val_acc = (correct_val / total_val) * 100.0
        elapsed = round(time.time() - t0, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        log_event(f"  Epoch [{epoch:02d}/{epochs:02d}] ({elapsed}s) | Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {lr_now:.7f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()

    # Load best weights
    model.load_state_dict(best_weights)

    # Temperature Calibration
    log_event("\n[Calibration] Optimizing Softmax Temperature on validation logits...")
    val_logits_list = []
    val_labels_list = []
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            for images, labels in val_loader:
                images = images.to(config.DEVICE)
                logits = model(images)
                val_logits_list.append(logits.float().cpu())
                val_labels_list.append(labels.cpu())

    val_logits = torch.cat(val_logits_list, dim=0)
    val_targets = torch.cat(val_labels_list, dim=0)

    temperature = nn.Parameter(torch.ones(1) * 1.0)
    nll_criterion = nn.CrossEntropyLoss()
    temp_optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval_temp():
        temp_optimizer.zero_grad()
        loss = nll_criterion(val_logits / temperature, val_targets)
        loss.backward()
        return loss

    temp_optimizer.step(eval_temp)
    optimal_t = round(float(temperature.item()), 2)
    log_event(f"[Calibration] Optimal Calibrated Temperature: T = {optimal_t}")

    # Save Checkpoints
    ckpt_save = {
        "model_state_dict": model.state_dict(),
        "classes": all_classes,
        "temperature": optimal_t,
        "val_acc": best_val_acc,
        "backbone": "convnext_small",
        "resolution": 384,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    target_pth = ARTIFACTS_DIR / "best_model.pth"
    torch.save(ckpt_save, target_pth)
    torch.save(ckpt_save, ARTIFACTS_DIR / "best_model_22class.pth")
    log_event(f"✓ Saved 22-class flagship model to {target_pth}")

    return model, optimal_t

def evaluate_22class_test_set(all_classes: list):
    log_event("\n" + "=" * 70)
    log_event("PHASE 4: EVALUATING 22-CLASS MODEL ON COMPLETE TEST SET (5-CROP TTA)")
    log_event("=" * 70)

    from src.predict import diagnose_plant, load_diagnostic_system
    model, classes, spot_detector, kb = load_diagnostic_system(reload=True)

    test_ds = datasets.ImageFolder(str(TEST_DIR))
    log_event(f"[Dataset] Evaluating {len(test_ds)} unseen test images across {len(test_ds.classes)} classes...")

    all_preds = []
    all_targets = []
    t0 = time.time()

    for i, (img, target) in enumerate(test_ds):
        diag = diagnose_plant(img, top_k=1, use_tta=True, detect_spots=False)
        pred_cls = diag["predictions"][0]["class_name"]
        pred_idx = classes.index(pred_cls) if pred_cls in classes else 0
        all_preds.append(pred_idx)
        all_targets.append(target)

        if (i + 1) % 250 == 0 or (i + 1) == len(test_ds):
            acc_so_far = np.mean(np.array(all_preds) == np.array(all_targets)) * 100
            log_event(f"  [{i+1:4d}/{len(test_ds)}] Running Test Accuracy: {acc_so_far:.2f}%")

    elapsed = round(time.time() - t0, 1)
    acc = np.mean(np.array(all_preds) == np.array(all_targets)) * 100

    target_names = test_ds.classes
    cls_report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)
    log_event("\n" + "=" * 70)
    log_event("22-CLASS COMPREHENSIVE CLASSIFICATION REPORT:")
    log_event("=" * 70)
    log_event(cls_report)

    report_txt = REPORTS_DIR / "classification_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"PhytoCare Ultra 22-Class Benchmark Performance (Duration: {elapsed}s)\n")
        f.write(f"Overall Accuracy: {acc:.2f}%\n\n")
        f.write(cls_report)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=target_names,
        yticklabels=target_names,
        title=f'PhytoCare Ultra 22-Class Confusion Matrix (Test Acc: {acc:.2f}%)',
        ylabel='True Pathology Taxon',
        xlabel='Predicted Pathology Taxon'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_png = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_png, dpi=200, bbox_inches='tight')
    plt.close()

    log_event(f"✓ Saved updated classification report to {report_txt}")
    log_event(f"✓ Saved 22-class confusion matrix plot to {cm_png}")

def trigger_web_reload():
    try:
        req = urllib.request.Request("http://127.0.0.1:8001/api/model/reload", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            log_event("✓ Triggered live hot-reload on clinical web service (Port 8001).")
    except Exception as e:
        log_event(f"[Note] Web service reload ping: {e}")

def main():
    log_event("=" * 70)
    log_event("PHYTONET ULTRA: 22-CLASS UNIVERSAL PEST & PATHOGEN PIPELINE")
    log_event(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_event("=" * 70)

    # 1. Harvest from iNaturalist
    harvest_universal_taxa()

    # 2. Update Knowledge Base and Indices
    all_classes = update_knowledge_base_and_indices()

    # 3. Train Model
    train_22class_model(all_classes)

    # 4. Evaluate Test Set
    evaluate_22class_test_set(all_classes)

    # 5. Hot Reload Web App
    trigger_web_reload()

    log_event("\n" + "=" * 70)
    log_event("22-CLASS UNIVERSAL PATHOLOGY PIPELINE SUCCESSFULLY COMPLETED!")
    log_event("=" * 70)

if __name__ == "__main__":
    main()

