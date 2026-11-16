"""Audit existing datasets, Flutter taxonomy alignment, and available public HuggingFace/Ultralytics models."""
from pathlib import Path
import json
import re
import torch

CORETECH_ROOT = Path(r"c:\Users\User\Desktop\FYP\CoreTech")
APP_ROOT = Path(r"c:\Users\User\Desktop\FYP\AppIntegration\Plant-App-Flutter-UI")

dart_txt = (APP_ROOT / "lib" / "data" / "flora_250_data.dart").read_text(encoding="utf-8")
# Match rawName: '...' including escaped quotes
raw_names = re.findall(r"rawName:\s*'((?:\\.|[^'])*)'|rawName:\s*\"((?:\\.|[^\"])*)\"", dart_txt)
dart_species = [ (a or b).replace("\\'", "'") for a, b in raw_names ]

p250 = CORETECH_ROOT / "Plant_Identification_Experiment" / "data" / "train"
dirs_250 = sorted([d.name for d in p250.iterdir() if d.is_dir()])

print(f"Dart species count: {len(dart_species)}")
print(f"250-set train dirs: {len(dirs_250)}")
missing_in_250 = [s for s in dart_species if s not in dirs_250]
extra_in_250 = [d for d in dirs_250 if d not in dart_species]
print("In Dart but NOT in 250-set:", missing_in_250)
print("In 250-set but NOT in Dart:", extra_in_250)

# Check Plant_Disease_Experiment class distribution
print("\n--- Plant_Disease_Experiment real vs synthetic train counts ---")
pd_train = CORETECH_ROOT / "Plant_Disease_Experiment" / "data" / "train"
for d in sorted(pd_train.iterdir()):
    if not d.is_dir():
        continue
    imgs = list(d.glob("*.*"))
    aug = sum(1 for f in imgs if f.name.startswith("aug_") or "synth" in f.name.lower())
    active = sum(1 for f in imgs if f.name.startswith("active_"))
    print(f"  {d.name:36s}: total={len(imgs):4d} | aug={aug:3d} | active_leaked={active:3d} | clean_est={len(imgs)-aug-active:4d}")
