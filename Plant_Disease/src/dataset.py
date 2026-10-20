"""
Hierarchical Dataset loading, botanical disease augmentations, and DataLoader pipelines.
Supports multi-task labels: Global Class (0..15), Species Index (0..4), and Condition Index.
"""

from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
try:
    from Plant_Disease.src import config
except (ImportError, ModuleNotFoundError):
    from src import config

def pil_rgb_loader(path: str) -> Image.Image:
    """Robust image loader ensuring 3-channel RGB output."""
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def get_transforms(image_size: int = config.IMAGE_SIZE):
    """Returns data transforms tailored for lesion-preserving plant disease diagnosis."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.04, 0.04),
            scale=(0.96, 1.04)
        ),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # 256 for 224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, val_test_transform

def parse_class_name(class_name: str):
    """
    Parses class folder name (e.g. 'Cactus_Dactylopius_Opuntia') into:
      species: 'Cactus'
      condition: 'Dactylopius_Opuntia'
    """
    for sp in config.SPECIES_NAMES:
        if class_name.startswith(sp):
            cond = class_name[len(sp):].lstrip("_")
            return sp, cond
    # Fallback
    parts = class_name.split("_", 1)
    return parts[0], parts[1] if len(parts) > 1 else "Unknown"

class HierarchicalImageDataset(Dataset):
    """
    Wraps standard ImageFolder to return:
      image: Tensor [3, 224, 224]
      global_label: int (0..15)
      species_label: int (0..4)
      condition_label: int (index within that species' condition list)
    """
    def __init__(self, root_dir, transform=None):
        self.base_dataset = datasets.ImageFolder(root=str(root_dir), loader=pil_rgb_loader)
        self.transform = transform
        self.classes = self.base_dataset.classes

        # Build mapping tables
        self.species_map = {sp: idx for idx, sp in enumerate(config.SPECIES_NAMES)}
        self.condition_maps = {
            sp: {cond: idx for idx, cond in enumerate(conds)}
            for sp, conds in config.PLANT_GROUPS.items()
        }

        self.sample_metadata = []
        for path, global_idx in self.base_dataset.samples:
            class_name = self.classes[global_idx]
            sp_name, cond_name = parse_class_name(class_name)
            sp_idx = self.species_map.get(sp_name, 0)
            cond_idx = self.condition_maps.get(sp_name, {}).get(cond_name, 0)

            self.sample_metadata.append({
                "path": path,
                "global_idx": global_idx,
                "species_idx": sp_idx,
                "species_name": sp_name,
                "condition_idx": cond_idx,
                "condition_name": cond_name,
                "is_healthy": "healthy" in cond_name.lower()
            })

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        path, global_target = self.base_dataset.samples[idx]
        img = self.base_dataset.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        meta = self.sample_metadata[idx]
        return img, meta["species_idx"], meta["condition_idx"], meta["global_idx"]

def compute_group_class_weights(dataset: HierarchicalImageDataset):
    """Computes class weights per plant group to handle intra-group imbalance."""
    group_weights = {}
    for sp_name, sp_idx in dataset.species_map.items():
        cond_counts = np.zeros(len(config.PLANT_GROUPS[sp_name]), dtype=np.float32)
        for m in dataset.sample_metadata:
            if m["species_idx"] == sp_idx:
                cond_counts[m["condition_idx"]] += 1.0

        # Avoid zero division
        cond_counts = np.maximum(cond_counts, 1.0)
        weights = cond_counts.sum() / (len(cond_counts) * cond_counts)
        # Smoothing
        smoothed = np.sqrt(weights)
        smoothed = smoothed / smoothed.mean()
        group_weights[sp_name] = torch.tensor(smoothed, dtype=torch.float32)

    return group_weights

def get_dataloaders(
    train_dir=config.TRAIN_DIR,
    val_dir=config.VAL_DIR,
    test_dir=config.TEST_DIR,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY
):
    """Builds PyTorch DataLoaders for Hierarchical Train, Val, and Test splits."""
    train_tf, eval_tf = get_transforms(config.IMAGE_SIZE)

    train_dataset = HierarchicalImageDataset(train_dir, transform=train_tf)
    val_dataset = HierarchicalImageDataset(val_dir, transform=eval_tf)
    test_dataset = HierarchicalImageDataset(test_dir, transform=eval_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, test_loader, train_dataset.classes
