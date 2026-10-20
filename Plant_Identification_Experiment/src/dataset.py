"""
Enhanced Dataset Loader with RandAugment, RandomErasing, and Class Weighting.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from collections import Counter
import numpy as np

from src import config

def pil_rgb_loader(path: str) -> Image.Image:
    """Robust image loader ensuring 3-channel RGB output."""
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def get_transforms(image_size: int = config.IMAGE_SIZE, use_rand_augment: bool = config.USE_RAND_AUGMENT):
    """
    Returns enhanced data augmentation pipeline.
    Combines RandAugment and RandomErasing to force model to learn fine-grained leaf venation
    and resist real-world background variations and partial occlusions.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_ops = [
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
    ]

    if use_rand_augment:
        train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    else:
        train_ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))

    train_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value="random")
    ])

    train_transform = transforms.Compose(train_ops)

    val_test_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, val_test_transform

def compute_class_weights(train_dataset, num_classes: int = config.NUM_CLASSES):
    """
    Computes smoothed inverse class frequency weights to penalize minority class errors.
    Formula: weight_c = N / (C * count_c), clamped between [0.5, 4.0].
    """
    targets = [target for _, target in train_dataset.samples]
    counts = Counter(targets)
    total_samples = len(targets)
    
    weights = []
    for cls_idx in range(num_classes):
        cls_count = counts.get(cls_idx, 1)
        raw_weight = total_samples / (num_classes * cls_count)
        clamped_weight = float(np.clip(raw_weight, 0.5, 4.0))
        weights.append(clamped_weight)

    return torch.tensor(weights, dtype=torch.float32)

def get_dataloaders(
    train_dir=config.TRAIN_DIR,
    val_dir=config.VAL_DIR,
    test_dir=config.TEST_DIR,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY
):
    """Builds PyTorch DataLoaders with enhanced augmentation and class weights."""
    train_tf, eval_tf = get_transforms(config.IMAGE_SIZE)

    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=train_tf,
        loader=pil_rgb_loader
    )

    val_dataset = datasets.ImageFolder(
        root=str(val_dir),
        transform=eval_tf,
        loader=pil_rgb_loader
    )

    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=eval_tf,
        loader=pil_rgb_loader
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    class_weights = compute_class_weights(train_dataset)

    return train_loader, val_loader, test_loader, train_dataset.classes, class_weights

