"""
Dataset and Loss Module for PhytoCare Pathology Engine
Implements Class-Balanced Focal Loss and Lesion-Preserving Data Augmentations.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config

def get_pathology_transforms(img_size: int = 384):
    """
    Lesion-preserving transformations for botanical pathology.
    Preserves diagnostic color signatures (rust pustules vs yellow chlorosis).
    """
    train_tf = transforms.Compose([
        transforms.Resize((int(img_size * 1.08), int(img_size * 1.08))),
        transforms.RandomResizedCrop(img_size, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((int(img_size * 1.08), int(img_size * 1.08))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_tf, eval_tf

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss based on Effective Number of Samples (Cui et al., CVPR 2019).
    Penalizes overconfidence on dominant classes while amplifying gradients on rare lesions.
    """
    def __init__(self, samples_per_class: list, beta: float = 0.999, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        effective_num = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(samples_per_class, dtype=torch.float32))
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(samples_per_class)
        self.register_buffer("weights", weights)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        # One-hot encoding with label smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Focal term: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - probs, self.gamma)
        class_weights = self.weights.to(logits.device)
        sample_weights = class_weights[targets].unsqueeze(1)

        loss = -sample_weights * focal_weight * smooth_targets * log_probs
        return loss.sum(dim=-1).mean()

