"""
Dataset loading and data augmentation pipelines for plant identification.
"""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from src import config

def pil_rgb_loader(path: str) -> Image.Image:
    """Robust image loader ensuring 3-channel RGB output."""
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def get_transforms(image_size: int = config.IMAGE_SIZE):
    """
    Returns data transforms tailored for botanical/leaf photography.
    Applies photometric and geometric augmentations to resist lighting and angle shifts.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0),
            ratio=(0.8, 1.25)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # e.g., 256 for 224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, val_test_transform

def get_dataloaders(
    train_dir=config.TRAIN_DIR,
    val_dir=config.VAL_DIR,
    test_dir=config.TEST_DIR,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY
):
    """Builds PyTorch DataLoaders for Train, Val, and Test splits."""
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

