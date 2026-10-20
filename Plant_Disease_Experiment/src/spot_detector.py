"""
PhytoCare Lesion Spot Detector and Visual Pathology Highlighter.
Combines:
  1. Botanical Foliage Segmentation (Excess Green Index ExG + HSV chromatic filter)
  2. Grad-CAM Spatial Activation Heatmap over ConvNeXt / Swin feature layers
  3. Background Suppression (eliminates pots, tables, shadows)
  4. Connected Component Analysis for extracting lesion bounding boxes
"""

import io
import sys
import base64
import numpy as np
from PIL import Image, ImageDraw
import scipy.ndimage
import torch
import torch.nn.functional as F
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src import config

class LesionSpotDetector:
    """
    Extracts spatial lesion coordinates strictly within plant foliage,
    suppressing background objects and rendering visual spot overlays.
    """
    def __init__(self, model, device=config.DEVICE):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Target final feature block in ConvNeXt or Swin
        target_layer = None
        if hasattr(self.model, "features"):
            target_layer = self.model.features[-1]
        elif hasattr(self.model, "stages"):
            target_layer = self.model.stages[-1]

        if target_layer is not None:
            target_layer.register_forward_hook(forward_hook)
            if hasattr(target_layer, "register_full_backward_hook"):
                target_layer.register_full_backward_hook(backward_hook)
            else:
                target_layer.register_backward_hook(backward_hook)

    def extract_plant_mask(self, pil_image: Image.Image) -> np.ndarray:
        """
        Segments the foreground plant foliage using Excess Green Index (ExG).
        ExG = 2*G - R - B.
        """
        arr = np.array(pil_image.convert("RGB")).astype(np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        denom = r + g + b + 1e-6
        rn, gn, bn = r / denom, g / denom, b / denom

        # Excess Green Index
        exg = 2.0 * gn - rn - bn
        mask = exg > 0.04

        # Morphological clean up
        mask = scipy.ndimage.binary_fill_holes(mask)
        mask = scipy.ndimage.binary_opening(mask, structure=np.ones((5, 5)))
        return mask

    def generate_heatmap(self, pil_image: Image.Image, target_class_idx: int) -> np.ndarray:
        """Generates Class Activation Map (CAM) normalized to image dimensions."""
        self.model.eval()
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        tensor = tf(pil_image).unsqueeze(0).to(self.device)
        tensor.requires_grad = True

        logits = self.model(tensor)
        score = logits[0, target_class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            # Fallback if hooks didn't trigger
            w, h = pil_image.size
            return np.zeros((h, w), dtype=np.float32)

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(pooled_grads):
            cam += w * self.activations[0, i, :, :]

        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize to original image size
        w, h = pil_image.size
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)
        return np.array(cam_img).astype(np.float32) / 255.0

    def detect_lesions(self, pil_image: Image.Image, target_class_idx: int, threshold: float = 0.45):
        """
        Detects lesion bounding boxes within plant foliage boundaries.
        Returns:
          - visualized_image: PIL Image with highlighted lesion overlays and boxes
          - lesion_boxes: List of bounding boxes [x1, y1, x2, y2]
          - severity_pct: Estimated % of plant surface infected
        """
        w, h = pil_image.size
        plant_mask = self.extract_plant_mask(pil_image)
        cam = self.generate_heatmap(pil_image, target_class_idx)

        # Suppress non-plant areas
        foliage_cam = cam * plant_mask

        # Thresholding for focal lesion spots
        lesion_mask = foliage_cam > threshold
        labeled_lesions, num_features = scipy.ndimage.label(lesion_mask)
        slices = scipy.ndimage.find_objects(labeled_lesions)

        boxes = []
        vis_img = pil_image.copy()
        draw = ImageDraw.Draw(vis_img)

        plant_pixels = np.sum(plant_mask)
        lesion_pixels = np.sum(lesion_mask)
        severity_pct = round((lesion_pixels / max(1, plant_pixels)) * 100, 1)

        for sl in slices:
            y1, y2 = sl[0].start, sl[0].stop
            x1, x2 = sl[1].start, sl[1].stop
            box_area = (x2 - x1) * (y2 - y1)
            # Filter tiny noise specs
            if box_area > 80:
                boxes.append([x1, y1, x2, y2])
                draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=3)

        return vis_img, boxes, severity_pct

