"""
Spatial Lesion Spot Detector and Visual Highlighter with Plant Foliage Foreground Masking.
Uses:
  1. Botanical Foliage Segmentation (HSV chromatic ranges + Excess Green Index ExG)
  2. Morphological Hole-Filling (ensures internal lesions and pale bugs are enclosed)
  3. Grad-CAM Spatial Activation Heatmap for targeted disease pathology
  4. Background Suppression (completely eliminates false detections on tables, walls, pots, shadows)
  5. Connected Component Analysis for extracting precise lesion bounding boxes
"""

import io
import base64
import numpy as np
from PIL import Image, ImageDraw
import scipy.ndimage
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
CORETECH_ROOT = Path(__file__).resolve().parent.parent.parent
if str(CORETECH_ROOT) not in sys.path:
    sys.path.insert(0, str(CORETECH_ROOT))

try:
    from Plant_Disease.src import config
except (ImportError, ModuleNotFoundError):
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
        self._preprocessor = None
        self._register_hook()

    def _get_preprocessor(self):
        if self._preprocessor is None:
            try:
                from preprocessing.methods import PlantPreprocessors
                self._preprocessor = PlantPreprocessors()
            except Exception as e:
                print(f"Warning: could not load PlantPreprocessors in spot_detector: {e}")
        return self._preprocessor

    def _register_hook(self):
        def hook_fn(module, input, output):
            self.activations = output

        # Hook the final feature conv layer before global pooling
        self.model.features[-1].register_forward_hook(hook_fn)

    def extract_plant_mask(self, pil_image: Image.Image) -> np.ndarray:
        """
        Segments the foreground plant foliage using CoreTech Preprocessing Method 2
        (Zero-hole lamina enclosure & AI container/soil rejection).
        """
        preprocessor = self._get_preprocessor()
        if preprocessor is not None:
            try:
                res = preprocessor.process_for_disease_identification(
                    np.array(pil_image.convert("RGB")),
                    filter_pot_soil=True,
                    preserve_leaf_shape=True
                )
                mask = res["mask"]
                if mask is not None and np.count_nonzero(mask) > 0:
                    return (mask > 0).astype(np.float32)
            except Exception as e:
                print(f"Warning: neural matting fallback in spot_detector: {e}")

        # Fallback heuristic
        arr = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        exg = 2.0 * g - r - b
        foliage = (exg > 0.01)
        foliage_dilated = scipy.ndimage.binary_dilation(foliage, iterations=4)
        filled_plant = scipy.ndimage.binary_fill_holes(foliage_dilated)
        return filled_plant.astype(np.float32)

    def detect_and_highlight(
        self,
        pil_image: Image.Image,
        tensor: torch.Tensor,
        species_name: str,
        condition_name: str,
        condition_idx: int,
        is_healthy: bool,
        threshold: float = config.SPOT_ACTIVATION_THRESHOLD,
        plant_mask: np.ndarray = None
    ) -> dict:
        """
        Analyzes spatial activations, restricts detection strictly to the plant body,
        and outlines diseased spots while eliminating background noise.
        """
        orig_w, orig_h = pil_image.size
        annotated = pil_image.copy().convert("RGBA")

        # Use precomputed mask if supplied (from Method 2), or extract it
        if plant_mask is None:
            plant_mask = self.extract_plant_mask(pil_image)
        else:
            plant_mask = (plant_mask > 0).astype(np.float32)

        # If condition is explicitly Healthy:
        if is_healthy:
            status_msg = "No pathological spots or symptoms detected. Plant foliage is clean and healthy."
            base64_str = self._pil_to_base64(annotated.convert("RGB"))
            return {
                "has_spots": False,
                "spots_count": 0,
                "spot_boxes": [],
                "infected_area_pct": 0.0,
                "annotated_image": annotated.convert("RGB"),
                "base64_image": base64_str,
                "status_message": status_msg
            }

        # Disease condition: Compute Grad-CAM on active disease head
        tensor = tensor.to(self.device)
        tensor.requires_grad = True

        _, disease_outputs = self.model(tensor, target_species=species_name)
        logits = disease_outputs[species_name]
        target_score = logits[0, condition_idx]

        gradients = torch.autograd.grad(target_score, self.activations, retain_graph=True)[0]
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = F.relu(cam)

        cam_np = cam.detach().cpu().numpy()
        if cam_np.max() > 0:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)

        # Upsample Grad-CAM to 224x224
        cam_tensor = torch.tensor(cam_np).unsqueeze(0).unsqueeze(0)
        cam_upsampled = F.interpolate(cam_tensor, size=(224, 224), mode="bicubic", align_corners=False)
        cam_map = cam_upsampled.squeeze().numpy()
        cam_map = np.clip(cam_map, 0.0, 1.0)

        # Resize plant mask to 224x224 to filter out background activations
        plant_mask_tensor = torch.tensor(plant_mask).unsqueeze(0).unsqueeze(0)
        plant_mask_224 = F.interpolate(plant_mask_tensor, size=(224, 224), mode="nearest").squeeze().numpy()

        # APPLY PLANT FOREGROUND FILTER: Zero out any background activation on walls/tables/pots
        cam_map = cam_map * (plant_mask_224 > 0.3).astype(np.float32)

        # Threshold to segment lesion hotspots on the plant
        binary_mask = (cam_map > threshold).astype(np.uint8)
        labeled, num_features = scipy.ndimage.label(binary_mask)
        slices = scipy.ndimage.find_objects(labeled)

        spot_boxes = []
        scale_x = orig_w / 224.0
        scale_y = orig_h / 224.0

        overlay = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        total_spot_pixels = 0
        valid_spot_idx = 1

        for s in slices:
            y_start, y_stop = s[0].start, s[0].stop
            x_start, x_stop = s[1].start, s[1].stop
            area = (y_stop - y_start) * (x_stop - x_start)

            if area < config.MIN_SPOT_AREA_PIXELS:
                continue

            # Convert to original image coordinates
            box_x1 = int(x_start * scale_x)
            box_y1 = int(y_start * scale_y)
            box_x2 = int(x_stop * scale_x)
            box_y2 = int(y_stop * scale_y)
            box_w = max(10, box_x2 - box_x1)
            box_h = max(10, box_y2 - box_y1)

            # SECONDARY FOREGROUND VERIFICATION: Ensure spot lies primarily on plant foliage
            box_crop_mask = plant_mask[box_y1:box_y2, box_x1:box_x2]
            if box_crop_mask.size > 0 and box_crop_mask.mean() < 0.25:
                # Outside plant body (table, floor, or pot edge) -> Skip!
                continue

            total_spot_pixels += area

            spot_boxes.append({
                "spot_id": valid_spot_idx,
                "x": box_x1,
                "y": box_y1,
                "w": box_w,
                "h": box_h
            })

            # Draw translucent highlighted spot fill
            draw.rectangle(
                [box_x1, box_y1, box_x2, box_y2],
                fill=(239, 68, 68, 60),        # translucent coral-red tint
                outline=(220, 38, 38, 240),    # bold red border
                width=3
            )

            # Draw small spot badge
            draw.rectangle(
                [box_x1, max(0, box_y1 - 18), box_x1 + 48, box_y1],
                fill=(220, 38, 38, 220)
            )
            draw.text(
                (box_x1 + 4, max(0, box_y1 - 16)),
                f"Spot #{valid_spot_idx}",
                fill=(255, 255, 255, 255)
            )

            valid_spot_idx += 1

        # Blend highlight overlay
        annotated = Image.alpha_composite(annotated, overlay).convert("RGB")
        base64_str = self._pil_to_base64(annotated)

        spots_count = len(spot_boxes)
        plant_area_pixels = max(1, (plant_mask_224 > 0.3).sum())
        raw_pct = (total_spot_pixels / plant_area_pixels) * 100
        infected_area_pct = min(100.0, round(raw_pct, 1))

        if spots_count > 0:
            status_msg = f"Detected {spots_count} unhealthy spot cluster(s) covering approximately {infected_area_pct}% of plant surface."
        else:
            status_msg = f"No localized spot clusters detected on foliage. Plant appears clean."

        return {
            "has_spots": spots_count > 0,
            "spots_count": spots_count,
            "spot_boxes": spot_boxes,
            "infected_area_pct": infected_area_pct,
            "annotated_image": annotated,
            "base64_image": base64_str,
            "status_message": status_msg
        }

    def _pil_to_base64(self, img: Image.Image) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
