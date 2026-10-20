import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image

try:
    from .normalization import (
        apply_white_balance,
        apply_clahe,
        clean_mask_components,
        composite_background
    )
except ImportError:
    from preprocessing.normalization import (
        apply_white_balance,
        apply_clahe,
        clean_mask_components,
        composite_background
    )

YOLO_WEIGHTS_PATH = Path(__file__).resolve().parent / "yolov8n-seg.pt"

class PlantPreprocessors:
    """
    Dedicated Preprocessing Pipelines:
    Method 1: Tailored for Plant Species Identification (Morphology & Margin Serrations)
    Method 2: Tailored for Disease & Health Diagnosis (Lesion Texture & Color Fidelity)
    
    Includes:
    - BiRefNet & U2Net neural matting engines
    - Multi-stage AI (YOLO) + Geometric Pot & Soil rejection
    - Complete Leaf Shape Preservation (apex tip & margin protection)
    - Dynamic safety padding so leaves are never clipped
    """

    def __init__(self):
        self._sessions = {}
        self._yolo_model = None
        self._mask_cache = {}

    def _get_session(self, model_name: str = "birefnet-general"):
        """Cached rembg sessions."""
        if model_name not in self._sessions:
            try:
                from rembg import new_session
                self._sessions[model_name] = new_session(model_name)
            except Exception as e:
                print(f"Warning: could not load {model_name}, falling back to u2net: {e}")
                if "u2net" not in self._sessions:
                    try:
                        from rembg import new_session
                        self._sessions["u2net"] = new_session("u2net")
                    except Exception as err:
                        print(f"Warning: could not load u2net either: {err}")
                        self._sessions["u2net"] = None
                return self._sessions["u2net"]
        return self._sessions[model_name]

    def _get_yolo(self):
        if self._yolo_model is None:
            if YOLO_WEIGHTS_PATH.exists():
                try:
                    from ultralytics import YOLO
                    self._yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))
                except Exception as e:
                    print(f"Warning: could not load YOLO model from {YOLO_WEIGHTS_PATH}: {e}")
                    self._yolo_model = None
            else:
                print(f"Warning: YOLO weights not found at {YOLO_WEIGHTS_PATH}")
                self._yolo_model = None
        return self._yolo_model

    def get_raw_segmentation_mask(
        self, 
        img_rgb: np.ndarray, 
        model_name: str = "birefnet-general",
        filter_pot_soil: bool = True,
        preserve_leaf_shape: bool = True,
        foliage_scope: str = "salient_focus"
    ) -> np.ndarray:
        """
        Extracts foreground plant mask with complete leaf shape preservation and spatial cohesion.
        foliage_scope:
          - 'salient_focus' (default): Prunes distant stray fragments and tightly focuses on the primary leaves.
          - 'full_canopy': Expands across contiguous vegetation to capture the entire multi-leaf thicket/shrub.
        """
        cache_key = (
            img_rgb.shape, 
            int(img_rgb.sum()), 
            model_name, 
            filter_pot_soil, 
            preserve_leaf_shape,
            foliage_scope
        )
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key].copy()
            
        session = self._get_session(model_name)
        pil_img = Image.fromarray(img_rgb)
        binary_mask = None
        
        if session is not None:
            try:
                from rembg import remove
                mask_pil = remove(
                    pil_img,
                    session=session,
                    only_mask=True,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=230,
                    alpha_matting_background_threshold=15
                )
                mask_np = np.array(mask_pil)
                if len(mask_np.shape) == 3:
                    mask_np = mask_np[:, :, 0]
                _, binary_mask = cv2.threshold(mask_np, 60, 255, cv2.THRESH_BINARY)
            except Exception as e:
                print(f"Error in deep matting with {model_name}: {e}")
                binary_mask = None

        if binary_mask is None:
            # High-quality botanical color-space threshold fallback (HSV + Excess Green)
            hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            b, g, r = cv2.split(img_rgb.astype(np.float32))
            exg = 2.0 * g - r - b
            is_green = (h >= 18) & (h <= 105) & (s >= 18) & (v >= 20)
            is_yellow = (h >= 12) & (h < 25) & (s >= 25) & (v >= 25)
            is_brown = (h >= 4) & (h <= 28) & (s >= 15) & (v >= 20)
            veg = (is_green | is_yellow | is_brown | (exg > 2.0))
            binary_mask = veg.astype(np.uint8) * 255

        # Retain all significant plant leaves (clean isolated speckle noise)
        binary_mask = self.retain_all_plant_leaves(binary_mask, min_component_ratio=0.010)

        # Leaf shape preservation: gradient-guided morphological watershed across shaded foliage
        if preserve_leaf_shape:
            binary_mask = self.restore_full_leaf_morphology(img_rgb, binary_mask)

        # Apply multi-layer Pot and Soil rejection
        if filter_pot_soil:
            binary_mask = self.filter_pot_and_soil(img_rgb, binary_mask)

        # Final cleanup ensuring all leaves are kept
        binary_mask = self.retain_all_plant_leaves(binary_mask, min_component_ratio=0.010)

        # Apply Foliage Scope processing:
        if foliage_scope == "full_canopy":
            binary_mask = self.expand_to_full_canopy(img_rgb, binary_mask)
        else:
            binary_mask = self.prune_disconnected_stray_fragments(binary_mask)

        self._mask_cache[cache_key] = binary_mask.copy()
        return binary_mask

    def restore_full_leaf_morphology(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Restores complete leaf shape for leaves truncated by cast shadows or complex lighting.
        Uses gradient-guided morphological watershed across the foliage domain.
        """
        if mask is None or np.count_nonzero(mask) == 0:
            return mask

        h, w = img_rgb.shape[:2]

        # 1. Edge and gradient map
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_u8 = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        grad_bgr = cv2.cvtColor(grad_u8, cv2.COLOR_GRAY2BGR)

        # 2. Comprehensive botanical tissue detector
        b, g, r = cv2.split(img_rgb.astype(np.float32))
        exg = 2.0 * g - r - b
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_ch, _, _ = cv2.split(lab)

        is_green = (h_ch >= 18) & (h_ch <= 105) & (s_ch >= 18) & (v_ch >= 20)
        is_yellow = (h_ch >= 12) & (h_ch < 25) & (s_ch >= 25) & (v_ch >= 25)
        is_wilt_brown = (h_ch >= 4) & (h_ch <= 28) & (s_ch >= 15) & (v_ch >= 20)
        is_variegated = (l_ch > 50) & (s_ch >= 10) & (s_ch < 60) & (h_ch >= 10) & (h_ch <= 105)
        is_purple_red = ((h_ch <= 12) | (h_ch >= 155)) & (s_ch >= 30) & (v_ch >= 25)
        is_veg = (exg > 0.5) | is_green | is_yellow | is_wilt_brown | is_variegated | is_purple_red | (mask > 0)

        is_white_neutral_bg = (s_ch < 15) & (v_ch > 100)

        # 3. Sure foreground: core of detected plant
        kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        sure_fg = cv2.erode(mask, kernel_fg, iterations=1)

        # 4. Sure background: non-vegetative outside the plant mask
        is_non_veg = (~is_veg) | is_white_neutral_bg | (v_ch < 20)
        search_band = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)
        sure_bg = (search_band == 0) | ((mask == 0) & is_non_veg)

        # 5. Watershed refinement
        markers = np.zeros((h, w), dtype=np.int32)
        markers[sure_bg > 0] = 2
        markers[sure_fg > 0] = 1

        ws_markers = markers.copy()
        try:
            cv2.watershed(grad_bgr, ws_markers)
            refined_mask = (ws_markers == 1).astype(np.uint8) * 255
            refined_mask = cv2.bitwise_and(refined_mask, search_band)
            refined_mask = cv2.bitwise_or(refined_mask, mask)
        except Exception:
            refined_mask = mask

        # 6. Smooth micro-gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel_close)

        return refined_mask

    def retain_all_plant_leaves(self, mask: np.ndarray, min_component_ratio: float = 0.010) -> np.ndarray:
        """
        Preserves all leaves and stems of a plant, removing only tiny speckle noise.
        """
        if mask is None or np.count_nonzero(mask) == 0:
            return mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return mask

        total_fg = np.count_nonzero(mask)
        clean_mask = np.zeros_like(mask)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > (total_fg * min_component_ratio):
                clean_mask[labels == i] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        return clean_mask

    def prune_disconnected_stray_fragments(self, mask: np.ndarray, max_distance_ratio: float = 0.35) -> np.ndarray:
        """
        Spatial Cohesion Pruning:
        Finds the primary plant subject cluster and prunes distant, disconnected stray fragments.
        """
        if mask is None or np.count_nonzero(mask) == 0:
            return mask

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 2:
            return mask

        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = np.argmax(areas) + 1
        max_area = stats[max_idx, cv2.CC_STAT_AREA]

        main_box = stats[max_idx]
        main_x = main_box[cv2.CC_STAT_LEFT]
        main_y = main_box[cv2.CC_STAT_TOP]
        main_w = main_box[cv2.CC_STAT_WIDTH]
        main_h = main_box[cv2.CC_STAT_HEIGHT]

        clean_mask = np.zeros_like(mask)
        clean_mask[labels == max_idx] = 255

        for i in range(1, num_labels):
            if i == max_idx:
                continue
            cx, cy = centroids[i]
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            comp_x = stats[i, cv2.CC_STAT_LEFT]
            comp_y = stats[i, cv2.CC_STAT_TOP]
            area = stats[i, cv2.CC_STAT_AREA]

            dx = max(0, max(main_x - (comp_x + comp_w), comp_x - (main_x + main_w)))
            dy = max(0, max(main_y - (comp_y + comp_h), comp_y - (main_y + main_h)))
            edge_dist = np.sqrt(dx**2 + dy**2)

            is_close_neighbor = edge_dist <= (max(main_w, main_h) * max_distance_ratio)
            is_significant = area > (max_area * 0.04)

            if is_close_neighbor and is_significant:
                clean_mask[labels == i] = 255

        return clean_mask

    def expand_to_full_canopy(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Full Bush / Canopy Expansion:
        Expands the segmentation mask across the entire contiguous vegetative cluster.
        """
        if mask is None or np.count_nonzero(mask) == 0:
            return mask

        h, w = img_rgb.shape[:2]

        b, g, r = cv2.split(img_rgb.astype(np.float32))
        exg = 2.0 * g - r - b
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        is_green = (h_ch >= 18) & (h_ch <= 105) & (s_ch >= 20) & (v_ch >= 20)
        is_yellow = (h_ch >= 12) & (h_ch < 25) & (s_ch >= 25) & (v_ch >= 25)
        is_wilt_brown = (h_ch >= 4) & (h_ch <= 28) & (s_ch >= 18) & (v_ch >= 20)
        is_veg = (is_green | is_yellow | is_wilt_brown | (exg > 2.0))

        seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        dilated_seed = cv2.dilate(mask, seed_kernel, iterations=2)

        canopy_candidates = is_veg.astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(canopy_candidates)

        canopy_mask = mask.copy()
        for i in range(1, num_labels):
            comp = (labels == i)
            if np.any(comp & (dilated_seed > 0)):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > 150:
                    canopy_mask[comp] = 255

        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        canopy_mask = cv2.morphologyEx(canopy_mask, cv2.MORPH_CLOSE, close_k)
        return canopy_mask

    def filter_pot_and_soil(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Multi-Stage Pot & Soil Rejection:
        1. AI-Driven Container Detection (YOLOv8 instance segmentation of vase/pot/bowl)
        2. Geometric Base & Rim Analysis (absorbs soil line and pot rim)
        3. Color/Index base pruning (removes terracotta/soil without touching foliage lesions)
        """
        if mask is None or np.count_nonzero(mask) == 0:
            return mask

        h, w = img_rgb.shape[:2]
        clean_plant_mask = mask.copy()

        # STAGE 1: AI Object-Level Pot/Vase Detection
        yolo_pot_mask = np.zeros((h, w), dtype=np.uint8)
        yolo = self._get_yolo()
        if yolo is not None:
            try:
                results = yolo(img_rgb, verbose=False, conf=0.20)
                if len(results) > 0 and results[0].masks is not None:
                    r = results[0]
                    target_classes = {"vase", "bowl", "cup", "potted plant"}
                    for idx, (cls_idx, conf) in enumerate(zip(r.boxes.cls, r.boxes.conf)):
                        cname = yolo.names[int(cls_idx)]
                        if cname in target_classes and cname != "potted plant":
                            m = r.masks.data[idx].cpu().numpy()
                            resized_m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                            yolo_pot_mask = np.maximum(yolo_pot_mask, (resized_m > 0.45).astype(np.uint8) * 255)
            except Exception as e:
                print(f"YOLO container detection warning: {e}")

        if np.count_nonzero(yolo_pot_mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 15))
            dilated_pot = cv2.dilate(yolo_pot_mask, kernel, iterations=1)
            clean_plant_mask = cv2.bitwise_and(clean_plant_mask, cv2.bitwise_not(dilated_pot))
            clean_plant_mask = self.retain_all_plant_leaves(clean_plant_mask, min_component_ratio=0.010)
            return clean_plant_mask

        # STAGE 2: Geometric & Vegetation Base Pruning
        b, g, r = cv2.split(img_rgb.astype(np.float32))
        exg = 2.0 * g - r - b
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_ch, _, _ = cv2.split(lab)

        is_green = (h_ch >= 18) & (h_ch <= 105) & (s_ch >= 18) & (v_ch >= 20)
        is_yellow_chlorosis = (h_ch >= 12) & (h_ch < 25) & (s_ch >= 30) & (v_ch >= 30)
        is_variegated_white = (l_ch > 55) & (s_ch < 45) & (v_ch > 55)
        is_vegetative = (is_green | is_yellow_chlorosis | is_variegated_white | (exg > 4)) & (clean_plant_mask > 0)

        non_veg = (clean_plant_mask > 0) & (~is_vegetative)
        non_veg_u8 = non_veg.astype(np.uint8) * 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_veg_u8)
        if num_labels > 1:
            veg_y_indices, _ = np.where(is_vegetative)
            if len(veg_y_indices) > 0:
                min_veg_y = np.min(veg_y_indices)
                max_veg_y = np.max(veg_y_indices)
                veg_height = max_veg_y - min_veg_y
                total_fg_area = np.count_nonzero(clean_plant_mask)

                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    comp_cy = centroids[i][1]
                    comp_top = stats[i, cv2.CC_STAT_TOP]
                    comp_h = stats[i, cv2.CC_STAT_HEIGHT]
                    comp_mask = (labels == i)

                    if area > (total_fg_area * 0.018):
                        is_at_base = comp_cy > (min_veg_y + 0.38 * veg_height)
                        extends_to_bottom = (comp_top + comp_h) >= (max_veg_y - 0.05 * veg_height)
                        if is_at_base and extends_to_bottom:
                            clean_plant_mask[comp_mask] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_plant_mask = cv2.morphologyEx(clean_plant_mask, cv2.MORPH_OPEN, kernel)
        return clean_plant_mask

    # =========================================================================
    # METHOD 1: Best for Plant Species Identification
    # =========================================================================
    def process_for_species_identification(
        self, 
        img_rgb: np.ndarray,
        bg_color: Tuple[int, int, int] = (128, 128, 128),
        model_name: str = "birefnet-general",
        filter_pot_soil: bool = True,
        preserve_leaf_shape: bool = True,
        padding_ratio: float = 0.12,
        foliage_scope: str = "salient_focus"
    ) -> Dict[str, Any]:
        """
        Optimized for SPECIES IDENTIFICATION:
        - Full leaf shape preservation (preserves apex, serrations, petiole).
        - Aspect-ratio preserving square letterbox with safe padding.
        - Neutral background compositing (128, 128, 128).
        """
        mask = self.get_raw_segmentation_mask(
            img_rgb, 
            model_name=model_name, 
            filter_pot_soil=filter_pot_soil,
            preserve_leaf_shape=preserve_leaf_shape,
            foliage_scope=foliage_scope
        )

        composited = composite_background(img_rgb, mask, bg_style="Neutral Gray (128)", custom_color=bg_color)
        cropped_plant, bbox = self._crop_with_square_padding(composited, mask, padding_ratio=padding_ratio, bg_color=bg_color)

        return {
            "method_name": "Method 1: Species Identification",
            "cropped_image": cropped_plant,
            "mask": mask,
            "bbox": bbox
        }

    # =========================================================================
    # METHOD 2: Best for Disease & Health Condition Identification
    # =========================================================================
    def process_for_disease_identification(
        self, 
        img_rgb: np.ndarray,
        bg_color: Tuple[int, int, int] = (128, 128, 128),
        model_name: str = "birefnet-general",
        filter_pot_soil: bool = True,
        preserve_leaf_shape: bool = True,
        padding_ratio: float = 0.08,
        foliage_scope: str = "salient_focus"
    ) -> Dict[str, Any]:
        """
        Optimized for DISEASE & HEALTH IDENTIFICATION:
        - Full leaf lamina enclosure (guarantees necrotic/chlorotic areas are retained).
        - Zero-Hole Solid Lamina (seals small lesion holes < 700px, preserves natural fenestrations).
        - Selective foreground-only CLAHE contrast enhancement for lesion/spore textures.
        - Tight symptomatic crop with safe padding.
        """
        base_mask = self.get_raw_segmentation_mask(
            img_rgb, 
            model_name=model_name, 
            filter_pot_soil=filter_pot_soil,
            preserve_leaf_shape=preserve_leaf_shape,
            foliage_scope=foliage_scope
        )

        # Lesion hole sealing: keeps necrotic holes < 700px inside leaf, keeps natural leaf fenestrations
        solid_mask = base_mask.copy()
        inv_mask = cv2.bitwise_not(base_mask)
        h_labels, h_lbls, h_stats, _ = cv2.connectedComponentsWithStats(inv_mask)
        h, w = base_mask.shape[:2]
        for i in range(1, h_labels):
            x = h_stats[i, cv2.CC_STAT_LEFT]
            y = h_stats[i, cv2.CC_STAT_TOP]
            bw = h_stats[i, cv2.CC_STAT_WIDTH]
            bh = h_stats[i, cv2.CC_STAT_HEIGHT]
            area = h_stats[i, cv2.CC_STAT_AREA]
            touches_edge = (x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h)
            if not touches_edge and area < 700:
                solid_mask[h_lbls == i] = 255

        # Selective foreground CLAHE
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        enhanced_bgr = apply_clahe(img_bgr, clip_limit=2.2, tile_grid_size=(8, 8))
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        leaf_enhanced = cv2.addWeighted(img_rgb, 0.60, enhanced_rgb, 0.40, 0)

        composited = composite_background(leaf_enhanced, solid_mask, bg_style="Neutral Gray (128)", custom_color=bg_color)
        cropped_plant, bbox = self._crop_tight_bbox(composited, solid_mask, padding_ratio=padding_ratio)

        return {
            "method_name": "Method 2: Disease & Health Diagnosis",
            "cropped_image": cropped_plant,
            "mask": solid_mask,
            "bbox": bbox
        }

    def _crop_with_square_padding(
        self, 
        img: np.ndarray, 
        mask: np.ndarray, 
        padding_ratio: float = 0.12,
        bg_color: Tuple[int, int, int] = (128, 128, 128)
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Crops to plant and pads into a clean square canvas with safe margin."""
        y_indices, x_indices = np.where(mask > 0)
        h, w = img.shape[:2]
        if len(y_indices) == 0:
            return img, (0, 0, w, h)

        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)

        pw = x_max - x_min
        ph = y_max - y_min
        pad = int(max(pw, ph) * padding_ratio)

        x1 = max(0, x_min - pad)
        y1 = max(0, y_min - pad)
        x2 = min(w, x_max + pad)
        y2 = min(h, y_max + pad)

        cropped = img[y1:y2, x1:x2]
        ch, cw = cropped.shape[:2]

        max_dim = max(ch, cw)
        square_canvas = np.full((max_dim, max_dim, 3), bg_color, dtype=np.uint8)
        off_x = (max_dim - cw) // 2
        off_y = (max_dim - ch) // 2
        square_canvas[off_y:off_y+ch, off_x:off_x+cw] = cropped

        return square_canvas, (x1, y1, x2, y2)

    def _crop_tight_bbox(
        self, 
        img: np.ndarray, 
        mask: np.ndarray, 
        padding_ratio: float = 0.08
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Tight crop around plant with safe margin to prevent clipping leaf tips."""
        y_indices, x_indices = np.where(mask > 0)
        h, w = img.shape[:2]
        if len(y_indices) == 0:
            return img, (0, 0, w, h)

        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)

        pw = x_max - x_min
        ph = y_max - y_min
        pad_x = int(pw * padding_ratio)
        pad_y = int(ph * padding_ratio)

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        cropped = img[y1:y2, x1:x2]
        return cropped, (x1, y1, x2, y2)

