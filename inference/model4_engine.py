"""Model 4 Engine: LRASPP MobileNetV3 Large 10-Class Semantic Segmentation.

Performs:
    Image (512x512)
    -> Semantic Segmentation (10 x 512 x 512)
    -> 512x512 Segmentation Mask
    -> Terrain Cost Map (512x512)
    -> 64x64 Planning Grid
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
from config.model_registry import (
    ModelMetadata,
    FLOODNET_CLASSES,
    FLOODNET_COSTS,
    FLOODNET_COLORS,
    get_model_registry,
)
from inference.cost_map import (
    create_cost_map,
    create_planning_grid,
    colorize_cost_map,
    colorize_segmentation_mask,
)


class Model4Engine:
    """LRASPP MobileNetV3 Large FloodNet segmentation engine."""

    def __init__(self, metadata: Optional[ModelMetadata] = None):
        if metadata is None:
            metadata = get_model_registry().get("model4")
        self.metadata = metadata
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.is_simulation_mode = False
        self._load_model()

    def _load_model(self):
        weight_path = self.metadata.get_absolute_path() if self.metadata else Path("models/terrain/best_floodnet.pth")
        if weight_path.exists():
            try:
                import torchvision.models.segmentation as seg
                # Construct LRASPP MobileNetV3 Large with 10 classes
                self.model = seg.lraspp_mobilenet_v3_large(num_classes=10)
                try:
                    state_dict = torch.load(str(weight_path), map_location=self.device, weights_only=False)
                except TypeError:
                    state_dict = torch.load(str(weight_path), map_location=self.device)
                # Handle possible 'model' or 'state_dict' key in checkpoint
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif isinstance(state_dict, dict) and "model" in state_dict:
                    state_dict = state_dict["model"]
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                self.is_simulation_mode = False
            except Exception as e:
                print(f"[WARN] Failed to load FloodNet model from {weight_path}: {e}. Fallback to simulated mask.")
                self.model = None
                self.is_simulation_mode = True
        else:
            self.model = None
            self.is_simulation_mode = True

    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Execute segmentation and generate cost map and planning grid.

        Input: image (H, W, 3) RGB.
        Returns:
            Dict containing:
                - original_512: np.ndarray (512, 512, 3)
                - mask: np.ndarray (512, 512) int32
                - cost_map: np.ndarray (512, 512) float32
                - planning_grid: np.ndarray (64, 64) float32
                - colored_mask: np.ndarray (512, 512, 3)
                - colored_cost_map: np.ndarray (512, 512, 3)
                - segmentation_overlay: np.ndarray (512, 512, 3)
                - inference_time_ms: float
                - device: str
        """
        start_time = time.perf_counter()
        orig_512 = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

        if not self.is_simulation_mode and self.model is not None:
            try:
                # Preprocess: Normalize image
                img_t = torch.from_numpy(orig_512).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_t = (img_t - mean) / std
                img_t = img_t.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(img_t)["out"]  # shape (1, 10, 512, 512)
                    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.int32)
            except Exception as e:
                print(f"[ERROR] Segmentation inference error: {e}")
                pred_mask = self._generate_simulated_mask(orig_512)
        else:
            pred_mask = self._generate_simulated_mask(orig_512)

        cost_map = create_cost_map(pred_mask)
        planning_grid = create_planning_grid(cost_map, grid_size=(64, 64))

        colored_mask = colorize_segmentation_mask(pred_mask)
        colored_cost_map = colorize_cost_map(cost_map)

        # Alpha-blended overlay (60% image + 40% mask)
        overlay = cv2.addWeighted(orig_512, 0.6, colored_mask, 0.4, 0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "original_512": orig_512,
            "mask": pred_mask,
            "cost_map": cost_map,
            "planning_grid": planning_grid,
            "colored_mask": colored_mask,
            "colored_cost_map": colored_cost_map,
            "segmentation_overlay": overlay,
            "inference_time_ms": elapsed_ms,
            "device": self.device.upper(),
            "is_simulation": self.is_simulation_mode,
        }

    def _generate_simulated_mask(self, image_512: np.ndarray) -> np.ndarray:
        """Create realistic FloodNet scene mask for demonstration testing."""
        h, w = 512, 512
        mask = np.full((h, w), 9, dtype=np.int32)  # Grass background

        # Non-flooded road diagonally across the scene (Class 4: Road Non-Flooded)
        for r in range(h):
            c_center = int(40 + (w - 80) * (r / h))
            c_start = max(0, c_center - 24)
            c_end = min(w, c_center + 24)
            mask[r, c_start:c_end] = 4

        # Water/Flooding on the right section (Class 5: Water)
        rr, cc = np.ogrid[:h, :w]
        water_region = ((rr - 160)**2 + (cc - 420)**2) < (110**2)
        mask[water_region] = 5

        # Flooded section on the road near water (Class 3: Road Flooded)
        flood_road_region = (rr > 120) & (rr < 220) & (mask == 4)
        mask[flood_road_region] = 3

        # Non-flooded buildings on the left (Class 2: Building Non-Flooded)
        mask[60:150, 40:120] = 2
        mask[240:340, 60:160] = 2

        # Flooded building adjacent to water (Class 1: Building Flooded)
        mask[180:260, 360:440] = 1

        # Trees scattered around (Class 6: Tree)
        mask[380:480, 200:300] = 6
        mask[40:100, 250:320] = 6

        # Vehicles on road (Class 7: Vehicle)
        mask[300:320, 260:290] = 7

        return mask
