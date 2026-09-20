"""Cost Map and Planning Grid Generator for Model 4.

Translates 512x512 FloodNet 10-class semantic segmentation masks into
cost maps and 64x64 grid representations for UAV path planning with D* Lite.
"""

import numpy as np
import cv2
from typing import Tuple, Dict
from config.model_registry import FLOODNET_COSTS, FLOODNET_COLORS


def create_cost_map(mask: np.ndarray, cost_table: Dict[int, int] = FLOODNET_COSTS) -> np.ndarray:
    """Map 10-class segmentation mask (512x512) to integer traversal costs (512x512).

    Costs:
        0 Background         -> 100
        1 Building Flooded   -> 180
        2 Building Non-Flood -> 100
        3 Road Flooded       -> 200
        4 Road Non-Flooded   -> 1 (Optimal)
        5 Water              -> 150
        6 Tree               -> 20
        7 Vehicle            -> 60
        8 Pool               -> 150
        9 Grass              -> 20
    """
    cost_map = np.zeros(mask.shape, dtype=np.float32)
    for class_id, cost_val in cost_table.items():
        cost_map[mask == class_id] = cost_val
    return cost_map


def create_planning_grid(
    cost_map: np.ndarray,
    grid_size: Tuple[int, int] = (64, 64),
    obstacle_threshold: float = 140.0
) -> np.ndarray:
    """Downsample 512x512 cost map to 64x64 planning grid using safety-oriented pooling.

    Uses maximum cost pooling over each block to ensure small obstacles or flood zones
    are not smoothed out.
    """
    h, w = cost_map.shape[:2]
    gh, gw = grid_size
    bh, bw = h // gh, w // gw

    # Reshape and take max over local blocks for safety-critical obstacle detection
    reshaped = cost_map[:gh * bh, :gw * bw].reshape(gh, bh, gw, bw)
    grid = reshaped.max(axis=(1, 3))
    return grid.astype(np.float32)


def colorize_cost_map(cost_map: np.ndarray) -> np.ndarray:
    """Generate professional engineering color visualization of terrain cost map.

    LOW COST (1-30)     -> Green (Safe / Optimal traversal)
    MEDIUM COST (30-90) -> Yellow/Orange (Moderate difficulty)
    HIGH COST (90-180)  -> Deep Orange/Red (Hazardous terrain)
    BLOCKED (>=180)     -> Crimson / Dark Red (Impassable / Severe hazard)
    """
    norm_cost = np.clip(cost_map / 200.0, 0.0, 1.0)
    # Apply JET or TURBO colormap for thermal/hazard representation
    cost_u8 = (norm_cost * 255).astype(np.uint8)
    colored = cv2.applyColorMap(cost_u8, cv2.COLORMAP_TURBO)
    # Convert BGR from OpenCV to RGB for display
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def colorize_segmentation_mask(mask: np.ndarray) -> np.ndarray:
    """Convert integer class mask (512x512) to standard RGB palette."""
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in FLOODNET_COLORS.items():
        rgb[mask == class_id] = color
    return rgb
