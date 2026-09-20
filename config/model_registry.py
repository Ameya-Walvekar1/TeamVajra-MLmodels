"""Model Registry for UAV AI Context-Switching System.

Centralized configuration of model metadata, checkpoint paths, target classes,
FloodNet cost definitions, and lifecycle states. Model 3 is marked as
TRAINING / UNAVAILABLE and provides an API to dynamically register weights
when training is completed without altering the GUI codebase.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List


# Resolve project root dynamically (parent of ai/ or current working directory)
CURRENT_FILE = Path(__file__).resolve()
AI_DIR = CURRENT_FILE.parents[1]
PROJECT_ROOT = AI_DIR.parent


FLOODNET_CLASSES: Dict[int, str] = {
    0: "Background",
    1: "Building Flooded",
    2: "Building Non-Flooded",
    3: "Road Flooded",
    4: "Road Non-Flooded",
    5: "Water",
    6: "Tree",
    7: "Vehicle",
    8: "Pool",
    9: "Grass",
}

FLOODNET_COSTS: Dict[int, int] = {
    0: 100,   # Background
    1: 180,   # Building Flooded
    2: 100,   # Building Non-Flooded
    3: 200,   # Road Flooded
    4: 1,     # Road Non-Flooded (optimal path)
    5: 150,   # Water
    6: 20,    # Tree
    7: 60,    # Vehicle
    8: 150,   # Pool
    9: 20,    # Grass
}

FLOODNET_COLORS: Dict[int, List[int]] = {
    0: [30, 30, 30],      # Background (Dark Grey)
    1: [180, 50, 50],     # Building Flooded (Crimson)
    2: [120, 120, 140],   # Building Non-Flooded (Steel Blue)
    3: [220, 100, 0],     # Road Flooded (Deep Orange)
    4: [40, 200, 80],     # Road Non-Flooded (Vibrant Green)
    5: [30, 120, 220],    # Water (Cyan-Blue)
    6: [20, 140, 50],     # Tree (Forest Green)
    7: [240, 220, 40],    # Vehicle (Yellow)
    8: [0, 200, 220],     # Pool (Aqua)
    9: [80, 200, 120],    # Grass (Light Green)
}


@dataclass
class ModelMetadata:
    """Metadata container for registered UAV models."""
    model_id: str
    display_name: str
    context_name: str
    task_type: str
    relative_path: str
    parameters: Optional[int] = None
    classes: Dict[int, str] = field(default_factory=dict)
    display_classes: Dict[int, str] = field(default_factory=dict)
    purpose: str = ""
    is_available: bool = True
    status_label: str = "READY"
    architecture: str = ""
    input_size: Optional[tuple] = None
    output_size: Optional[tuple] = None
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

    def get_absolute_path(self) -> Path:
        """Resolve path relative to project root, ~/SIH1, cwd, or fallback."""
        candidates = [
            PROJECT_ROOT / self.relative_path,
            Path.cwd() / self.relative_path,
            Path.home() / "SIH1" / self.relative_path,
            AI_DIR / self.relative_path.replace("ai/", ""),
            Path(self.relative_path),
        ]
        for p in candidates:
            if p.exists():
                return p.resolve()
        # Default to PROJECT_ROOT path if not yet present
        return (PROJECT_ROOT / self.relative_path).resolve()

    def weights_exist(self) -> bool:
        """Check if weight file exists on disk."""
        if not self.is_available:
            return False
        return self.get_absolute_path().exists()


class ModelRegistry:
    """Central registry tracking all AI models in the UAV context-switching system."""

    def __init__(self):
        self._registry: Dict[str, ModelMetadata] = {}
        self._initialize_default_registry()

    def _initialize_default_registry(self):
        # MODEL 1 — VISIBLE HUMAN DETECTION
        self._registry["model1"] = ModelMetadata(
            model_id="model1",
            display_name="MODEL 1 — VISIBLE HUMAN DETECTION",
            context_name="Visible Person Search",
            task_type="detect",
            relative_path="models/person/best.pt",
            parameters=2590035,
            classes={0: "item"},
            # Model 1 internally classifies as 'item', GUI must display 'PERSON'
            display_classes={0: "PERSON"},
            purpose="Visible human/person detection, Search and rescue, Aerial/disaster imagery",
            is_available=True,
            status_label="READY",
            architecture="YOLO Object Detection",
        )

        # MODEL 2 — FIRE & SMOKE HAZARD DETECTION
        self._registry["model2"] = ModelMetadata(
            model_id="model2",
            display_name="MODEL 2 — FIRE & SMOKE HAZARD DETECTION",
            context_name="Fire / Smoke",
            task_type="detect",
            relative_path="models/fire_smoke/best.pt",
            parameters=2590425,
            classes={0: "fire", 1: "smoke", 2: "other"},
            display_classes={0: "fire", 1: "smoke", 2: "other"},
            purpose="Fire detection, Smoke detection, Fire/hazard scene analysis",
            is_available=True,
            status_label="READY",
            architecture="YOLO Object Detection",
        )

        # MODEL 3 — TERRAIN SEGMENTATION & NAVIGATION (formerly Model 4)
        self._registry["model4"] = ModelMetadata(
            model_id="model4",
            display_name="MODEL 3 — TERRAIN SEGMENTATION & NAVIGATION",
            context_name="Terrain / Navigation",
            task_type="segmentation_and_planning",
            relative_path="models/terrain/best_floodnet.pth",
            parameters=1870000,
            classes=FLOODNET_CLASSES,
            display_classes=FLOODNET_CLASSES,
            purpose="FloodNet 10-class semantic segmentation, cost map generation, D* Lite path planning",
            is_available=True,
            status_label="READY",
            architecture="LRASPP MobileNetV3 Large (10 Classes)",
            input_size=(512, 512),
            output_size=(10, 512, 512),
            custom_attributes={
                "planning_grid_size": (64, 64),
                "cost_mapping": FLOODNET_COSTS,
            },
        )

        # MODEL 4 — THERMAL HUMAN DETECTION (formerly Model 5)
        self._registry["model5"] = ModelMetadata(
            model_id="model5",
            display_name="MODEL 4 — THERMAL HUMAN DETECTION",
            context_name="Thermal Search",
            task_type="detect",
            relative_path="models/thermal/best.pt",
            parameters=3011043,
            classes={0: "person"},
            display_classes={0: "person"},
            purpose="Thermal human detection, Low-visibility/night search, Thermal UAV imagery",
            is_available=True,
            status_label="READY",
            architecture="YOLO Thermal Detection",
        )

        # MODEL 5 — INFERENCE CONTEXT SWITCHER (Main Model 6)
        self._registry["model6"] = ModelMetadata(
            model_id="model6",
            display_name="MODEL 5 — INFERENCE SWITCHER",
            context_name="Inference-Driven Context Switching",
            task_type="classify",
            relative_path="models/classifier/best.pt",
            parameters=1450000,
            classes={
                0: "Visible Person Search",
                1: "Fire / Smoke",
                2: "Terrain / Navigation",
                3: "Thermal Search",
            },
            display_classes={
                0: "Visible Person Search",
                1: "Fire / Smoke",
                2: "Terrain / Navigation",
                3: "Thermal Search",
            },
            purpose="Real-time scene feature analysis and autonomous model context switching",
            is_available=True,
            status_label="ACTIVE / OPERATIONAL",
            architecture="Context Classification Network",
        )

    def get(self, model_id: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata by model_id."""
        return self._registry.get(model_id)

    def get_by_context(self, context_name: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata matching context name."""
        for meta in self._registry.values():
            if meta.context_name.lower() == context_name.lower():
                return meta
        return None

    def list_all(self) -> List[ModelMetadata]:
        """Return all registered models in order."""
        order = ["model1", "model2", "model4", "model5", "model6"]
        return [self._registry[k] for k in order if k in self._registry]


# Global registry instance
MODEL_REGISTRY = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Retrieve global model registry instance."""
    return MODEL_REGISTRY
