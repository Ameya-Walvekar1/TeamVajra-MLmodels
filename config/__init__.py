"""Configuration package for UAV AI Context-Switching System."""
from .model_registry import (
    ModelMetadata,
    ModelRegistry,
    MODEL_REGISTRY,
    get_model_registry,
    FLOODNET_CLASSES,
    FLOODNET_COSTS,
)

__all__ = [
    "ModelMetadata",
    "ModelRegistry",
    "MODEL_REGISTRY",
    "get_model_registry",
    "FLOODNET_CLASSES",
    "FLOODNET_COSTS",
]
