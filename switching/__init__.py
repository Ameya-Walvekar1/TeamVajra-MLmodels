"""Switching package for UAV AI Context-Switching System."""
from .model_manager import ModelManager, get_model_manager
from .model_switcher import (
    ModelSwitcher,
    get_model_switcher,
    SwitchEvent,
    CONTEXT_MODEL_MAP,
    CONTEXT_REASONS,
    AUTO_SWITCH_SEQUENCE,
)

__all__ = [
    "ModelManager",
    "get_model_manager",
    "ModelSwitcher",
    "get_model_switcher",
    "SwitchEvent",
    "CONTEXT_MODEL_MAP",
    "CONTEXT_REASONS",
    "AUTO_SWITCH_SEQUENCE",
]
