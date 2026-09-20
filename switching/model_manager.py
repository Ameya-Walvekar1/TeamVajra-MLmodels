"""Model Manager for UAV AI Context-Switching System.

Handles lifecycle management, caching, device assignment (CPU/CUDA),
inference profiling, and active/standby state maintenance for all UAV models.
"""

from typing import Dict, Any, Optional, List
import time
import torch
from config.model_registry import ModelMetadata, get_model_registry
from inference.yolo_loader import YOLOInferenceEngine
from inference.model4_pipeline import Model4Pipeline


class ModelManager:
    """Manages model instances, hardware allocation, and live performance metrics."""

    def __init__(self):
        self.registry = get_model_registry()
        self.engines: Dict[str, Any] = {}
        self.active_model_id: str = "model6"
        self.device = "CUDA" if torch.cuda.is_available() else "CPU"
        self.perf_metrics: Dict[str, Any] = {
            "inference_time_ms": 0.0,
            "fps": 0.0,
            "frame_number": 0,
            "total_detections": 0,
            "active_model": "MODEL 5 — INFERENCE SWITCHER",
            "device": self.device,
        }

    def get_or_load_engine(self, model_id: str) -> Optional[Any]:
        """Lazy-load and cache the inference engine for the given model_id."""
        if model_id in self.engines:
            return self.engines[model_id]

        metadata = self.registry.get(model_id)
        if not metadata or not metadata.is_available:
            return None

        if model_id in ("model1", "model2", "model5"):
            engine = YOLOInferenceEngine(metadata)
            self.engines[model_id] = engine
            return engine
        elif model_id == "model4":
            engine = Model4Pipeline()
            self.engines[model_id] = engine
            return engine

        return None

    def set_active_model(self, model_id: str) -> bool:
        """Set the active model ID and update model statuses."""
        meta = self.registry.get(model_id)
        if not meta or not meta.is_available:
            return False

        self.active_model_id = model_id
        # Ensure engine is loaded
        self.get_or_load_engine(model_id)
        self.perf_metrics["active_model"] = meta.display_name
        return True

    def get_model_statuses(self) -> List[Dict[str, Any]]:
        """Return status for all 5 models (ACTIVE, STANDBY, TRAINING / UNAVAILABLE)."""
        statuses = []
        for meta in self.registry.list_all():
            if not meta.is_available:
                state = "TRAINING / UNAVAILABLE"
                css_badge = "status-training"
            elif meta.model_id == self.active_model_id:
                state = "ACTIVE"
                css_badge = "status-active"
            else:
                state = "READY / STANDBY"
                css_badge = "status-standby"

            statuses.append({
                "model_id": meta.model_id,
                "name": meta.display_name,
                "context": meta.context_name,
                "state": state,
                "css_badge": css_badge,
                "parameters": meta.parameters,
                "architecture": meta.architecture,
            })
        return statuses

    def run_active_inference(self, image: Any) -> Dict[str, Any]:
        """Run inference using currently active model engine and update telemetry."""
        engine = self.get_or_load_engine(self.active_model_id)
        if engine is None:
            return {"error": f"Engine for {self.active_model_id} unavailable"}

        start_t = time.perf_counter()

        if self.active_model_id in ("model1", "model2", "model5"):
            results = engine.run_inference(image)
            total_det = results.get("total_detections", 0)
        elif self.active_model_id == "model4":
            results = engine.execute_pipeline(image)
            total_det = results.get("path_length", 0)
        else:
            return {"error": "Unknown model type"}

        dur = time.perf_counter() - start_t
        fps = (1.0 / dur) if dur > 0 else 0.0

        self.perf_metrics["frame_number"] += 1
        self.perf_metrics["inference_time_ms"] = dur * 1000.0
        self.perf_metrics["fps"] = fps
        self.perf_metrics["total_detections"] = total_det
        self.perf_metrics["device"] = self.device

        return results


# Global singleton manager
GLOBAL_MODEL_MANAGER = ModelManager()


def get_model_manager() -> ModelManager:
    """Access the global model manager singleton."""
    return GLOBAL_MODEL_MANAGER
