"""YOLO Inference Engine for UAV Detection Models.

Supports:
- Model 1 (Visible Person Search) with internal 'item' -> 'PERSON' display mapping
- Model 2 (Fire / Smoke) with classes fire, smoke, other
- Model 5 (Thermal Person) with class person

Automatically utilizes GPU (CUDA) if available, with CPU fallback. Provides a
deterministic simulation mode when weights files are not present on disk to ensure
graceful fallback in test/secondary environments without crashing.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
from config.model_registry import ModelMetadata, get_model_registry


class YOLOInferenceEngine:
    """Wrapper for YOLO model loading and inference."""

    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self.model = None
        self.is_simulation_mode = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        weight_path = self.metadata.get_absolute_path()
        if weight_path.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(weight_path))
                self.is_simulation_mode = False
            except Exception as e:
                print(f"[WARN] Error loading weights from {weight_path}: {e}. Engaging simulation mode.")
                self.model = None
                self.is_simulation_mode = True
        else:
            # Fallback simulation mode when weights not on current test machine
            self.model = None
            self.is_simulation_mode = True

    def run_inference(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.45,
        frame_idx: int = 0,
    ) -> Dict[str, Any]:
        """Execute detection inference on input RGB image.

        Returns:
            Dict containing:
                - detections: List[Dict] with class, confidence, box [x1, y1, x2, y2]
                - annotated_image: np.ndarray (RGB)
                - total_detections: int
                - highest_confidence: float
                - average_confidence: float
                - inference_time_ms: float
                - device: str
                - is_simulation: bool
        """
        start_time = time.perf_counter()
        detections: List[Dict[str, Any]] = []

        if not self.is_simulation_mode and self.model is not None:
            # Real YOLO inference
            try:
                results = self.model(
                    image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=self.device,
                    verbose=False,
                )
                if results and len(results) > 0:
                    r = results[0]
                    boxes = r.boxes
                    for box in boxes:
                        cls_idx = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = [int(v) for v in box.xyxy[0].tolist()]

                        # Apply class mapping (specifically 'item' -> 'PERSON' for Model 1, 'person' -> 'THERMAL PERSON' for Model 5)
                        display_class = self.metadata.display_classes.get(
                            cls_idx, self.metadata.classes.get(cls_idx, f"class_{cls_idx}")
                        )
                        if self.metadata.model_id == "model1" and display_class.lower() == "item":
                            display_class = "PERSON"
                        elif self.metadata.model_id == "model5":
                            display_class = "THERMAL PERSON"

                        # Ignore non-hazard background/other classes
                        if display_class.lower() in ("other", "background"):
                            continue

                        detections.append({
                            "class": display_class,
                            "confidence": conf,
                            "box": xyxy,
                        })
            except Exception as e:
                print(f"[ERROR] Inference failed on model {self.metadata.model_id}: {e}")
                self.is_simulation_mode = True
                detections = self._generate_simulated_detections(image)
        else:
            detections = self._generate_simulated_detections(image)

        # For Model 2 (Fire / Smoke), if YOLO detections are sparse, augment with strict spectral fire/smoke detection
        if self.metadata.model_id == "model2" and len(detections) == 0:
            spectral_dets = self._detect_fire_smoke_spectral(image)
            detections.extend(spectral_dets)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Calculate statistics
        total = len(detections)
        confs = [d["confidence"] for d in detections]
        highest_conf = max(confs) if confs else 0.0
        avg_conf = (sum(confs) / total) if total > 0 else 0.0

        annotated = self._draw_detections(image, detections)

        return {
            "detections": detections,
            "annotated_image": annotated,
            "total_detections": total,
            "highest_confidence": highest_conf,
            "average_confidence": avg_conf,
            "inference_time_ms": elapsed_ms,
            "device": self.device.upper(),
            "is_simulation": self.is_simulation_mode,
        }

    def _detect_fire_smoke_spectral(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect active flame and smoke regions using high-precision spectral HSV signatures."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        detections = []

        # Precise Flame mask: Hue [0..22] or [168..180], Saturation >= 130, Value >= 140
        flame_mask = (
            ((hsv[:, :, 0] <= 22) | (hsv[:, :, 0] >= 168))
            & (hsv[:, :, 1] >= 130)
            & (hsv[:, :, 2] >= 140)
        ).astype(np.uint8)

        contours, _ = cv2.findContours(flame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > (h * w * 0.001):  # At least 0.1% of frame
                x, y, bw, bh = cv2.boundingRect(c)
                conf = min(0.96, max(0.65, 0.70 + (area / (h * w)) * 5.0))
                detections.append({
                    "class": "fire",
                    "confidence": float(conf),
                    "box": [int(x), int(y), int(x + bw), int(y + bh)],
                })

        # Precise Smoke mask: Saturation <= 35, Value [90..210], area > 1.5%
        smoke_mask = (
            (hsv[:, :, 1] <= 35)
            & (hsv[:, :, 2] >= 90)
            & (hsv[:, :, 2] <= 210)
        ).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        smoke_clean = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(smoke_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > (h * w * 0.015):  # Smoke plume region (at least 1.5% of frame)
                x, y, bw, bh = cv2.boundingRect(c)
                conf = min(0.92, max(0.60, 0.65 + (area / (h * w)) * 2.0))
                detections.append({
                    "class": "smoke",
                    "confidence": float(conf),
                    "box": [int(x), int(y), int(x + bw), int(y + bh)],
                })

        return detections

    def _generate_simulated_detections(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Deterministic simulation for demonstration when weights are not on local PC."""
        h, w = image.shape[:2]
        detections = []

        if self.metadata.model_id == "model1":
            # Visible Person Search: Detect realistic human coordinates
            detections.append({
                "class": "PERSON",
                "confidence": 0.91,
                "box": [int(w * 0.32), int(h * 0.40), int(w * 0.42), int(h * 0.65)],
            })
            detections.append({
                "class": "PERSON",
                "confidence": 0.84,
                "box": [int(w * 0.58), int(h * 0.45), int(w * 0.66), int(h * 0.72)],
            })
        elif self.metadata.model_id == "model2":
            # Fire / Smoke
            detections.append({
                "class": "fire",
                "confidence": 0.94,
                "box": [int(w * 0.45), int(h * 0.55), int(w * 0.62), int(h * 0.78)],
            })
            detections.append({
                "class": "smoke",
                "confidence": 0.88,
                "box": [int(w * 0.38), int(h * 0.25), int(w * 0.70), int(h * 0.58)],
            })
        elif self.metadata.model_id == "model5":
            # Thermal Person
            detections.append({
                "class": "person",
                "confidence": 0.93,
                "box": [int(w * 0.48), int(h * 0.38), int(w * 0.57), int(h * 0.68)],
            })

        return detections

    def _draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Render high-contrast tactical bounding boxes and labels with precise geometry alignment."""
        annotated = image.copy()
        h, w = annotated.shape[:2]

        for det in detections:
            x1_raw, y1_raw, x2_raw, y2_raw = det["box"]
            cls_name = det["class"]
            conf = det["confidence"]

            # Strict bounds clipping and coordinate alignment
            x1 = max(0, min(w - 1, int(x1_raw)))
            y1 = max(0, min(h - 1, int(y1_raw)))
            x2 = max(0, min(w - 1, int(x2_raw)))
            y2 = max(0, min(h - 1, int(y2_raw)))

            if x2 <= x1 or y2 <= y1:
                continue

            # Tactical color palette
            if cls_name.lower() in ("fire", "smoke"):
                color = (255, 60, 0) if cls_name.lower() == "fire" else (180, 180, 200)
            elif cls_name.lower() in ("person", "item"):
                color = (0, 230, 115) if self.metadata.model_id == "model1" else (255, 200, 0)
            else:
                color = (0, 200, 255)

            # Draw crisp aligned bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Tactical corner reticle accents for precise UI geometry
            corner_len = min(12, max(4, int(min(x2 - x1, y2 - y1) * 0.15)))
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, 3)

            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, 3)

            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, 3)

            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, 3)

            # Label banner
            label = f"{cls_name.upper()} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Position label box neatly above box if space permits, else inside top edge of box
            if y1 - th - 6 > 0:
                box_y1 = y1 - th - 6
                box_y2 = y1
                text_y = y1 - 4
            else:
                box_y1 = y1
                box_y2 = y1 + th + 6
                text_y = y1 + th + 2

            cv2.rectangle(annotated, (x1, box_y1), (min(w, x1 + tw + 8), box_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, text_y),
                font,
                font_scale,
                (10, 10, 10),
                thickness,
                cv2.LINE_AA,
            )

        return annotated
