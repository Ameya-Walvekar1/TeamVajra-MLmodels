"""Video Stream Processing and Context-Switching Timeline Tracker.

Processes uploaded UAV video (.mp4, .avi, .mov, .mkv), annotates frames with
detections, active model badges, and telemetry watermarks, and generates a synchronized
context switching event timeline.
"""

import os
import time
import tempfile
import warnings
from typing import Dict, Any, List, Optional, Tuple, Callable
import cv2
import numpy as np

warnings.filterwarnings("ignore")

from config.model_registry import get_model_registry
from switching.model_manager import get_model_manager
from switching.model_switcher import get_model_switcher, CONTEXT_MODEL_MAP


class VideoProcessor:
    """Processes video frames with context switching and visual annotation."""

    def __init__(self):
        self.manager = get_model_manager()
        self.switcher = get_model_switcher()
        self.registry = get_model_registry()

    def process_video_stream(
        self,
        input_path: str,
        output_path: str,
        mode: str = "INFERENCE_DRIVEN",
        target_model_id: Optional[str] = None,
        video_name: Optional[str] = None,
        switch_interval_sec: float = 0.0,
        initial_context: str = "Visible Person Search",
        max_duration_sec: Optional[float] = 30.0,
        conf_threshold: float = 0.15,
        progress_callback: Optional[Callable[[float, str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Process video frame-by-frame applying context switching and returning full detection telemetry."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Standardize working size for optimal performance and H.264 macroblock alignment
        target_w, target_h = 960, 544
        max_frames = int(max_duration_sec * fps) if max_duration_sec else total_frames
        frames_to_process = min(total_frames, max_frames) if total_frames > 0 else 900

        # Universal H.264 MP4 writer (YUV420p) for 100% Linux, Windows, macOS & browser playback
        is_imageio = False
        writer = None

        try:
            import imageio
            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=8,
            )
            is_imageio = True
        except Exception:
            try:
                import imageio_ffmpeg
                import imageio
                writer = imageio.get_writer(
                    output_path,
                    fps=fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=8,
                )
                is_imageio = True
            except Exception:
                fourcc_options = [
                    cv2.VideoWriter_fourcc(*"avc1"),
                    cv2.VideoWriter_fourcc(*"H264"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                ]
                for fourcc in fourcc_options:
                    w_test = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
                    if w_test.isOpened():
                        writer = w_test
                        break
                if writer is None or not writer.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
                is_imageio = False

        timeline_events = []
        contexts_cycle = [
            "Visible Person Search",
            "Fire / Smoke",
            "Terrain / Navigation",
            "Thermal Search",
        ]

        effective_name = video_name or input_path

        current_ctx = initial_context
        if target_model_id and target_model_id in ("model1", "model2", "model4", "model5"):
            meta = self.registry.get(target_model_id)
            if meta:
                current_ctx = meta.context_name
        elif not target_model_id:
            # Auto-infer initial context from first frame and video metadata
            ret_init, frame_init = cap.read()
            if ret_init:
                f_init_rgb = cv2.cvtColor(cv2.resize(frame_init, (target_w, target_h)), cv2.COLOR_BGR2RGB)
                inferred_ctx, _, _ = self.switcher.classifier.classify_scene(f_init_rgb, video_path=effective_name)
                current_ctx = inferred_ctx
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Log initial timeline event showing Model 5 Context Switcher active at start
        timeline_events.append({
            "timestamp": "00:00.0",
            "model_name": "MODEL 5 — INFERENCE SWITCHER",
            "context_name": "Inference-Driven Context Switching",
            "transition": "INIT -> MODEL 5 — INFERENCE SWITCHER",
        })

        if target_model_id and target_model_id in ("model1", "model2", "model4", "model5"):
            self.switcher.switch_context(current_ctx, mode="MANUAL")
            curr_model_id = target_model_id
            timeline_events.append({
                "timestamp": "00:00.0",
                "model_name": self.registry.get(curr_model_id).display_name if self.registry.get(curr_model_id) else curr_model_id,
                "context_name": current_ctx,
                "transition": f"MODEL 5 — INFERENCE SWITCHER -> {self.registry.get(curr_model_id).display_name if self.registry.get(curr_model_id) else curr_model_id}",
                "reason": "Manual target model override assigned",
            })
        else:
            self.switcher.switch_context(current_ctx, mode="INFERENCE_DRIVEN")
            curr_model_id = self.switcher.active_model_id

        frame_idx = 0
        start_proc_time = time.perf_counter()

        all_detections: List[Dict[str, Any]] = []
        latest_results: Optional[Dict[str, Any]] = None
        latest_model4_results: Optional[Dict[str, Any]] = None
        latest_model4_pipeline: Optional[Any] = None
        latest_annotated_rgb: Optional[np.ndarray] = None

        while cap.isOpened() and frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = frame_idx / fps
            frame_resized = cv2.resize(frame, (target_w, target_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Execute Inference Switching Model on frame (if in auto mode and no forced model)
            if not target_model_id and frame_idx % 6 == 0 and mode == "INFERENCE_DRIVEN":
                pred_ctx, did_switch, switch_event = self.switcher.infer_and_switch(frame_rgb, video_path=effective_name)
                if did_switch and switch_event:
                    current_ctx = pred_ctx
                    curr_model_id = self.switcher.active_model_id
                    curr_model_meta = self.registry.get(curr_model_id)

                    mins = int(current_sec // 60)
                    secs = current_sec % 60
                    ts_str = f"{mins:02d}:{secs:04.1f}"

                    timeline_events.append({
                        "timestamp": ts_str,
                        "model_name": curr_model_meta.display_name,
                        "context_name": current_ctx,
                        "transition": f"{switch_event.previous_model_name} -> {curr_model_meta.display_name}",
                        "reason": switch_event.reason,
                    })

            # Run inference with active model
            annotated_frame_bgr, frame_results, m4_results, m4_pipeline = self._process_frame_for_context(
                frame_rgb=frame_rgb,
                frame_bgr=frame_resized,
                context_name=current_ctx,
                model_id=curr_model_id,
                current_sec=current_sec,
                frame_idx=frame_idx,
                conf_threshold=conf_threshold,
            )

            if frame_results and "detections" in frame_results:
                latest_results = frame_results
                for d in frame_results["detections"]:
                    all_detections.append({
                        **d,
                        "frame_idx": frame_idx,
                        "timestamp_sec": current_sec,
                    })
            if m4_results:
                latest_model4_results = m4_results
                latest_model4_pipeline = m4_pipeline

            annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
            if is_imageio:
                writer.append_data(annotated_frame_rgb)
            else:
                writer.write(annotated_frame_bgr)

            frame_idx += 1
            if progress_callback and (frame_idx % 4 == 0 or frame_idx == frames_to_process):
                progress = frame_idx / max(1, frames_to_process)
                status_msg = f"Processing video stream: frame {frame_idx}/{frames_to_process} [{current_ctx}]"
                progress_callback(progress, status_msg)

        cap.release()
        if is_imageio:
            writer.close()
        else:
            writer.release()
            self._ensure_browser_h264_compatible(output_path, fps)

        total_elapsed = time.perf_counter() - start_proc_time
        effective_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0.0

        confs = [d["confidence"] for d in all_detections] if all_detections else []
        highest_conf = max(confs) if confs else 0.0
        avg_conf = (sum(confs) / len(confs)) if confs else 0.0

        return {
            "output_path": output_path,
            "total_frames_processed": frame_idx,
            "duration_sec": frame_idx / fps,
            "effective_fps": effective_fps,
            "timeline": timeline_events,
            "all_detections": all_detections,
            "highest_confidence": highest_conf,
            "average_confidence": avg_conf,
            "latest_results": latest_results,
            "latest_model4_results": latest_model4_results,
            "latest_model4_pipeline": latest_model4_pipeline,
            "active_model_id": curr_model_id,
        }

    def _process_frame_for_context(
        self,
        frame_rgb: np.ndarray,
        frame_bgr: np.ndarray,
        context_name: str,
        model_id: str,
        current_sec: float,
        frame_idx: int,
        conf_threshold: float = 0.15,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Any]]:
        """Run context-specific inference and composite HUD overlay."""
        model_meta = self.registry.get(model_id)
        out_bgr = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        frame_results = None
        m4_results = None
        m4_pipeline = None

        if model_id in ("model1", "model2", "model5"):
            engine = self.manager.get_or_load_engine(model_id)
            if engine:
                frame_results = engine.run_inference(frame_rgb, conf_threshold=conf_threshold, frame_idx=frame_idx)
                # Convert annotated RGB back to BGR for video writer
                out_bgr = cv2.cvtColor(frame_results["annotated_image"], cv2.COLOR_RGB2BGR)
        elif model_id == "model4":
            # Terrain segmentation
            engine = self.manager.get_or_load_engine("model4")
            if engine:
                m4_pipeline = engine
                m4_results = engine.execute_pipeline(frame_rgb)
                overlay = m4_results["annotated_image"]
                overlay_resized = cv2.resize(overlay, (w, h))
                out_bgr = cv2.cvtColor(overlay_resized, cv2.COLOR_RGB2BGR)

        # Composite Mission Control HUD Telemetry Bar
        display_name = model_meta.display_name if model_meta else model_id
        self._render_hud_overlay(out_bgr, display_name, context_name, current_sec, frame_idx)

        return out_bgr, frame_results, m4_results, m4_pipeline

    def _render_hud_overlay(
        self,
        img: np.ndarray,
        model_name: str,
        context_name: str,
        current_sec: float,
        frame_idx: int,
    ):
        """Draw tactical telemetry overlay atop video frame."""
        h, w = img.shape[:2]

        # Top banner background (semi-transparent dark)
        banner_h = 36
        banner = img[:banner_h, :].copy()
        cv2.rectangle(banner, (0, 0), (w, banner_h), (12, 16, 24), -1)
        cv2.addWeighted(banner, 0.85, img[:banner_h, :], 0.15, 0, img[:banner_h, :])

        # Border line
        cv2.line(img, (0, banner_h), (w, banner_h), (0, 229, 255), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        mins = int(current_sec // 60)
        secs = current_sec % 60
        time_str = f"T+{mins:02d}:{secs:04.1f} | FRM #{frame_idx}"

        # Context badge
        cv2.putText(img, f"CTX: {context_name.upper()}", (14, 23), font, 0.48, (0, 229, 255), 1, cv2.LINE_AA)

        # Model badge
        cv2.putText(img, f"ACTIVE: {model_name}", (w // 3, 23), font, 0.48, (0, 230, 115), 1, cv2.LINE_AA)

        # Telemetry timer
        cv2.putText(img, time_str, (w - 240, 23), font, 0.46, (200, 210, 220), 1, cv2.LINE_AA)

    def _ensure_browser_h264_compatible(self, video_path: str, fps: float):
        """Re-encode MP4 video to baseline H.264 (yuv420p) if OpenCV created an mp4v stream."""
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            temp_h264 = video_path + ".h264.mp4"
            import subprocess
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i", video_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                temp_h264,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(temp_h264) and os.path.getsize(temp_h264) > 0:
                os.replace(temp_h264, video_path)
        except Exception:
            pass
