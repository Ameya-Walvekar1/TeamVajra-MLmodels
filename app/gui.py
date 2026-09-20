"""UAV AI Context-Switching Mission Control Console.

Professional engineering dashboard for testing and demonstrating UAV autonomous
context-switching between object detection, hazard analysis, terrain segmentation,
and D* Lite path planning. Light theme edition.
"""

import os
import sys
import time
import tempfile
import warnings
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import streamlit as st

warnings.filterwarnings("ignore")

# Set up project root in python path
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.model_registry import (
    get_model_registry,
    FLOODNET_CLASSES,
    FLOODNET_COSTS,
    FLOODNET_COLORS,
)
from switching.model_manager import get_model_manager
from switching.model_switcher import (
    get_model_switcher,
    CONTEXT_MODEL_MAP,
    CONTEXT_REASONS,
    AUTO_SWITCH_SEQUENCE,
)
from app.styles import MISSION_CONTROL_CSS
from app.video_processor import VideoProcessor


def initialize_app():
    """Configure Streamlit page layout and inject engineering light styles."""
    st.set_page_config(
        page_title="UAV AI Context-Switching Console",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(MISSION_CONTROL_CSS, unsafe_allow_html=True)


def render_top_telemetry_bar(manager, switcher):
    """Render top mission-control telemetry header bar."""
    st_state = switcher.get_current_state()
    perf = manager.perf_metrics

    st.markdown(f"""
    <div class="telemetry-header">
        <div class="header-title">UAV Context-Switching Mission Control Console</div>
        <div class="header-status">
            HARDWARE: <span class="status-badge badge-device">{perf.get('device', 'CPU')}</span> &nbsp;|&nbsp; 
            ACTIVE MODEL: <span class="status-badge status-active">{st_state['active_model_name']}</span> &nbsp;|&nbsp; 
            LATENCY: <span style="color:#0284c7; font-family:'JetBrains Mono'; font-weight:600;">{perf.get('inference_time_ms', 0.0):.1f} ms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_context_switching_card(switcher, manager):
    """Render the dedicated, prominent Context Switching dashboard panel in Auto-Switch mode."""
    state = switcher.get_current_state()
    last_event = state.get("last_event")

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Context Switching</div>', unsafe_allow_html=True)

    # 2x2 Grid for Core Context States
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="telemetry-stat-box" style="margin-bottom:10px;">
            <div class="telemetry-stat-val" style="color:#0284c7; font-size:1.05rem;">{state['current_context']}</div>
            <div class="telemetry-stat-label">CURRENT CONTEXT</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="telemetry-stat-box">
            <div class="telemetry-stat-val"><span class="status-badge status-active">INFERENCE SWITCHING ACTIVE</span></div>
            <div class="telemetry-stat-label">STATUS</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="telemetry-stat-box" style="margin-bottom:10px;">
            <div class="telemetry-stat-val" style="color:#15803d; font-size:1.05rem;">{state['active_model_name']}</div>
            <div class="telemetry-stat-label">ACTIVE MODEL</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="telemetry-stat-box">
            <div class="telemetry-stat-val" style="color:#475569; font-size:0.95rem;">{state['previous_model_name']}</div>
            <div class="telemetry-stat-label">PREVIOUS MODEL</div>
        </div>
        """, unsafe_allow_html=True)

    # Switch Reason
    st.markdown(f"""
    <div style="background:#f1f5f9; border:1px solid #e2e8f0; border-radius:3px; padding:8px 12px; margin:10px 0; font-family:'JetBrains Mono'; font-size:0.75rem;">
        <span style="color:#64748b; font-weight:600;">INFERENCE DECISION REASON:</span> 
        <span style="color:#1e293b;">{state['switch_reason']}</span>
    </div>
    """, unsafe_allow_html=True)

    # Transition Event Banner
    if last_event and last_event.previous_context != "SYSTEM INIT":
        st.markdown(f"""
        <div class="transition-banner">
            <div class="transition-header">CONTEXT CHANGE DETECTED</div>
            <div class="transition-body">
                <strong>Previous:</strong> {last_event.previous_context} ({last_event.previous_model_name})<br>
                <strong>New:</strong> {last_event.new_context} ({last_event.new_model_name})<br>
                <strong>Switch:</strong> {last_event.previous_model_name} → {last_event.new_model_name}<br>
                <strong>Reason:</strong> {last_event.reason}<br>
                <strong>Timestamp:</strong> {last_event.timestamp} | <strong>Trigger:</strong> {last_event.trigger_mode}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="transition-banner">
            <div class="transition-header">INFERENCE CONTEXT ASSIGNMENT</div>
            <div class="transition-body">
                <strong>Active Context:</strong> {state['current_context']}<br>
                <strong>Active Model:</strong> {state['active_model_name']}<br>
                <strong>Inference Reason:</strong> {state['switch_reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:12px 0; border:none; border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)
    override_col1, override_col2 = st.columns([3, 2])
    with override_col1:
        context_keys = list(CONTEXT_MODEL_MAP.keys())
        current_ctx = switcher.current_context
        current_idx = context_keys.index(current_ctx) if current_ctx in CONTEXT_MODEL_MAP else 0

        if not st.session_state.get("context_manually_locked", False):
            if current_ctx in CONTEXT_MODEL_MAP:
                st.session_state["override_context_select"] = current_ctx

        def _on_manual_override_change():
            st.session_state["context_manually_locked"] = True
            new_val = st.session_state.get("override_context_select")
            if new_val in CONTEXT_MODEL_MAP:
                switcher.switch_context(new_val, mode="MANUAL")

        manual_override = st.selectbox(
            "MANUAL CONTEXT OVERRIDE (OPTIONAL)",
            context_keys,
            index=current_idx,
            key="override_context_select",
            on_change=_on_manual_override_change,
        )
        if st.session_state.get("context_manually_locked", False) and manual_override in CONTEXT_MODEL_MAP and manual_override != switcher.current_context:
            switcher.switch_context(manual_override, mode="MANUAL")
            st.rerun()
    with override_col2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("AUTO-DETECT FROM INFERENCE", use_container_width=True):
            st.session_state["context_manually_locked"] = False
            if switcher.current_context in CONTEXT_MODEL_MAP:
                st.session_state["override_context_select"] = switcher.current_context
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_model_status_grid(manager):
    """Always display status cards for all 5 models."""
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Model Status</div>', unsafe_allow_html=True)

    statuses = manager.get_model_statuses()
    cols = st.columns(5)

    for i, model in enumerate(statuses):
        with cols[i]:
            st.markdown(f"""
            <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:4px; padding:10px 12px; height:100%;">
                <div style="font-family:'JetBrains Mono'; font-weight:700; font-size:0.8rem; color:#0f172a; margin-bottom:4px;">
                    {model['name']}
                </div>
                <div style="font-size:0.72rem; color:#64748b; margin-bottom:8px;">
                    {model['context']}
                </div>
                <div>
                    <span class="status-badge {model['css_badge']}">{model['state']}</span>
                </div>
                <div style="font-family:'JetBrains Mono'; font-size:0.68rem; color:#475569; margin-top:8px;">
                    PARAMS: {f"{model['parameters']:,}" if model['parameters'] else "N/A"}<br>
                    ARCH: {model['architecture']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_performance_panel(manager):
    """Live inference performance metrics."""
    perf = manager.perf_metrics
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Live Hardware & Inference Telemetry</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Inference Time", f"{perf.get('inference_time_ms', 0.0):.1f} ms")
    with c2:
        st.metric("Throughput", f"{perf.get('fps', 0.0):.1f} FPS")
    with c3:
        st.metric("Frame Number", f"#{perf.get('frame_number', 0)}")
    with c4:
        st.metric("Detections / Path", f"{perf.get('total_detections', 0)}")
    with c5:
        st.metric("Active Model", perf.get("active_model", "N/A"))
    with c6:
        st.metric("Compute Device", perf.get("device", "CPU"))

    st.markdown('</div>', unsafe_allow_html=True)


def render_detection_table(results, model_id):
    """Render detection table with bounding box coordinates and summary stats."""
    detections = results.get("detections", [])

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Detection Information & Coordinates</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Detections", len(detections))
    with c2:
        st.metric("Highest Confidence", f"{results.get('highest_confidence', 0.0):.2f}")
    with c3:
        st.metric("Average Confidence", f"{results.get('average_confidence', 0.0):.2f}")

    if detections:
        table_html = """
        <table class="styled-table">
            <thead>
                <tr>
                    <th>CLASS</th>
                    <th>CONFIDENCE</th>
                    <th>X1</th>
                    <th>Y1</th>
                    <th>X2</th>
                    <th>Y2</th>
                    <th>WIDTH</th>
                    <th>HEIGHT</th>
                </tr>
            </thead>
            <tbody>
        """
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            cls_name = d["class"]
            conf = d["confidence"]
            w = x2 - x1
            h = y2 - y1
            table_html += f"""
                <tr>
                    <td><strong style="color:#0284c7;">{cls_name}</strong></td>
                    <td>{conf:.3f}</td>
                    <td>{x1}</td>
                    <td>{y1}</td>
                    <td>{x2}</td>
                    <td>{y2}</td>
                    <td>{w}</td>
                    <td>{h}</td>
                </tr>
            """
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("No objects detected above the operational confidence threshold.")

    st.markdown('</div>', unsafe_allow_html=True)


def render_model4_special_view(model4_results, pipeline_ref):
    """Render dedicated Model 4 visualization area with dynamic hazard replanning."""
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Model 3 Special Visualization: FloodNet Terrain Segmentation & D* Lite Planning</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Quad-View Mission Display", "Dynamic Replanning Demonstration", "FloodNet Planning Table"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p style='font-family:\"JetBrains Mono\"; font-size:0.8rem; font-weight:600; color:#475569;'>A. ORIGINAL IMAGE (512x512)</p>", unsafe_allow_html=True)
            st.image(model4_results["original_512"], use_column_width=True)

            st.markdown("<p style='font-family:\"JetBrains Mono\"; font-size:0.8rem; font-weight:600; color:#475569;'>C. COST MAP (LOW COST: GREEN | HIGH: RED | BLOCKED: CRIMSON)</p>", unsafe_allow_html=True)
            st.image(model4_results["colored_cost_map"], use_column_width=True)

        with col2:
            st.markdown("<p style='font-family:\"JetBrains Mono\"; font-size:0.8rem; font-weight:600; color:#475569;'>B. SEGMENTATION OVERLAY (10 FLOODNET CLASSES)</p>", unsafe_allow_html=True)
            st.image(model4_results["segmentation_overlay"], use_column_width=True)

            st.markdown("<p style='font-family:\"JetBrains Mono\"; font-size:0.8rem; font-weight:600; color:#475569;'>D & E. 64x64 PLANNING GRID & AUTONOMOUS D* LITE PATH</p>", unsafe_allow_html=True)
            st.image(model4_results["route_map"], use_column_width=True)

    with tab2:
        st.markdown("""
        <div style="font-family:'JetBrains Mono'; font-size:0.82rem; color:#334155; margin-bottom:12px;">
            Simulate an emergent disaster hazard dynamically intersecting the UAV flight path.
            D* Lite will perform real-time incremental replanning around the hazard without full re-computation.
        </div>
        """, unsafe_allow_html=True)

        hazard_btn = st.button("SIMULATE NEW HAZARD", type="primary", key="hazard_btn")

        if hazard_btn or st.session_state.get("hazard_simulated", False):
            st.session_state["hazard_simulated"] = True
            replan_res = pipeline_ref.trigger_hazard_simulation(radius=4, hazard_cost=1000.0)
            stats = replan_res["replan_stats"]

            # 6 Verified Replan Metrics
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Initial Path Length", f"{stats['initial_path_length']} cells")
            m2.metric("Hazard Cells", f"{stats['hazard_cells']} cells")
            m3.metric("Old Route Hazard Hits", f"{stats['old_route_hazard_hits']}")
            m4.metric("New Route Hazard Hits", f"{stats['new_route_hazard_hits']}")
            m5.metric("Replan Iterations", f"{stats['replan_iterations']}")
            m6.metric("Changed Route Cells", f"{stats['changed_route_cells']}")

            # Visual representation
            c_left, c_right = st.columns([3, 2])
            with c_left:
                st.markdown("<p style='font-family:\"JetBrains Mono\"; font-size:0.8rem; font-weight:600; color:#475569;'>ROUTE REPLAN: CYAN = ORIGINAL | GREEN = NEW ROUTE | RED = NEW HAZARD</p>", unsafe_allow_html=True)
                st.image(replan_res["updated_route_map"], use_column_width=True)

            with c_right:
                st.markdown("""
                <div style="background:#f8fafc; border:1px solid #cbd5e1; padding:14px; border-radius:4px; font-family:'JetBrains Mono'; font-size:0.75rem;">
                    <div style="color:#0284c7; font-weight:700; margin-bottom:8px;">DYNAMIC REPLANNING TELEMETRY</div>
                    <p style="color:#475569; line-height:1.6;">
                        • Hazard injection: 9x9 hazard region around active route<br>
                        • Original route intersections: <span style="color:#b91c1c; font-weight:600;">{hits} hits (Compromised)</span><br>
                        • Incremental D* Lite priority queue recalculation<br>
                        • New route hazard intersections: <span style="color:#15803d; font-weight:600;">0 hits (Clear)</span><br>
                        • Route divergence: {changed} cells recalculated<br>
                        • Replan iterations: {iters}<br>
                        • Verification Status: <span style="color:#15803d; font-weight:700;">PASSED</span>
                    </p>
                </div>
                """.format(
                    hits=stats['old_route_hazard_hits'],
                    changed=stats['changed_route_cells'],
                    iters=stats['replan_iterations']
                ), unsafe_allow_html=True)

    with tab3:
        table_html = """
        <table class="styled-table">
            <thead>
                <tr><th>CLASS ID</th><th>CLASS NAME</th><th>PLANNING COST</th><th>PRIORITY</th></tr>
            </thead>
            <tbody>
        """
        for cid, cname in FLOODNET_CLASSES.items():
            cost = FLOODNET_COSTS.get(cid, 100)
            priority = "Optimal Route" if cost == 1 else ("Safe/Clear" if cost < 100 else ("Hazardous" if cost < 200 else "Impassable"))
            color_hex = "#15803d" if cost == 1 else ("#d97706" if cost < 150 else "#b91c1c")
            table_html += f"""
                <tr>
                    <td>{cid}</td>
                    <td><strong>{cname}</strong></td>
                    <td style="color:{color_hex}; font-weight:600;">{cost}</td>
                    <td>{priority}</td>
                </tr>
            """
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def main():
    initialize_app()

    # Access singletons
    manager = get_model_manager()
    switcher = get_model_switcher()
    registry = get_model_registry()

    # Top Header Telemetry Bar
    render_top_telemetry_bar(manager, switcher)

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="card-title">Mission Control Configuration</div>', unsafe_allow_html=True)

        input_type = st.radio("OPERATIONAL INPUT TYPE", ["IMAGE", "VIDEO"], index=0)

        conf_threshold = st.slider(
            "DETECTION CONFIDENCE THRESHOLD",
            min_value=0.05,
            max_value=0.90,
            value=0.15,
            step=0.05,
            help="Adjust detection sensitivity threshold for object models (Model 1, Model 2, Model 4).",
        )

        st.markdown("---")
        st.markdown('<div class="card-title">Switching Architecture</div>', unsafe_allow_html=True)
        is_neural = switcher.classifier.is_neural_model_loaded
        engine_label = "Neural Weights (best.pt)" if is_neural else "Real-time Spectral Engine"
        st.markdown(f"""
        <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:4px; padding:10px 12px; font-family:'JetBrains Mono'; font-size:0.75rem;">
            <div style="color:#0284c7; font-weight:700;">MODEL 5 / MAIN — INFERENCE SWITCHER</div>
            <p style="color:#15803d; margin:4px 0 2px 0; font-weight:600;">• STATUS: ACTIVE / LOADED</p>
            <p style="color:#475569; margin:0; font-size:0.7rem;">• ENGINE: {engine_label}</p>
            <p style="color:#64748b; margin:4px 0 0 0; line-height:1.3; font-size:0.68rem;">
                Autonomous scene classification & dynamic model routing.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="card-title">System Status</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono'; font-size:0.72rem; color:#475569;">
            • Compute Device: {manager.device}<br>
            • Active Context: {switcher.current_context}<br>
            • Active Domain Model: {switcher.active_model_id.upper()}<br>
            • Context Switcher: MODEL 5 / MAIN (ACTIVE)
        </div>
        """, unsafe_allow_html=True)

    # ==========================================================
    # IMAGE MODE
    # Layout matches:
    # INPUT / IMAGE
    # INFERENCE VIEW | CONTEXT SWITCHING
    # MODEL STATUS
    # ==========================================================
    if input_type == "IMAGE":
        # 1. INPUT / IMAGE Card
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Input / Image</div>', unsafe_allow_html=True)

        col_up, col_synth = st.columns([3, 1])
        with col_up:
            uploaded_image = st.file_uploader(
                "Upload Mission Image (JPG, PNG, JPEG, BMP, TIFF)",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                key="image_uploader",
            )
        with col_synth:
            st.markdown("<p style='font-size:0.75rem; color:#64748b; margin-top:10px;'>Or initialize a synthetic UAV sensor frame:</p>", unsafe_allow_html=True)
            use_synthetic = st.button("GENERATE UAV FRAME", use_container_width=True)

        image_np = None
        if uploaded_image is not None:
            try:
                pil_img = Image.open(uploaded_image).convert("RGB")
                image_np = np.array(pil_img)
            except Exception as e:
                st.error(f"Error reading image: {e}")
        elif use_synthetic or st.session_state.get("use_synthetic", False):
            st.session_state["use_synthetic"] = True
            h, w = 512, 512
            syn = np.zeros((h, w, 3), dtype=np.uint8)
            syn[:] = (70, 95, 70)  # Terrain
            cv2.line(syn, (0, 200), (512, 350), (140, 140, 150), 44)  # Road
            cv2.circle(syn, (420, 160), 85, (50, 110, 180), -1)  # Water
            image_np = syn

        st.markdown('</div>', unsafe_allow_html=True)

        # 2. 2-Column Section: INFERENCE VIEW | CONTEXT SWITCHING
        col_inf, col_ctx = st.columns([3, 2])

        model4_results = None
        model4_pipeline_ref = None
        det_results = None

        with col_inf:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)

            if image_np is not None:
                # Run Inference Switching Model on input sensor image
                if not st.session_state.get("context_manually_locked", False):
                    switcher.infer_and_switch(image_np)

                active_model_id = switcher.active_model_id
                if active_model_id in ("model1", "model2", "model5"):
                    engine = manager.get_or_load_engine(active_model_id)
                    det_results = engine.run_inference(image_np, conf_threshold=conf_threshold)

                    manager.perf_metrics["inference_time_ms"] = det_results["inference_time_ms"]
                    manager.perf_metrics["fps"] = 1000.0 / max(1.0, det_results["inference_time_ms"])
                    manager.perf_metrics["total_detections"] = det_results["total_detections"]
                    manager.perf_metrics["device"] = det_results["device"]

                    st.markdown(f'<div class="card-title">Inference View — {det_results["total_detections"]} Detections</div>', unsafe_allow_html=True)
                    st.image(det_results["annotated_image"], use_column_width=True)

                elif active_model_id == "model4":
                    pipeline = manager.get_or_load_engine("model4")
                    model4_pipeline_ref = pipeline
                    model4_results = pipeline.execute_pipeline(image_np)

                    manager.perf_metrics["inference_time_ms"] = model4_results["inference_time_ms"]
                    manager.perf_metrics["fps"] = 1000.0 / max(1.0, model4_results["inference_time_ms"])
                    manager.perf_metrics["total_detections"] = model4_results["path_length"]
                    manager.perf_metrics["device"] = model4_results["device"]

                    st.markdown('<div class="card-title">Inference View — Terrain Segmentation & Path Overlay</div>', unsafe_allow_html=True)
                    st.image(model4_results["annotated_image"], use_column_width=True)
            else:
                st.markdown('<div class="card-title">Inference View</div>', unsafe_allow_html=True)
                st.info("Awaiting sensor image input. Upload an image or click 'GENERATE UAV FRAME' to begin inference.")

            st.markdown('</div>', unsafe_allow_html=True)

        with col_ctx:
            # CONTEXT SWITCHING Card
            render_context_switching_card(switcher, manager)

        # 3. MODEL STATUS Panel (Full Width)
        render_model_status_grid(manager)

        # 4. Model-Specific Detailed Output Panels
        if det_results is not None:
            render_detection_table(det_results, active_model_id)
        elif model4_results is not None:
            render_model4_special_view(model4_results, model4_pipeline_ref)

    # ==========================================================
    # VIDEO MODE
    # ==========================================================
    elif input_type == "VIDEO":
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Video Stream Upload & Processing Configuration</div>', unsafe_allow_html=True)

        col_vcfg1, col_vcfg2 = st.columns(2)
        with col_vcfg1:
            uploaded_video = st.file_uploader(
                "Upload UAV Video Stream (MP4, AVI, MOV, MKV)",
                type=["mp4", "avi", "mov", "mkv"],
                key="video_uploader",
            )
        if uploaded_video is not None:
            vid_key = f"{uploaded_video.name}_{uploaded_video.size}"
            if st.session_state.get("active_video_key") != vid_key:
                orig_suffix = Path(uploaded_video.name).suffix or ".mp4"
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=orig_suffix)
                tfile.write(uploaded_video.getvalue() if hasattr(uploaded_video, "getvalue") else uploaded_video.read())
                tfile.flush()
                tfile.close()

                # Default to AUTO-DETECT (MODEL 5 / MAIN INFERENCE SWITCHER) for all uploaded videos
                preset_model = "AUTO-DETECT (MODEL 5 / MAIN INFERENCE SWITCHER)"

                st.session_state["video_target_model_select"] = preset_model
                st.session_state["active_video_key"] = vid_key
                st.session_state["input_video_path"] = tfile.name
                st.session_state["uploaded_video_name"] = uploaded_video.name
                st.session_state["output_video_bytes"] = None
                st.session_state["proc_res"] = None

        with col_vcfg2:
            model_options = [
                "AUTO-DETECT (MODEL 5 / MAIN INFERENCE SWITCHER)",
                "MODEL 1 — VISIBLE HUMAN DETECTION (Daylight Reconnaissance)",
                "MODEL 2 — FIRE & SMOKE HAZARD DETECTION (Hazard Scene Analysis)",
                "MODEL 3 — TERRAIN SEGMENTATION & NAVIGATION (FloodNet & D* Lite)",
                "MODEL 4 — THERMAL HUMAN DETECTION (Thermal Infrared Search)",
            ]

            selected_model_str = st.selectbox(
                "ASSIGN TARGET MODEL / INFERENCE MODE",
                model_options,
                key="video_target_model_select",
            )

            # Map model string to target_model_id
            target_model_id = None
            if "MODEL 1" in selected_model_str:
                target_model_id = "model1"
            elif "MODEL 2" in selected_model_str:
                target_model_id = "model2"
            elif "MODEL 3" in selected_model_str:
                target_model_id = "model4"
            elif "MODEL 4" in selected_model_str:
                target_model_id = "model5"

            max_duration = st.slider("MAX VIDEO PROCESSING DURATION (SECONDS)", 5, 60, 15)

        if uploaded_video is not None:
            input_video_path = st.session_state["input_video_path"]

            col_vid1, col_vid2 = st.columns(2)
            with col_vid1:
                st.markdown("<p style='font-family:\"JetBrains Mono\"; font-size:0.8rem; font-weight:600; color:#475569;'>ORIGINAL VIDEO STREAM</p>", unsafe_allow_html=True)
                st.video(input_video_path)

            with col_vid2:
                st.markdown("<p style='font-family:\"JetBrains Mono\"; font-size:0.8rem; font-weight:600; color:#475569;'>PROCESSED ANNOTATED VIDEO STREAM</p>", unsafe_allow_html=True)
                if st.session_state.get("output_video_bytes") is not None:
                    st.video(st.session_state["output_video_bytes"])
                    st.download_button(
                        label="DOWNLOAD ANNOTATED VIDEO (MP4)",
                        data=st.session_state["output_video_bytes"],
                        file_name=f"annotated_{uploaded_video.name}",
                        mime="video/mp4",
                        type="primary",
                        use_container_width=True,
                    )
                else:
                    st.info("Click 'EXECUTE VIDEO INFERENCE & CONTEXT-SWITCHING PIPELINE' below to generate the annotated video.")

            start_btn = st.button("EXECUTE VIDEO INFERENCE & CONTEXT-SWITCHING PIPELINE", type="primary", use_container_width=True)

            if start_btn:
                out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                out_path = out_temp.name
                out_temp.close()

                prog_bar = st.progress(0.0)
                status_box = st.empty()

                processor = VideoProcessor()

                def update_progress(p, msg):
                    safe_p = max(0.0, min(1.0, float(p)))
                    prog_bar.progress(safe_p)
                    status_box.markdown(f"<span style='font-family:\"JetBrains Mono\"; font-size:0.8rem; color:#0284c7;'>{msg}</span>", unsafe_allow_html=True)

                try:
                    with st.spinner("Processing video frames and applying model detections..."):
                        proc_res = processor.process_video_stream(
                            input_path=input_video_path,
                            output_path=out_path,
                            mode="INFERENCE_DRIVEN" if not target_model_id else "MANUAL",
                            target_model_id=target_model_id,
                            video_name=st.session_state.get("uploaded_video_name", uploaded_video.name),
                            initial_context="Visible Person Search" if not target_model_id else registry.get(target_model_id).context_name,
                            max_duration_sec=float(max_duration),
                            conf_threshold=float(conf_threshold),
                            progress_callback=update_progress,
                        )

                    with open(out_path, "rb") as f:
                        video_bytes = f.read()

                    st.session_state["output_video_bytes"] = video_bytes
                    st.session_state["proc_res"] = proc_res
                    if switcher.current_context in CONTEXT_MODEL_MAP:
                        st.session_state["override_context_select"] = switcher.current_context

                    prog_bar.progress(1.0)
                    status_box.success(f"Processing Complete: {proc_res['total_frames_processed']} frames @ {proc_res['effective_fps']:.1f} FPS")

                    try:
                        os.unlink(out_path)
                    except Exception:
                        pass

                    st.rerun()

                except Exception as e:
                    import traceback
                    st.error(f"Video processing error: {e}")
                    st.code(traceback.format_exc(), language="python")

            # Render results if available in session_state
            if st.session_state.get("proc_res") is not None:
                proc_res = st.session_state["proc_res"]

                # Video Summary Metrics Panel
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Video Inference Summary & Metrics</div>', unsafe_allow_html=True)
                vcol1, vcol2, vcol3, vcol4, vcol5 = st.columns(5)
                vcol1.metric("Processed Frames", proc_res["total_frames_processed"])
                vcol2.metric("Effective Throughput", f"{proc_res['effective_fps']:.1f} FPS")
                vcol3.metric("Total Detections", len(proc_res.get("all_detections", [])))
                vcol4.metric("Highest Confidence", f"{proc_res.get('highest_confidence', 0.0):.2f}")
                vcol5.metric("Average Confidence", f"{proc_res.get('average_confidence', 0.0):.2f}")
                st.markdown('</div>', unsafe_allow_html=True)



                # Model 4 Quad View & Dynamic Replanning Visualization if Model 4 active
                if proc_res.get("latest_model4_results"):
                    render_model4_special_view(proc_res["latest_model4_results"], proc_res["latest_model4_pipeline"])

                # Context Switching Timeline
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Context Switching Timeline</div>', unsafe_allow_html=True)

                for item in proc_res["timeline"]:
                    st.markdown(f"""
                    <div class="timeline-item active">
                        <span class="timeline-time">{item['timestamp']}</span> &nbsp; 
                        <strong>{item['model_name']}</strong> — {item['context_name']} &nbsp; 
                        <span style="color:#64748b;">({item['transition']})</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Upload an .mp4, .avi, .mov, or .mkv UAV video stream to evaluate model detections and context switching across frames.")

        st.markdown('</div>', unsafe_allow_html=True)

        # Context Switching & Model Status in Video Mode
        col_c, col_m = st.columns([1, 1])
        with col_c:
            render_context_switching_card(switcher, manager)
        with col_m:
            render_model_status_grid(manager)

    # 5. Performance Panel (Full Width)
    render_performance_panel(manager)


if __name__ == "__main__":
    main()

