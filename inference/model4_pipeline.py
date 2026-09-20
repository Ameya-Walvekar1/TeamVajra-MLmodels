"""End-to-end Model 4 Pipeline: Segmentation to D* Lite Autonomous Path Planning.

Orchestrates:
    Image
    -> LRASPP MobileNetV3 Large
    -> 512x512 Segmentation Mask
    -> 512x512 Terrain Cost Map
    -> 64x64 Planning Grid
    -> D* Lite Path Planning & Dynamic Hazard Simulation
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from inference.model4_engine import Model4Engine
from inference.dstar_lite_engine import DStarLiteEngine


class Model4Pipeline:
    """Integrated pipeline combining Model 4 segmentation with D* Lite navigation."""

    def __init__(self):
        self.seg_engine = Model4Engine()
        self.last_planner: Optional[DStarLiteEngine] = None
        self.last_results: Optional[Dict[str, Any]] = None

    def execute_pipeline(
        self,
        image: np.ndarray,
        start_cell: Optional[Tuple[int, int]] = None,
        goal_cell: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Run full segmentation, cost mapping, grid downsampling, and D* Lite path planning."""
        seg_res = self.seg_engine.process_image(image)
        planning_grid = seg_res["planning_grid"]

        # Initialize D* Lite engine on the generated planning grid
        planner = DStarLiteEngine(planning_grid, start=start_cell, goal=goal_cell)
        path = planner.compute_initial_path()
        self.last_planner = planner

        # Render 512x512 planning route map
        route_map = planner.render_route_map(canvas_size=512, show_hazard=False, show_new_route=False)

        # Overlay planned route onto the 512x512 original image
        annotated_image = seg_res["original_512"].copy()
        scale = 512.0 / planner.width
        if path:
            pts = [(int((c + 0.5) * scale), int((r + 0.5) * scale)) for r, c in path]
            for i in range(len(pts) - 1):
                cv2.line(annotated_image, pts[i], pts[i + 1], (0, 220, 255), 2, cv2.LINE_AA)

        results = {
            **seg_res,
            "path": path,
            "path_length": len(path),
            "planner_engine": planner,
            "route_map": route_map,
            "annotated_image": annotated_image,
        }
        self.last_results = results
        return results

    def trigger_hazard_simulation(
        self,
        radius: int = 4,
        hazard_cost: float = 1000.0,
    ) -> Dict[str, Any]:
        """Trigger dynamic hazard injection and D* Lite replanning on current grid."""
        if not self.last_planner:
            raise RuntimeError("Pipeline must be executed once before triggering hazard simulation.")

        replan_res = self.last_planner.simulate_hazard_and_replan(
            radius=radius, hazard_cost=hazard_cost
        )

        # Render updated route map showing hazard region, original route, and new route
        updated_map = self.last_planner.render_route_map(
            canvas_size=512, show_hazard=True, show_new_route=True
        )

        return {
            "replan_stats": replan_res["stats"],
            "original_path": replan_res["original_path"],
            "new_path": replan_res["new_path"],
            "hazard_cells": replan_res["hazard_cells"],
            "updated_route_map": updated_map,
        }
