"""D* Lite High-Level Engine for UAV Path Planning and Dynamic Replanning.

Handles initial path generation, real-time dynamic obstacle injection,
incremental replanning metrics calculation, and grid-based route visualization.
"""

from typing import Tuple, List, Dict, Optional, Set
import numpy as np
import cv2
from inference.dstar_lite import DStarLite


class DStarLiteEngine:
    """High-level orchestration of D* Lite on a 64x64 planning grid."""

    def __init__(
        self,
        planning_grid: np.ndarray,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ):
        self.height, self.width = planning_grid.shape
        self.base_grid = np.copy(planning_grid).astype(np.float32)

        # Default start (top-left clearance) and goal (bottom-right clearance)
        self.start = start if start else self._find_clear_cell(start=(4, 4))
        self.goal = goal if goal else self._find_clear_cell(start=(self.height - 5, self.width - 5))

        self.planner = DStarLite(self.base_grid, self.start, self.goal)
        self.original_path: List[Tuple[int, int]] = []
        self.new_path: List[Tuple[int, int]] = []
        self.hazard_cells: Set[Tuple[int, int]] = set()
        self.replan_stats: Dict[str, int] = {}

    def _find_clear_cell(self, start: Tuple[int, int], radius: int = 5) -> Tuple[int, int]:
        """Find the nearest low-cost cell around a preferred coordinate."""
        r0, c0 = start
        best_cell = (r0, c0)
        best_cost = float("inf")
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < self.height and 0 <= c < self.width:
                    cost = self.base_grid[r, c]
                    if cost < best_cost:
                        best_cost = cost
                        best_cell = (r, c)
        return best_cell

    def compute_initial_path(self) -> List[Tuple[int, int]]:
        """Run initial path planning from start to goal."""
        self.planner.compute_shortest_path()
        self.original_path = self.planner.extract_path()
        return self.original_path

    def simulate_hazard_and_replan(
        self,
        center_cell: Optional[Tuple[int, int]] = None,
        radius: int = 4,
        hazard_cost: float = 1000.0,
    ) -> Dict[str, any]:
        """Inject a dynamic hazard across the current trajectory and execute incremental replan.

        Produces verified metrics:
            - Initial path length
            - Hazard cells count
            - Old route hazard hits
            - New route hazard hits (target: 0)
            - Replan iterations
            - Changed route cells
        """
        if not self.original_path:
            self.compute_initial_path()

        if not self.original_path:
            return {"error": "No initial path found"}

        # Select a hazard center along the middle section of the original route
        if center_cell is None:
            mid_idx = len(self.original_path) // 2
            center_cell = self.original_path[mid_idx]

        cr, cc = center_cell
        changed_cells: Dict[Tuple[int, int], float] = {}
        hazard_set: Set[Tuple[int, int]] = set()

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = cr + dr, cc + dc
                if 0 <= r < self.height and 0 <= c < self.width:
                    # Avoid blocking start or goal
                    if (r, c) != self.start and (r, c) != self.goal:
                        changed_cells[(r, c)] = hazard_cost
                        hazard_set.add((r, c))

        self.hazard_cells = hazard_set

        # Execute incremental D* Lite update
        iterations = self.planner.update_obstacles(changed_cells)
        self.new_path = self.planner.extract_path()

        # Compute metric breakdown
        old_set = set(self.original_path)
        new_set = set(self.new_path)

        old_hits = len(old_set.intersection(self.hazard_cells))
        new_hits = len(new_set.intersection(self.hazard_cells))

        # Changed route cells: count of cells in original path that are not in new path
        changed_cells_count = len(old_set.symmetric_difference(new_set))

        self.replan_stats = {
            "initial_path_length": len(self.original_path),
            "hazard_cells": len(self.hazard_cells),
            "old_route_hazard_hits": old_hits,
            "new_route_hazard_hits": new_hits,
            "replan_iterations": iterations,
            "changed_route_cells": changed_cells_count,
            "new_path_length": len(self.new_path),
        }

        return {
            "original_path": self.original_path,
            "new_path": self.new_path,
            "hazard_cells": list(self.hazard_cells),
            "stats": self.replan_stats,
        }

    def render_route_map(
        self,
        canvas_size: int = 512,
        show_hazard: bool = True,
        show_new_route: bool = True,
    ) -> np.ndarray:
        """Render high-resolution visual representation of the planning grid, routes, and hazards."""
        # Upscale 64x64 grid to display canvas
        norm_grid = np.clip(self.planner.grid / 200.0, 0.0, 1.0)
        grid_u8 = (norm_grid * 255).astype(np.uint8)
        colored = cv2.applyColorMap(grid_u8, cv2.COLORMAP_TURBO)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        canvas = cv2.resize(colored, (canvas_size, canvas_size), interpolation=cv2.INTER_NEAREST)

        scale = canvas_size / self.width

        # Render Hazard Region
        if show_hazard and self.hazard_cells:
            overlay = canvas.copy()
            for r, c in self.hazard_cells:
                x1 = int(c * scale)
                y1 = int(r * scale)
                x2 = int((c + 1) * scale)
                y2 = int((r + 1) * scale)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 30, 30), -1)
            # Alpha blend hazard overlay
            cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        # Render Original Route (Cyan / Electric Blue)
        if self.original_path:
            pts_old = [
                (int((c + 0.5) * scale), int((r + 0.5) * scale))
                for r, c in self.original_path
            ]
            for i in range(len(pts_old) - 1):
                cv2.line(canvas, pts_old[i], pts_old[i + 1], (0, 220, 255), 2, cv2.LINE_AA)

        # Render New Route (Fluorescent Amber/Green) if replanned
        if show_new_route and self.new_path:
            pts_new = [
                (int((c + 0.5) * scale), int((r + 0.5) * scale))
                for r, c in self.new_path
            ]
            for i in range(len(pts_new) - 1):
                cv2.line(canvas, pts_new[i], pts_new[i + 1], (50, 255, 100), 3, cv2.LINE_AA)

        # Render Start and Goal nodes
        sx, sy = int((self.start[1] + 0.5) * scale), int((self.start[0] + 0.5) * scale)
        gx, gy = int((self.goal[1] + 0.5) * scale), int((self.goal[0] + 0.5) * scale)

        cv2.circle(canvas, (sx, sy), 7, (0, 255, 0), -1)
        cv2.circle(canvas, (sx, sy), 9, (255, 255, 255), 2)
        cv2.circle(canvas, (gx, gy), 7, (255, 50, 50), -1)
        cv2.circle(canvas, (gx, gy), 9, (255, 255, 255), 2)

        return canvas
