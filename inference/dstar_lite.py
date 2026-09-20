"""D* Lite Path Planning Algorithm for UAV Mission Navigation.

Implementation of Koenig & Likhachev's D* Lite algorithm for incremental heuristic search
on a 2D grid. Enables instantaneous dynamic replanning when new obstacles or hazards
appear in the UAV trajectory.
"""

import math
import heapq
from typing import Tuple, List, Dict, Optional, Set
import numpy as np


class PriorityQueue:
    """Priority queue supporting key updates and deletion."""

    def __init__(self):
        self._heap = []
        self._entry_finder: Dict[Tuple[int, int], List] = {}
        self._counter = 0

    def insert(self, item: Tuple[int, int], priority: Tuple[float, float]):
        if item in self._entry_finder:
            self.remove(item)
        entry = [priority[0], priority[1], self._counter, item]
        self._counter += 1
        self._entry_finder[item] = entry
        heapq.heappush(self._heap, entry)

    def remove(self, item: Tuple[int, int]):
        entry = self._entry_finder.pop(item, None)
        if entry:
            entry[-1] = None  # mark as removed

    def pop(self) -> Tuple[Tuple[float, float], Tuple[int, int]]:
        while self._heap:
            k1, k2, _, item = heapq.heappop(self._heap)
            if item is not None:
                del self._entry_finder[item]
                return (k1, k2), item
        raise KeyError("pop from empty priority queue")

    def top_key(self) -> Tuple[float, float]:
        while self._heap:
            k1, k2, _, item = self._heap[0]
            if item is None:
                heapq.heappop(self._heap)
            else:
                return (k1, k2)
        return (float("inf"), float("inf"))

    def contains(self, item: Tuple[int, int]) -> bool:
        return item in self._entry_finder

    def is_empty(self) -> bool:
        return not bool(self._entry_finder)


class DStarLite:
    """Incremental D* Lite path planner on a 2D cost grid.

    Grid coordinates: (r, c) where 0 <= r < height, 0 <= c < width.
    Search direction: Goal to Start (backward search) for efficient start-state movement.
    """

    def __init__(
        self,
        cost_grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacle_cost: float = 1000.0,
    ):
        self.grid = np.copy(cost_grid)
        self.height, self.width = self.grid.shape
        self.start = start
        self.goal = goal
        self.obstacle_cost = obstacle_cost

        self.s_start = start
        self.s_last = start
        self.s_goal = goal
        self.k_m = 0.0

        self.rhs: Dict[Tuple[int, int], float] = {}
        self.g: Dict[Tuple[int, int], float] = {}
        self.queue = PriorityQueue()
        self.replan_iterations = 0

        self._initialize()

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Octile distance heuristic for 8-connected grid."""
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        return (dr + dc) + (math.sqrt(2.0) - 2.0) * min(dr, dc)

    def _calculate_key(self, s: Tuple[int, int]) -> Tuple[float, float]:
        min_g_rhs = min(self._get_g(s), self._get_rhs(s))
        return (
            min_g_rhs + self._heuristic(self.s_start, s) + self.k_m,
            min_g_rhs,
        )

    def _get_g(self, s: Tuple[int, int]) -> float:
        return self.g.get(s, float("inf"))

    def _get_rhs(self, s: Tuple[int, int]) -> float:
        return self.rhs.get(s, float("inf"))

    def _cost(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Traversal cost from node a to neighbor b based on terrain cost."""
        if not self._is_valid(b) or not self._is_valid(a):
            return float("inf")

        cell_cost = float(self.grid[b[0], b[1]])
        if cell_cost >= self.obstacle_cost:
            return float("inf")

        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        dist = math.sqrt(2.0) if (dr == 1 and dc == 1) else 1.0
        return dist * (1.0 + cell_cost / 10.0)

    def _is_valid(self, s: Tuple[int, int]) -> bool:
        return 0 <= s[0] < self.height and 0 <= s[1] < self.width

    def _neighbors(self, s: Tuple[int, int]) -> List[Tuple[int, int]]:
        nbrs = []
        r, c = s
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if self._is_valid((nr, nc)):
                    nbrs.append((nr, nc))
        return nbrs

    def _initialize(self):
        self.queue = PriorityQueue()
        self.k_m = 0.0
        self.rhs.clear()
        self.g.clear()
        self.rhs[self.s_goal] = 0.0
        self.queue.insert(self.s_goal, self._calculate_key(self.s_goal))

    def _update_vertex(self, u: Tuple[int, int]):
        if u != self.s_goal:
            min_rhs = float("inf")
            for sprime in self._neighbors(u):
                c = self._cost(u, sprime)
                val = c + self._get_g(sprime)
                if val < min_rhs:
                    min_rhs = val
            self.rhs[u] = min_rhs

        if self.queue.contains(u):
            self.queue.remove(u)

        if self._get_g(u) != self._get_rhs(u):
            self.queue.insert(u, self._calculate_key(u))

    def compute_shortest_path(self, max_iterations: int = 5000) -> int:
        """Run shortest path computation until start is consistent."""
        iterations = 0
        while not self.queue.is_empty() and (
            self.queue.top_key() < self._calculate_key(self.s_start)
            or self._get_rhs(self.s_start) != self._get_g(self.s_start)
        ):
            if iterations >= max_iterations:
                break
            iterations += 1

            k_old, u = self.queue.pop()
            k_new = self._calculate_key(u)

            if k_old < k_new:
                self.queue.insert(u, k_new)
            elif self._get_g(u) > self._get_rhs(u):
                self.g[u] = self._get_rhs(u)
                for s in self._neighbors(u):
                    self._update_vertex(s)
            else:
                self.g[u] = float("inf")
                for s in self._neighbors(u) + [u]:
                    self._update_vertex(s)

        self.replan_iterations = iterations
        return iterations

    def update_obstacles(self, changed_cells: Dict[Tuple[int, int], float]) -> int:
        """Update grid cell costs dynamically and perform incremental replan."""
        self.k_m += self._heuristic(self.s_last, self.s_start)
        self.s_last = self.s_start

        for (r, c), new_cost in changed_cells.items():
            if self._is_valid((r, c)):
                self.grid[r, c] = new_cost
                u = (r, c)
                self._update_vertex(u)
                for s in self._neighbors(u):
                    self._update_vertex(s)

        return self.compute_shortest_path()

    def extract_path(self) -> List[Tuple[int, int]]:
        """Extract optimal path from s_start to s_goal."""
        if self._get_g(self.s_start) == float("inf"):
            return []

        path = [self.s_start]
        curr = self.s_start
        visited = {curr}

        while curr != self.s_goal:
            nbrs = self._neighbors(curr)
            best_nbr = None
            best_cost = float("inf")

            for nbr in nbrs:
                cost = self._cost(curr, nbr) + self._get_g(nbr)
                if cost < best_cost:
                    best_cost = cost
                    best_nbr = nbr

            if best_nbr is None or best_nbr in visited:
                break  # Cycle detected or unreachable

            curr = best_nbr
            path.append(curr)
            visited.add(curr)

            if len(path) > (self.height * self.width):
                break

        return path
