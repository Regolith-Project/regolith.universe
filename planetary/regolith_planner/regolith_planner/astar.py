# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Cost-aware A* over an occupancy grid: not shortest-path, cheapest-risk-path.

Cells with cost 100 (from regolith_costmap) are impassable. Cells with lower
cost are still traversable but expensive, so the search prefers routing around
them when a similarly-short alternative exists, rather than always taking the
geometrically shortest route.
"""

import heapq

import numpy as np

_NEIGHBORS = [
    (-1, -1, np.sqrt(2)), (-1, 0, 1.0), (-1, 1, np.sqrt(2)),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, np.sqrt(2)), (1, 0, 1.0), (1, 1, np.sqrt(2)),
]

LETHAL_COST = 100


def _cost_multiplier(cell_cost: int) -> float:
    """Higher-cost cells are more expensive to cross, biasing the search toward
    low-risk routing without forbidding moderate-cost cells outright."""
    return 1.0 + (cell_cost / 20.0)


def plan_path(cost_grid: np.ndarray, start_rc: tuple, goal_rc: tuple) -> list:
    """Returns a list of (row, col) grid cells from start to goal, or an empty
    list if no path exists."""
    rows, cols = cost_grid.shape
    if cost_grid[start_rc] >= LETHAL_COST or cost_grid[goal_rc] >= LETHAL_COST:
        return []

    def heuristic(rc: tuple) -> float:
        return float(np.hypot(rc[0] - goal_rc[0], rc[1] - goal_rc[1]))

    open_heap = [(heuristic(start_rc), 0.0, start_rc)]
    came_from = {}
    best_cost = {start_rc: 0.0}
    visited = set()

    while open_heap:
        _, g_cost, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal_rc:
            break

        for dr, dc, dist in _NEIGHBORS:
            neighbor = (current[0] + dr, current[1] + dc)
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            neighbor_cost = int(cost_grid[neighbor])
            if neighbor_cost >= LETHAL_COST or neighbor in visited:
                continue

            step_cost = dist * _cost_multiplier(neighbor_cost)
            tentative_g = g_cost + step_cost
            if tentative_g < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative_g + heuristic(neighbor), tentative_g, neighbor))

    if goal_rc not in came_from and start_rc != goal_rc:
        return []

    path = [goal_rc]
    while path[-1] != start_rc:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def smooth_path(path_xy: list, iterations: int = 3, weight: float = 0.4) -> list:
    """Light moving-average smoothing to soften the grid search's jagged
    diagonal/axis-aligned steps, keeping the endpoints fixed."""
    if len(path_xy) < 3:
        return path_xy
    points = [np.array(p, dtype=np.float64) for p in path_xy]
    for _ in range(iterations):
        smoothed = [points[0]]
        for i in range(1, len(points) - 1):
            averaged = (points[i - 1] + points[i + 1]) / 2.0
            smoothed.append((1 - weight) * points[i] + weight * averaged)
        smoothed.append(points[-1])
        points = smoothed
    return [(float(p[0]), float(p[1])) for p in points]
