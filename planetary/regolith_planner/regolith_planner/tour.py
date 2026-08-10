# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Pick a scripted tour's waypoints from the costmap the rover will actually plan on.

WHY THIS EXISTS. `tour_mission.py` used to drive a hardcoded list of five (x, y) pairs,
chosen by hand before the terrain went to res40 and before rock collision started
working. Measured against the shipped costmap afterwards, waypoints were sitting on
LETHAL cells - seed 42's first, seed 7's second, seed 123's second and third. The
planner refuses those outright ("Goal cell (138, 143) is lethal - pick another goal"),
so the leg was skipped after a 90 s timeout and the unattended demo spent minutes
sitting still. A hand-picked constant cannot survive terrain that changes; a route
derived from the terrain can.

WHAT MAKES A GOOD LEG, and how each part is checked rather than assumed:

  * **Plannable.** The waypoint's own cell and its eight neighbours must be non-lethal
    (`planner_node` snaps a goal to a cell centre, so a clear cell in a lethal
    neighbourhood is not reliably plannable), and A* must actually find a path to it
    from the previous waypoint. That is the same `plan_path` the planner node will run,
    not an approximation of it - a flood fill would say "connected" for a route A* is
    free to reject.
  * **Worth driving.** The straight line from the previous waypoint has to be blocked by
    something lethal, or the leg is a drive across open regolith that exercises no
    obstacle avoidance at all. This is the property the original hand-picked list was
    trying to have: its own comment notes the shipped route crossed an obstacle on only
    1 of 5 legs.
  * **A tour, not a wander.** Legs are 10-20 m (short legs are deliberate - see
    tour_mission.py), waypoints keep clear of each other, and the last outbound
    waypoint is within one leg of the spawn point so the rover can come home.

Deterministic from `seed`: the same terrain always produces the same tour, so a run is
reproducible and a screenshot keeps meaning what it meant.
"""

import math
import random

import numpy as np

from regolith_planner.astar import LETHAL_COST, plan_path

LEG_COUNT = 5  # goals published, the last of which is the return to spawn
LEG_MIN_M = 10.0
LEG_MAX_M = 20.0
# Keeps the route from doubling back onto a waypoint it has already flagged, which
# would read as the rover shuttling between two points.
MIN_SEPARATION_M = 8.0
# No waypoint further than this from spawn. Four legs of up to 20 m can otherwise walk
# the rover 50+ m out, and then no single leg brings it home - which is exactly how the
# first version of this ended seed 2's tour with a 37.6 m final leg, well outside the
# 10-20 m band the short-leg design exists to stay inside.
MAX_RANGE_M = 30.0
DRAWS_PER_LEG = 400
# Candidates that pass every hard requirement are collected rather than taken first
# come, and the one with the widest pinch point wins. Measured on seeds 42/7/123, the
# first-match route put the tightest squeeze of 13 of its 15 legs at 0.78 m - one
# costmap cell. The costmap inflates obstacles by the rover radius, so one free cell is
# "passable" on paper; a 0.5 m skid-steer rover slewing through it wedges, and a live
# seed 42 tour wedged three times in one such pinch and timed the leg out.
CANDIDATE_POOL = 8
# How far out to look for the nearest obstacle when scoring a path cell. Beyond this
# the corridor is wide enough that more room stops mattering.
CLEARANCE_SEARCH_CELLS = 4


def to_cell(x_m, y_m, resolution_m, origin_x, origin_y) -> tuple:
    return (int((y_m - origin_y) / resolution_m), int((x_m - origin_x) / resolution_m))


def _neighbourhood_clear(cost_grid: np.ndarray, rc: tuple) -> bool:
    rows, cols = cost_grid.shape
    r, c = rc
    if not (0 <= r < rows and 0 <= c < cols):
        return False
    block = cost_grid[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
    return not bool((block >= LETHAL_COST).any())


def path_clearance_cells(cost_grid: np.ndarray, path: list) -> float:
    """Width of the tightest pinch on a path, in cells: the smallest distance from any
    cell of it to the nearest lethal cell, capped at CLEARANCE_SEARCH_CELLS.

    Local window search rather than a distance transform, to keep this module's
    dependencies to numpy - the paths are short and the window is small.
    """
    rows, cols = cost_grid.shape
    reach = CLEARANCE_SEARCH_CELLS
    offsets = [
        (dr, dc, math.hypot(dr, dc))
        for dr in range(-reach, reach + 1)
        for dc in range(-reach, reach + 1)
        if math.hypot(dr, dc) <= reach
    ]
    offsets.sort(key=lambda o: o[2])

    worst = float(reach)
    for r, c in path:
        for dr, dc, distance in offsets:
            if distance >= worst:
                break  # nothing closer available here than what we have already
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or cost_grid[nr, nc] >= LETHAL_COST:
                worst = distance
                break
    return worst


def _straight_line_blocked(cost_grid: np.ndarray, a_rc: tuple, b_rc: tuple) -> bool:
    """Does the straight line between two cells cross anything lethal?

    Sampled at half-cell steps - a leg whose direct line is clear is one the planner
    would fly straight down, which tests nothing.
    """
    steps = max(2, int(2 * math.dist(a_rc, b_rc)))
    rows, cols = cost_grid.shape
    for i in range(steps + 1):
        t = i / steps
        r = int(round(a_rc[0] + (b_rc[0] - a_rc[0]) * t))
        c = int(round(a_rc[1] + (b_rc[1] - a_rc[1]) * t))
        if 0 <= r < rows and 0 <= c < cols and cost_grid[r, c] >= LETHAL_COST:
            return True
    return False


def plan_tour(
    cost_grid: np.ndarray,
    resolution_m: float,
    origin_x: float,
    origin_y: float,
    spawn_xy: tuple = (0.0, 0.0),
    seed: int = 0,
    leg_count: int = LEG_COUNT,
    leg_min_m: float = LEG_MIN_M,
    leg_max_m: float = LEG_MAX_M,
) -> dict:
    """Returns {"waypoints": [(x, y), ...], "notes": [str, ...]}.

    `waypoints` has `leg_count` entries and ends at `spawn_xy`. `notes` records every
    place a requirement had to be relaxed to fill the route, so a weaker tour reports
    itself rather than looking like a good one.
    """
    cost_grid = np.asarray(cost_grid)
    rng = random.Random(seed)
    notes = []

    def cell(xy):
        return to_cell(xy[0], xy[1], resolution_m, origin_x, origin_y)

    spawn_rc = cell(spawn_xy)
    if not _neighbourhood_clear(cost_grid, spawn_rc):
        # Not survivable by relaxing anything: with no clear spawn there is no route.
        raise ValueError(f"spawn {spawn_xy} is not in clear terrain - cannot build a tour")

    waypoints = []
    last = tuple(spawn_xy)
    outbound = leg_count - 1  # the final leg is the return to spawn

    for index in range(outbound):
        # The last outbound waypoint has to leave the rover one leg from home.
        wants_home = index == outbound - 1
        chosen = None
        # Requirements are given up one at a time, in the order they are least missed:
        # a leg over open ground is still a leg, and a long way home is still a way
        # home. Whatever was surrendered is recorded rather than absorbed silently.
        relaxations = [
            (True, wants_home, None),
            (False, wants_home, "crosses open ground - no candidate had a blocked direct line"),
        ]
        if wants_home:
            # Only worth trying without the homeward requirement if there was one -
            # otherwise these repeat the two attempts above draw for draw.
            relaxations += [
                (True, False, "leaves the rover more than one leg from home"),
                (False, False,
                 "crosses open ground and leaves the rover more than one leg from home"),
            ]
        for require_blocked, homeward, note in relaxations:
            chosen = _draw_waypoint(
                cost_grid, rng, last, waypoints, spawn_xy, cell,
                leg_min_m, leg_max_m, require_blocked, homeward,
            )
            if chosen is not None:
                if note:
                    notes.append(f"leg {index + 1} {note}")
                break
        if chosen is None:
            notes.append(
                f"leg {index + 1}: no plannable waypoint found in "
                f"{leg_min_m:.0f}-{leg_max_m:.0f} m after {DRAWS_PER_LEG} draws per "
                f"attempt - the tour runs {len(waypoints)} legs out instead of {outbound}"
            )
            break
        waypoints.append(chosen)
        last = chosen

    # Come home. The rover has to be able to plan back, or the tour ends stranded.
    if plan_path(cost_grid, cell(last), spawn_rc):
        waypoints.append(tuple(spawn_xy))
        home_leg = math.dist(last, spawn_xy)
        if home_leg > leg_max_m:
            notes.append(
                f"the return leg is {home_leg:.1f} m, longer than the {leg_max_m:.0f} m "
                f"the short-leg design keeps to"
            )
    else:
        notes.append(
            f"no path back to spawn from {last} - the tour ends where it finishes "
            f"rather than looping"
        )
    return {"waypoints": waypoints, "notes": notes}


def _draw_waypoint(
    cost_grid, rng, last, chosen_so_far, spawn_xy, cell,
    leg_min_m, leg_max_m, require_blocked, homeward,
):
    """One waypoint, by rejection sampling. Cheap tests first, A* only on survivors -
    A* over the whole grid is far too expensive to run on every draw.

    Survivors are pooled rather than taken first come, and the winner is the one whose
    path has the widest pinch point: every candidate here is already plannable, so the
    remaining question is which route the rover can physically get through.
    """
    last_rc = cell(last)
    pool = []
    for _ in range(DRAWS_PER_LEG):
        distance = rng.uniform(leg_min_m, leg_max_m)
        bearing = rng.uniform(-math.pi, math.pi)
        candidate = (
            round(last[0] + distance * math.cos(bearing), 2),
            round(last[1] + distance * math.sin(bearing), 2),
        )
        from_spawn = math.dist(candidate, spawn_xy)
        if homeward and not (leg_min_m <= from_spawn <= leg_max_m):
            continue
        if from_spawn > MAX_RANGE_M or from_spawn < MIN_SEPARATION_M:
            continue
        if any(math.dist(candidate, w) < MIN_SEPARATION_M for w in chosen_so_far):
            continue
        candidate_rc = cell(candidate)
        if not _neighbourhood_clear(cost_grid, candidate_rc):
            continue
        if require_blocked and not _straight_line_blocked(cost_grid, last_rc, candidate_rc):
            continue
        path = plan_path(cost_grid, last_rc, candidate_rc)
        if not path:
            continue
        pool.append((path_clearance_cells(cost_grid, path), candidate))
        if len(pool) >= CANDIDATE_POOL:
            break
    if pool:
        return max(pool, key=lambda entry: entry[0])[1]
    return None
