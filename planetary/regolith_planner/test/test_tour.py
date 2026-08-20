# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""The tour route builder, on grids whose answer is known by construction.

The property that matters is the one the hardcoded route lost: every waypoint has to be
somewhere the planner will actually accept. `regolith_bringup` carries the same check
against real generated terrain; this file pins the behaviour on grids small and obvious
enough that a failure says where the fault is.
"""

import math

import numpy as np
import pytest
from regolith_planner.astar import LETHAL_COST
from regolith_planner.tour import LEG_MAX_M
from regolith_planner.tour import LEG_MIN_M
from regolith_planner.tour import MAX_RANGE_M
from regolith_planner.tour import MIN_SEPARATION_M
from regolith_planner.tour import plan_tour
from regolith_planner.tour import to_cell

# 1 m cells, 120 x 120 m centred on the spawn - room for a 30 m tour with margin.
RES = 1.0
ORIGIN = -60.0
SIZE = 120


def _grid(fill=0):
    return np.full((SIZE, SIZE), fill, dtype=np.int8)


def _scatter_obstacles(grid, count=90, radius=3, seed=0):
    """Lethal blobs everywhere except a clear disc around the spawn."""
    rng = np.random.default_rng(seed)
    rows, cols = grid.shape
    for _ in range(count):
        r, c = rng.integers(0, rows), rng.integers(0, cols)
        x, y = c * RES + ORIGIN, r * RES + ORIGIN
        if math.hypot(x, y) < 8.0:
            continue
        grid[max(0, r - radius) : r + radius, max(0, c - radius) : c + radius] = LETHAL_COST
    return grid


def _cells(waypoints):
    return [to_cell(x, y, RES, ORIGIN, ORIGIN) for x, y in waypoints]


def test_every_waypoint_is_somewhere_the_planner_will_accept():
    """The regression this module exists for: waypoints on lethal cells, which the planner refuses outright ("Goal cell is lethal - pick another goal")."""
    grid = _scatter_obstacles(_grid())
    result = plan_tour(grid, RES, ORIGIN, ORIGIN, seed=42)

    assert result["waypoints"], "no route produced on a grid with plenty of free space"
    for (x, y), (r, c) in zip(result["waypoints"], _cells(result["waypoints"])):
        # The cell AND its neighbours: planner_node snaps a goal to a cell centre, so a
        # clear cell in a lethal neighbourhood is not reliably plannable.
        block = grid[r - 1 : r + 2, c - 1 : c + 2]
        assert not (block >= LETHAL_COST).any(), f"waypoint ({x}, {y}) is not plannable"


def test_the_route_comes_home():
    grid = _scatter_obstacles(_grid())
    result = plan_tour(grid, RES, ORIGIN, ORIGIN, seed=7)
    assert result["waypoints"][-1] == (0.0, 0.0)


def test_legs_stay_in_the_short_leg_band():
    """Short legs are a deliberate safety choice, not an accident - see tour_mission.py."""
    grid = _scatter_obstacles(_grid())
    result = plan_tour(grid, RES, ORIGIN, ORIGIN, seed=123)
    points = [(0.0, 0.0)] + list(result["waypoints"])
    legs = [math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
    assert all(leg <= LEG_MAX_M + 1e-6 for leg in legs), legs
    # The return leg is the one that can come up short - it lands exactly on spawn.
    assert all(leg >= LEG_MIN_M - 1e-6 for leg in legs[:-1]), legs


def test_waypoints_stay_within_reach_of_home():
    grid = _scatter_obstacles(_grid())
    result = plan_tour(grid, RES, ORIGIN, ORIGIN, seed=2)
    for x, y in result["waypoints"]:
        assert math.hypot(x, y) <= MAX_RANGE_M + 1e-6


def test_waypoints_do_not_pile_up_on_each_other():
    grid = _scatter_obstacles(_grid())
    points = plan_tour(grid, RES, ORIGIN, ORIGIN, seed=5)["waypoints"][:-1]
    for i, a in enumerate(points):
        for b in points[i + 1 :]:
            assert math.dist(a, b) >= MIN_SEPARATION_M


def test_the_same_seed_gives_the_same_route():
    """A run has to be reproducible, and a screenshot has to keep meaning what it meant."""
    grid = _scatter_obstacles(_grid())
    assert (
        plan_tour(grid, RES, ORIGIN, ORIGIN, seed=42)["waypoints"]
        == plan_tour(grid, RES, ORIGIN, ORIGIN, seed=42)["waypoints"]
    )


def test_different_seeds_give_different_routes():
    grid = _scatter_obstacles(_grid())
    assert (
        plan_tour(grid, RES, ORIGIN, ORIGIN, seed=42)["waypoints"]
        != plan_tour(grid, RES, ORIGIN, ORIGIN, seed=7)["waypoints"]
    )


def test_open_ground_is_reported_rather_than_passed_off_as_a_real_leg():
    """On an empty grid no leg can cross an obstacle, because there are none. The route is still driveable, and the notes have to say what it is missing - a tour that exercises no avoidance must not look like one that does."""
    result = plan_tour(_grid(), RES, ORIGIN, ORIGIN, seed=42)
    assert len(result["waypoints"]) == 5
    assert sum("open ground" in n for n in result["notes"]) == 4


def test_a_boxed_in_spawn_is_refused_rather_than_worked_around():
    """Nothing can be built from it, and pretending otherwise would hand the mission a route it cannot drive."""
    grid = _grid(fill=LETHAL_COST)
    grid[55:65, 55:65] = 0  # a small clear pocket containing spawn, nothing beyond
    result = plan_tour(grid, RES, ORIGIN, ORIGIN, seed=42)
    assert result["waypoints"] in ([], [(0.0, 0.0)])
    assert any("no plannable waypoint" in n for n in result["notes"])


def test_clearance_measures_the_tightest_pinch_not_the_average():
    """What wedges the rover is the narrowest point of a corridor, not its typical width - so the score has to be a minimum over the path."""
    from regolith_planner.tour import CLEARANCE_SEARCH_CELLS
    from regolith_planner.tour import path_clearance_cells

    grid = _grid()
    # A wide corridor with one narrow throat: walls at c = 46 and c = 54, closing to
    # c = 49 and c = 51 for a few rows.
    grid[:, 46] = LETHAL_COST
    grid[:, 54] = LETHAL_COST
    grid[58:62, 47:50] = LETHAL_COST
    grid[58:62, 51:54] = LETHAL_COST
    path = [(r, 50) for r in range(50, 70)]

    # Inside the throat the walls are one cell away on both sides.
    assert path_clearance_cells(grid, path) == pytest.approx(1.0)
    # Well clear of it, the same corridor is limited only by the search cap (the walls
    # at c = 46 and c = 54 are 4 cells from c = 50).
    assert path_clearance_cells(grid, [(r, 50) for r in range(66, 70)]) == pytest.approx(4.0)
    assert CLEARANCE_SEARCH_CELLS == 4


def test_open_ground_scores_the_full_search_radius():
    from regolith_planner.tour import CLEARANCE_SEARCH_CELLS
    from regolith_planner.tour import path_clearance_cells

    assert path_clearance_cells(_grid(), [(60, 60), (61, 61)]) == float(CLEARANCE_SEARCH_CELLS)


def test_the_wider_corridor_wins_when_both_are_plannable():
    """Two ways round one obstacle - a one-cell slot and an open detour. Every candidate reaching the pool is already plannable, so the pool exists precisely to break this tie in favour of the route the rover can physically get through."""
    from regolith_planner.tour import path_clearance_cells

    grid = _grid()
    grid[58:63, 40:60] = LETHAL_COST  # a wall across the way
    grid[60, 50] = 0  # a one-cell slot through it

    slot = [(r, 50) for r in range(55, 66)]
    detour = [(r, 65) for r in range(55, 66)]  # around the end of the wall
    assert path_clearance_cells(grid, slot) < path_clearance_cells(grid, detour)


def test_a_lethal_spawn_raises():
    grid = _grid(fill=LETHAL_COST)
    with pytest.raises(ValueError, match="spawn"):
        plan_tour(grid, RES, ORIGIN, ORIGIN, seed=42)
