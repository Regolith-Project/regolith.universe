# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""The tour's route, on real generated terrain, judged by the planner that will drive it.

This is the check the hardcoded waypoint list never had. Those five (x, y) pairs were
picked by hand and were fine when they were picked; then the terrain went to res40 and
rock collision started working, and nothing re-examined them. Measured afterwards against
the shipped costmap:

    seed  route  lethal waypoints  legs plannable  legs needing avoidance
      42    old                 1             3/5                     2/5
       7    old                 1             3/5                     2/5
     123    old                 2             2/5                     4/5

Two of five legs plannable means most of an unattended demo was the rover sitting still
waiting out a 90 s timeout on a goal the planner had already refused.

The route is now drawn from the costmap (`regolith_planner.tour`), and this file holds
it to the standard the old one failed: on terrain generated from scratch, every waypoint
plannable and every leg reachable by the same A* the planner node runs. It crosses three
packages on purpose - terrain generator, costmap and planner - because that is where the
old failure lived: each package was individually fine.
"""

import json
import math

import numpy as np
import pytest
from regolith_costmap.costmap_node import build_costmap
from regolith_costmap.costmap_node import load_heightmap
from regolith_planner.astar import LETHAL_COST
from regolith_planner.astar import plan_path
from regolith_planner.tour import plan_tour
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.generate import generate_world

SEEDS = [42, 7, 123]
SPAWN_XY = (0.0, 0.0)

# The costmap settings hello_moon.launch.py gives costmap_node. At runtime the tour
# subscribes to /costmap rather than rebuilding it, precisely so these cannot drift -
# a test has to rebuild, so it states them in one place.
RESOLUTION_M = 1.0
ROVER_RADIUS_M = 0.3
SLOPE_LETHAL_DEG = 20.0


@pytest.fixture(scope="module")
def costmaps(tmp_path_factory):
    grids = {}
    for seed in SEEDS:
        world = tmp_path_factory.mktemp(f"seed{seed}")
        generate_world(TerrainConfig(seed=seed), world, start_paused=True)
        manifest = json.loads((world / "manifest.json").read_text())
        grid, resolution, origin_x, origin_y = build_costmap(
            manifest,
            load_heightmap(manifest),
            RESOLUTION_M,
            ROVER_RADIUS_M,
            SLOPE_LETHAL_DEG,
        )
        grids[seed] = (np.asarray(grid), resolution, origin_x, origin_y)
    return grids


def _cell(xy, resolution, origin_x, origin_y):
    return (int((xy[1] - origin_y) / resolution), int((xy[0] - origin_x) / resolution))


@pytest.mark.parametrize("seed", SEEDS)
def test_no_waypoint_lands_on_terrain_the_planner_refuses(costmaps, seed):
    grid, resolution, origin_x, origin_y = costmaps[seed]
    route = plan_tour(grid, resolution, origin_x, origin_y, seed=seed)["waypoints"]
    assert route, f"seed {seed}: no route built"

    for i, waypoint in enumerate(route):
        row, col = _cell(waypoint, resolution, origin_x, origin_y)
        block = grid[row - 1 : row + 2, col - 1 : col + 2]
        assert not (block >= LETHAL_COST).any(), (
            f"seed {seed}: waypoint {i + 1} at {waypoint} is on or beside a lethal cell - "
            f"the planner will refuse it and the leg will burn its 90 s timeout"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_every_leg_is_plannable_by_the_planner_that_will_drive_it(costmaps, seed):
    """Not a flood fill, not a straight line - the same plan_path planner_node calls."""
    grid, resolution, origin_x, origin_y = costmaps[seed]
    route = plan_tour(grid, resolution, origin_x, origin_y, seed=seed)["waypoints"]
    points = [SPAWN_XY] + list(route)

    for i in range(len(points) - 1):
        path = plan_path(
            grid,
            _cell(points[i], resolution, origin_x, origin_y),
            _cell(points[i + 1], resolution, origin_x, origin_y),
        )
        assert path, f"seed {seed}: leg {i + 1}, {points[i]} -> {points[i + 1]}, has no path"


@pytest.mark.parametrize("seed", SEEDS)
def test_the_route_is_a_loop_of_short_legs(costmaps, seed):
    grid, resolution, origin_x, origin_y = costmaps[seed]
    route = plan_tour(grid, resolution, origin_x, origin_y, seed=seed)["waypoints"]

    assert route[-1] == SPAWN_XY, f"seed {seed}: the tour does not return to the start"
    points = [SPAWN_XY] + list(route)
    legs = [math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
    assert max(legs) <= 20.0 + 1e-6, (
        f"seed {seed}: longest leg {max(legs):.1f} m - long traverses are what the "
        f"short-leg design avoids (see tour_mission.py)"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_the_rover_has_to_avoid_something_on_most_legs(costmaps, seed):
    """A tour of open regolith would drive perfectly and demonstrate nothing. The old hand-picked route managed 2 of 5 on two of these seeds."""
    from regolith_planner.tour import _straight_line_blocked

    grid, resolution, origin_x, origin_y = costmaps[seed]
    route = plan_tour(grid, resolution, origin_x, origin_y, seed=seed)["waypoints"]
    points = [SPAWN_XY] + list(route)
    blocked = sum(
        _straight_line_blocked(
            grid,
            _cell(points[i], resolution, origin_x, origin_y),
            _cell(points[i + 1], resolution, origin_x, origin_y),
        )
        for i in range(len(points) - 1)
    )
    assert blocked >= 3, (
        f"seed {seed}: only {blocked} of {len(points) - 1} legs have a blocked direct "
        f"line, so the planner is barely being asked to route around anything"
    )


def test_the_old_hardcoded_route_would_still_fail(costmaps):
    """Guards the guard. If these assertions cannot fail, they prove nothing - and the route they replaced is the one case known to be bad."""
    old_route = [(12.0, 8.0), (18.0, -4.0), (4.0, -14.0), (-10.0, -6.0), (0.0, 0.0)]
    offenders = 0
    for seed in SEEDS:
        grid, resolution, origin_x, origin_y = costmaps[seed]
        for waypoint in old_route:
            row, col = _cell(waypoint, resolution, origin_x, origin_y)
            if grid[row, col] >= LETHAL_COST:
                offenders += 1
    assert offenders >= 3, (
        "the hardcoded waypoints are no longer on lethal cells - either the terrain "
        "changed again or this check has stopped measuring what it thinks it does"
    )
