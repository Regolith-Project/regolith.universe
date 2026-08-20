# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""The planner used to refuse outright when the rover's own cell was lethal.

Two things that really happen put it there: localization drift moving the
estimate inside an inflated rock, and the rover marking a keep-out zone around
the boulder it was just wedged against (costmap_node's hazard marking - the
rover is standing next to the hazard it just reported). Refusing in that state
strands the rover for the rest of the run, which is exactly the failure the
keep-out zones were added to fix, so the escape hatch is load-bearing.
"""

import numpy as np
from regolith_planner.astar import LETHAL_COST
from regolith_planner.planner_node import nearest_free_cell


def _free_grid(size=20):
    return np.zeros((size, size), dtype=np.int16)


def test_free_start_is_left_alone():
    grid = _free_grid()
    # The caller only asks when the start is lethal, but the nearest free cell
    # to a free cell is still an immediate neighbour, never further.
    found = nearest_free_cell(grid, (10, 10), 5)
    assert max(abs(found[0] - 10), abs(found[1] - 10)) == 1


def test_escapes_a_small_lethal_patch():
    grid = _free_grid()
    grid[9:12, 9:12] = LETHAL_COST  # rover sits in the middle of a 3x3 keep-out
    found = nearest_free_cell(grid, (10, 10), 5)
    assert grid[found] < LETHAL_COST
    assert max(abs(found[0] - 10), abs(found[1] - 10)) == 2  # first free ring


def test_prefers_the_closest_free_cell():
    grid = _free_grid()
    grid[8:13, 8:13] = LETHAL_COST
    grid[8, 10] = 0  # one free cell in the middle of the top edge
    found = nearest_free_cell(grid, (10, 10), 5)
    assert found == (8, 10)


def test_gives_up_rather_than_searching_forever():
    grid = np.full((20, 20), LETHAL_COST, dtype=np.int16)
    assert nearest_free_cell(grid, (10, 10), 5) is None


def test_free_cell_beyond_the_search_radius_is_not_used():
    """A cell 6 rings away is too far to be a sensible plan start - saying so is
    better than starting a path the rover cannot reach."""
    grid = np.full((20, 20), LETHAL_COST, dtype=np.int16)
    grid[16, 10] = 0
    assert nearest_free_cell(grid, (10, 10), 5) is None
    assert nearest_free_cell(grid, (10, 10), 6) == (16, 10)


def test_search_stays_inside_the_grid():
    grid = _free_grid()
    grid[0:3, 0:3] = LETHAL_COST
    found = nearest_free_cell(grid, (0, 0), 5)  # corner: most of the ring is off-grid
    assert found is not None
    assert grid[found] < LETHAL_COST
