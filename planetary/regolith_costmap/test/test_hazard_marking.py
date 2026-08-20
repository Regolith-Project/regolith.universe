# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep-out zones learned from wedge events.

The a-priori costmap knows every rock's footprint but not whether the gap
between two of them is really drivable. A wedge is information the map did not
have, and without recording it the planner keeps routing through the same gap
after every recovery - measured as 64 stuck events across three acceptance
runs, each one re-approaching the obstacle that caused the last (PROGRESS.md).
"""

import numpy as np
from regolith_costmap.costmap_node import stamp_hazard


def test_marks_a_disc_lethal():
    grid = np.zeros((21, 21), dtype=np.int16)
    marked = stamp_hazard(grid, 10, 10, 2)
    assert grid[10, 10] == 100
    assert grid[10, 12] == 100  # on the radius
    assert grid[10, 13] == 0  # outside it
    assert marked == int((grid == 100).sum())


def test_hazards_accumulate():
    """A second wedge must not erase the keep-out zone from the first."""
    grid = np.zeros((21, 21), dtype=np.int16)
    stamp_hazard(grid, 5, 5, 2)
    stamp_hazard(grid, 15, 15, 2)
    assert grid[5, 5] == 100
    assert grid[15, 15] == 100


def test_clipped_at_the_grid_edge():
    grid = np.zeros((21, 21), dtype=np.int16)
    marked = stamp_hazard(grid, 0, 0, 3)
    assert grid[0, 0] == 100
    assert 0 < marked < 21 * 21


def test_does_not_lower_existing_costs():
    """Stamping is one-way: it can only make a cell lethal, never traversable."""
    grid = np.full((21, 21), 60, dtype=np.int16)
    stamp_hazard(grid, 10, 10, 2)
    assert grid[10, 10] == 100
    assert grid[0, 0] == 60
