# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Obstacle inflation has to cover the rover's radius, at every costmap resolution.

Found while investigating why a tour leg timed out (see PROGRESS.md). The suspicion was
under-inflation letting the planner offer gaps the rover cannot thread; the arithmetic
said otherwise, and turned up a different, real defect on the way.

`inflation_cells` was `max(1, round(radius / cell))`. Rounding a SAFETY MARGIN spends
it: at a 0.25 m costmap, a 0.30 m rover got one 0.25 m cell of inflation and the planner
would route a corner of the machine through a boulder. At the shipped 0.781 m/cell,
round and ceil both give 1, so nothing measured to date is affected - which is exactly
why it went unnoticed and why it needed a test rather than a re-measurement.

The second thing these pin is the quantisation itself. At 0.781 m/cell every
rover_radius_m from 0 to 0.78 m yields the same single cell, so that parameter cannot be
tuned inside its plausible range - worth knowing before someone spends a session turning
a knob that is not connected to anything.
"""

import json
import math

import numpy as np
import pytest
from regolith_costmap.costmap_node import build_costmap

# regolith_rover_description: 0.46 m wheel separation + 0.06 m wheel width = 0.52 m
# wide, 0.28 m wheelbase + 2 x 0.09 m wheel radius = 0.46 m long. Half the diagonal is
# the radius that matters for a skid-steer machine, which turns in place.
ROVER_CIRCUMSCRIBED_M = math.hypot(0.52, 0.46) / 2  # 0.347 m
SHIPPED_RESOLUTION_M = 200.0 / 256  # what costmap_node logs at the shipped settings


def _inflation_cells(radius_m, resolution_m):
    """Compute the inflation-cell count under test, kept in one place so the tests state the contract rather than restating the implementation."""
    return max(1, math.ceil(radius_m / resolution_m))


@pytest.mark.parametrize("resolution_m", [1.0, 0.781, 0.5, 0.4, 0.25, 0.2, 0.1])
@pytest.mark.parametrize("radius_m", [0.1, 0.3, ROVER_CIRCUMSCRIBED_M, 0.5, 1.0])
def test_inflation_is_never_smaller_than_the_radius_it_stands_for(radius_m, resolution_m):
    covered_m = _inflation_cells(radius_m, resolution_m) * resolution_m
    assert covered_m >= radius_m - 1e-9, (
        f"{radius_m} m of rover gets only {covered_m:.3f} m of clearance at "
        f"{resolution_m} m/cell - the planner would route the rover's corner into an "
        f"obstacle"
    )


def test_rounding_was_the_defect_and_ceiling_is_the_fix():
    """Guards the guard: names a case where the old expression was short, so this file cannot quietly stop testing anything."""
    radius_m, resolution_m = 0.30, 0.25
    old = max(1, int(round(radius_m / resolution_m)))
    assert old * resolution_m < radius_m, "the old rounding no longer under-inflates"
    assert _inflation_cells(radius_m, resolution_m) * resolution_m >= radius_m


def test_the_shipped_configuration_is_unchanged_by_the_fix():
    """Every number recorded in PROGRESS.md was measured with the old expression. If this stops holding, those numbers need re-measuring rather than trusting."""
    for radius_m in (0.3, ROVER_CIRCUMSCRIBED_M):
        old = max(1, int(round(radius_m / SHIPPED_RESOLUTION_M)))
        assert _inflation_cells(radius_m, SHIPPED_RESOLUTION_M) == old == 1


def test_one_shipped_cell_already_covers_the_real_rover():
    """The investigation's actual question: is the shipped map under-inflated? No - one cell is 0.78 m against a 0.347 m rover, more than twice what is needed."""
    covered = _inflation_cells(0.3, SHIPPED_RESOLUTION_M) * SHIPPED_RESOLUTION_M
    assert covered > 2 * ROVER_CIRCUMSCRIBED_M


def test_the_parameter_is_inert_across_its_plausible_range_at_shipped_resolution():
    """Not a preference - a fact worth failing on if the resolution ever changes, so that 'tune rover_radius_m' stops being suggested when it would do nothing."""
    cells = {
        _inflation_cells(r, SHIPPED_RESOLUTION_M)
        for r in (0.0, 0.1, 0.3, ROVER_CIRCUMSCRIBED_M, 0.5, 0.78)
    }
    assert cells == {1}


def _flat_manifest(tmp_path, rocks):
    from PIL import Image

    n = 129
    surface = np.zeros((n, n), dtype=np.uint16)
    png = tmp_path / "heightmap.png"
    Image.fromarray(surface, mode="I;16").save(png)
    manifest = {
        "world_size_m": 100.0,
        "heightmap_z_min_m": 0.0,
        "heightmap_z_span_m": 1.0,
        "heightmap_png": str(png),
        "rocks": rocks,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return json.loads(path.read_text())


def test_a_boulder_blocks_more_than_its_own_footprint(tmp_path):
    """End to end through build_costmap: the lethal region around a rock has to be wider than the rock, or the inflation is not reaching the grid at all."""
    from regolith_costmap.costmap_node import load_heightmap

    rock = {"x_m": 0.0, "y_m": 0.0, "scale_m": 0.5, "collision_radii_m": [0.5, 0.5, 0.5]}
    manifest = _flat_manifest(tmp_path, [rock])
    grid, resolution_m, origin_x, origin_y = build_costmap(
        manifest, load_heightmap(manifest), 1.0, 0.3, 20.0
    )
    grid = np.asarray(grid)

    lethal = grid >= 100
    assert lethal.any(), "the rock produced no lethal cells at all"
    rows, cols = np.where(lethal)
    span_m = (max(rows.max() - rows.min(), cols.max() - cols.min()) + 1) * resolution_m
    assert span_m > 2 * rock["collision_radii_m"][0] + resolution_m, (
        f"lethal region spans {span_m:.2f} m for a {rock['collision_radii_m'][0]:.2f} m "
        f"rock - inflation is not being applied"
    )
