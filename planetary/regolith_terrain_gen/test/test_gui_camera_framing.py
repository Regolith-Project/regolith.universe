# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the opening GUI camera pose (see PROGRESS.md).

Two distinct ways this has already been got wrong, one guarded by each test below:
  1. Too far away to see the rover at all. The pose was "-110 -110 35" (~155 m, rover
     2-3 px) and then "-22 -22 13" (~32 m, rover ~13 px); both were reported by the
     user as Gazebo simply not showing the rover.
  2. Underground. A hand-picked absolute z (2.5 m) sat below the ~5.2 m local terrain,
     so the camera rendered the underside of the terrain.
Both are invisible to every other test in this suite - nothing else looks at <gui>.
"""

import numpy as np
import pytest
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.heightmap import build_heightmap
from regolith_terrain_gen.worldgen import _gui_camera_pose

# The rover is a 0.4 m chassis. Beyond roughly 10 m it stops being recognisable as a
# rover in the GUI's dark lunar lighting (measured: ~13 px of lit chassis at 32 m,
# ~43 px at 7 m in a 1200 px-wide window).
MAX_USEFUL_DISTANCE_M = 10.0
MIN_GROUND_CLEARANCE_M = 1.0

SEEDS = [42, 123, 7, 1, 2]


def _pose_and_terrain(seed):
    cfg = TerrainConfig(seed=seed)
    _raw, _visual, _craters, elevation_lookup = build_heightmap(
        cfg, np.random.default_rng(cfg.seed)
    )
    pose = [float(v) for v in _gui_camera_pose(cfg, elevation_lookup).split()]
    return cfg, elevation_lookup, pose


@pytest.mark.parametrize("seed", SEEDS)
def test_camera_is_close_enough_to_see_the_rover(seed):
    cfg, elevation_lookup, pose = _pose_and_terrain(seed)
    x, y, z = pose[0], pose[1], pose[2]
    sx, sy = cfg.spawn_zone_center
    spawn_z = elevation_lookup(sx, sy)

    distance = float(np.linalg.norm([sx - x, sy - y, spawn_z - z]))
    assert distance < MAX_USEFUL_DISTANCE_M, (
        f"seed {seed}: opening GUI camera is {distance:.1f} m from the rover - it will "
        f"render as a handful of pixels and read as 'Gazebo isn't showing the rover'"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_camera_is_above_the_terrain_beneath_it(seed):
    """Clearance is checked at the CAMERA's own (x, y), not the spawn point - terrain
    rising behind the rover is exactly how a pose that clears the spawn elevation can
    still end up buried."""
    _cfg, elevation_lookup, pose = _pose_and_terrain(seed)
    x, y, z = pose[0], pose[1], pose[2]

    clearance = z - elevation_lookup(x, y)
    assert clearance > MIN_GROUND_CLEARANCE_M, (
        f"seed {seed}: opening GUI camera is only {clearance:.2f} m above the ground "
        f"directly beneath it - at or below 0 it renders the underside of the terrain"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_camera_actually_points_at_the_rover(seed):
    """A close camera aimed the wrong way is just as blank as a distant one, and the
    pitch/yaw are computed rather than hardcoded, so they're worth pinning down."""
    cfg, elevation_lookup, pose = _pose_and_terrain(seed)
    x, y, z, _roll, pitch, yaw = pose
    sx, sy = cfg.spawn_zone_center
    spawn_z = elevation_lookup(sx, sy)

    # Unit vector the camera looks along, for gz's x-forward camera convention:
    # positive pitch tilts the view down.
    forward = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
    to_rover = np.array([sx - x, sy - y, spawn_z - z])
    to_rover /= np.linalg.norm(to_rover)

    off_axis_deg = np.degrees(np.arccos(np.clip(forward @ to_rover, -1.0, 1.0)))
    assert (
        off_axis_deg < 10.0
    ), f"seed {seed}: rover sits {off_axis_deg:.1f} deg off the camera's view axis"
