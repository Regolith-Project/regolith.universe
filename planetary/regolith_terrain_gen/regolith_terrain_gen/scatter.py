# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Scatter rock instances across the world, keeping the spawn zone clear."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.rocks import RockVariant


@dataclass
class RockInstance:
    x_m: float
    y_m: float
    z_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    scale_m: float
    variant: str
    # Ellipsoid semi-axes in metres standing in for the mesh as collision geometry
    # (mesh collision is a no-op in this install - see rocks.RockVariant).
    collision_radii_m: tuple


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """SDF pose convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def seat_rock_z(
    x_m: float,
    y_m: float,
    scale_m: float,
    world_vertices: np.ndarray,
    elevation_lookup: Callable[[float, float], float],
    embed_frac: float,
) -> float:
    """Model-origin z that rests this rock ON the terrain, then sinks it by embed_frac.

    Placement used to be ``elevation_lookup(x, y) - 0.12 * scale``, i.e. it put the mesh
    ORIGIN a fixed fraction of the scale below the ground and assumed that buried the
    rock. It does not. Meshes are normalized by their bounding RADIUS, but
    displace_rock's anisotropic stretch leaves each variant's lowest vertex somewhere
    between ~0.51 and 1.0 units below the origin - so every rock hovered
    (bottom_offset - 0.12) x scale above the surface. Measured across 12 variants that
    is 0.39-0.88 x scale of clear air: up to ~2.1 m under a 2.4 m boulder, which is
    exactly the "big floating rocks" the terrain was reported for.

    The fix seats the rock off its ACTUAL geometry instead of a constant. Each vertex,
    already scaled and rotated into world axes, must clear the ground beneath its own
    (x, y) - so the resting origin height is::

        z = max over vertices of ( terrain(x + vx, y + vy) - vz )

    which is exact for these convex-ish boulders: it drops the rock until its first
    vertex touches down and no vertex is underground. Taking the max over vertices (not
    the terrain height at the rock's centre) is also what keeps rocks on SLOPED ground
    from hanging off their downhill edge - the uphill vertices set the height, and the
    downhill side gets buried rather than floating.

    embed_frac then sinks it a little further so it reads as a boulder partly buried in
    regolith rather than one set down on the surface.
    """
    ground_under_vertex = np.array(
        [elevation_lookup(x_m + float(v[0]), y_m + float(v[1])) for v in world_vertices]
    )
    resting_z = float(np.max(ground_under_vertex - world_vertices[:, 2]))
    return resting_z - embed_frac * scale_m


def scatter_rocks(
    cfg: TerrainConfig,
    rng: np.random.Generator,
    variants: list,
    elevation_lookup: Callable[[float, float], float],
) -> list:
    half_world = cfg.world_size_m / 2.0
    spawn_cx, spawn_cy = cfg.spawn_zone_center
    rocks = []
    for _ in range(cfg.rock_count):
        scale = rng.uniform(cfg.rock_scale_min_m, cfg.rock_scale_max_m)
        keep_out = cfg.spawn_zone_radius_m + scale
        for _ in range(50):
            x = rng.uniform(-half_world + scale, half_world - scale)
            y = rng.uniform(-half_world + scale, half_world - scale)
            if np.hypot(x - spawn_cx, y - spawn_cy) > keep_out:
                break
        else:
            continue

        variant: RockVariant = variants[rng.integers(0, len(variants))]
        # A small random tilt as well as yaw: boulders settle at whatever angle the
        # ground and their own shape leave them, and yaw alone left every rock sitting
        # on the same axis of its displaced icosphere. Kept modest so they still read as
        # resting rather than balanced on a point.
        roll = float(rng.uniform(-cfg.rock_tilt_max_rad, cfg.rock_tilt_max_rad))
        pitch = float(rng.uniform(-cfg.rock_tilt_max_rad, cfg.rock_tilt_max_rad))
        yaw = float(rng.uniform(0.0, 2.0 * np.pi))

        world_vertices = (variant.vertices * scale) @ _rotation_matrix(roll, pitch, yaw).T
        z = seat_rock_z(x, y, scale, world_vertices, elevation_lookup, cfg.rock_embed_frac)

        rocks.append(
            RockInstance(
                x_m=x,
                y_m=y,
                z_m=z,
                roll_rad=roll,
                pitch_rad=pitch,
                yaw_rad=yaw,
                scale_m=scale,
                variant=variant.name,
                collision_radii_m=tuple(float(r * scale) for r in variant.collision_radii),
            )
        )
    return rocks
