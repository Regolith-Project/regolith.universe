# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Scatter rock instances across the world, keeping the spawn zone clear."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from regolith_terrain_gen.config import TerrainConfig


@dataclass
class RockInstance:
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float
    scale_m: float
    variant: str


def scatter_rocks(
    cfg: TerrainConfig,
    rng: np.random.Generator,
    variant_names: list,
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

        terrain_z = elevation_lookup(x, y)
        embed = 0.12 * scale  # partially embed so rocks don't look like they're floating
        rocks.append(
            RockInstance(
                x_m=x,
                y_m=y,
                z_m=terrain_z - embed,
                yaw_rad=rng.uniform(0.0, 2.0 * np.pi),
                scale_m=scale,
                variant=variant_names[rng.integers(0, len(variant_names))],
            )
        )
    return rocks
