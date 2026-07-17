# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Compose the base terrain, regional slope, and crater field into a heightmap."""

from pathlib import Path

import numpy as np
from PIL import Image

from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.craters import apply_craters, place_craters
from regolith_terrain_gen.noise import fbm


def build_heightmap(cfg: TerrainConfig, rng: np.random.Generator) -> tuple:
    """Returns (heightmap_meters, craters, elevation_lookup) where elevation_lookup(x_m, y_m)
    samples terrain height in meters at a world-frame coordinate."""
    n = cfg.heightmap_resolution_px
    px_per_m = (n - 1) / cfg.world_size_m

    base = fbm(
        (n, n),
        rng,
        octaves=cfg.fbm_octaves,
        base_cell=cfg.fbm_base_cell_px,
        lacunarity=cfg.fbm_lacunarity,
        gain=cfg.fbm_gain,
    )
    heightmap = base * cfg.fbm_weight_m

    # Gentle regional slope in a random direction.
    slope_rad = np.deg2rad(cfg.regional_slope_deg)
    rise_over_world = np.tan(slope_rad) * cfg.world_size_m
    slope_dir = rng.uniform(0.0, 2.0 * np.pi)
    half_world = cfg.world_size_m / 2.0
    lin = np.linspace(-half_world, half_world, n)
    gx, gy = np.meshgrid(lin, lin)
    heightmap += (gx * np.cos(slope_dir) + gy * np.sin(slope_dir)) / cfg.world_size_m * rise_over_world

    craters = place_craters(cfg, rng)
    apply_craters(heightmap, cfg, craters, px_per_m)

    # Normalize to exactly the target height range, preserving relative shape.
    heightmap -= heightmap.min()
    span = heightmap.max()
    if span > 1e-9:
        heightmap *= cfg.height_range_m / span

    def elevation_lookup(x_m: float, y_m: float) -> float:
        px = int(np.clip((x_m + half_world) * px_per_m, 0, n - 1))
        py = int(np.clip((y_m + half_world) * px_per_m, 0, n - 1))
        return float(heightmap[py, px])

    return heightmap, craters, elevation_lookup


def save_heightmap_png(heightmap: np.ndarray, path: Path) -> None:
    """Save as a 16-bit grayscale PNG, the format Gazebo's heightmap geometry expects."""
    normalized = heightmap / heightmap.max() if heightmap.max() > 0 else heightmap
    as_uint16 = (np.clip(normalized, 0.0, 1.0) * 65535).astype(np.uint16)
    Image.fromarray(as_uint16, mode="I;16").save(path)
