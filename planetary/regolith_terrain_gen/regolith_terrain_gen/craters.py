# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Power-law crater field: placement, radial profile, and heightmap sculpting."""

from dataclasses import dataclass

import numpy as np

from regolith_terrain_gen.config import TerrainConfig


@dataclass
class Crater:
    x_m: float
    y_m: float
    diameter_m: float
    depth_m: float
    rim_height_m: float


def sample_power_law_diameters(
    count: int, d_min: float, d_max: float, exponent: float, rng: np.random.Generator
) -> np.ndarray:
    """Inverse-CDF sample from a pdf ~ d^-exponent on [d_min, d_max].

    This gives a lunar-like size-frequency distribution: many small craters,
    a handful of large ones, per N(>D) ~ D^-exponent.
    """
    u = rng.uniform(0.0, 1.0, size=count)
    a = 1.0 - exponent
    d_min_a = d_min**a
    d_max_a = d_max**a
    return (d_min_a + u * (d_max_a - d_min_a)) ** (1.0 / a)


def place_craters(cfg: TerrainConfig, rng: np.random.Generator) -> list:
    """Place craters biggest-first, rejecting positions that would encroach on the spawn zone."""
    diameters = sample_power_law_diameters(
        cfg.crater_count,
        cfg.crater_diameter_min_m,
        cfg.crater_diameter_max_m,
        cfg.crater_size_exponent,
        rng,
    )
    diameters = np.sort(diameters)[::-1]

    half_world = cfg.world_size_m / 2.0
    spawn_cx, spawn_cy = cfg.spawn_zone_center
    craters = []
    for diameter in diameters:
        radius = diameter / 2.0
        keep_out = cfg.spawn_zone_radius_m + radius
        for _ in range(50):
            x = rng.uniform(-half_world + radius, half_world - radius)
            y = rng.uniform(-half_world + radius, half_world - radius)
            if np.hypot(x - spawn_cx, y - spawn_cy) > keep_out:
                break
        else:
            continue  # couldn't find a valid spot in 50 tries; skip this crater

        depth = cfg.crater_depth_to_diameter * diameter
        rim_height = cfg.crater_rim_height_frac * depth
        craters.append(Crater(x, y, diameter, depth, rim_height))
    return craters


def crater_radial_profile(x_norm: np.ndarray, rim_width_frac: float) -> np.ndarray:
    """Bowl (paraboloid, depth 1 at center -> 0 at rim) plus a Gaussian raised rim at x_norm=1.

    x_norm is distance from crater center divided by crater radius.
    Returned values are in [-1, ~rim_amplitude], to be scaled by depth/rim_height by the caller.
    """
    bowl = np.where(x_norm <= 1.0, -(1.0 - x_norm**2), 0.0)
    rim = np.exp(-(((x_norm - 1.0) / rim_width_frac) ** 2))
    return bowl, rim


def apply_craters(
    heightmap: np.ndarray, cfg: TerrainConfig, craters: list, px_per_m: float
) -> None:
    """Sculpt craters directly into the heightmap array, in place."""
    h, w = heightmap.shape
    half_world = cfg.world_size_m / 2.0

    for crater in craters:
        radius_m = crater.diameter_m / 2.0
        # Profile is negligible beyond ~1.6x radius (rim gaussian tail); only touch that window.
        reach_m = radius_m * 1.6
        cx_px = (crater.x_m + half_world) * px_per_m
        cy_px = (crater.y_m + half_world) * px_per_m
        reach_px = int(np.ceil(reach_m * px_per_m))

        x0, x1 = max(0, int(cx_px - reach_px)), min(w, int(cx_px + reach_px))
        y0, y1 = max(0, int(cy_px - reach_px)), min(h, int(cy_px + reach_px))
        if x0 >= x1 or y0 >= y1:
            continue

        xs = (np.arange(x0, x1) - cx_px) / px_per_m
        ys = (np.arange(y0, y1) - cy_px) / px_per_m
        gx, gy = np.meshgrid(xs, ys)
        x_norm = np.hypot(gx, gy) / radius_m

        bowl, rim = crater_radial_profile(x_norm, cfg.crater_rim_width_frac)
        heightmap[y0:y1, x0:x1] += bowl * crater.depth_m + rim * crater.rim_height_m
