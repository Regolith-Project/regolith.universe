# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the "rover seems to be underground" bug (see PROGRESS.md):
the rendered heightmap and the collision boxes the rover actually rests on must be
the same surface, everywhere, for every seed - not just close enough on average.
"""

import numpy as np
import pytest
from PIL import Image

from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.heightmap import (
    _build_smoothed_surface,
    build_heightmap,
    build_terrain_collision_boxes_sdf,
    save_heightmap_png,
)

WHEEL_RADIUS_M = 0.09


def _collision_top_z(grid: dict, x: float, y: float) -> float:
    """Height of the tilted collision plane at (x, y), independent of the
    production _synthesize_visual_heightmap code path (so this test doesn't just
    check the synthesis function against itself)."""
    col = int(np.clip(round((x + grid["half_world"]) / grid["cell_size_m"] - 0.5), 0, grid["cols_blocks"] - 1))
    row = int(np.clip(round((y + grid["half_world"]) / grid["cell_size_m"] - 0.5), 0, grid["rows_blocks"] - 1))
    h0 = grid["surface"][row, col]
    gx = grid["grad_x"][row, col]
    gy = grid["grad_y"][row, col]
    dx = x - grid["xs"][col]
    dy = y - grid["ys"][row]
    return h0 + gx * dx + gy * dy


@pytest.mark.parametrize("seed", [42, 123, 7, 1, 2])
def test_visual_heightmap_matches_collision_surface(seed):
    cfg = TerrainConfig(seed=seed)
    rng = np.random.default_rng(cfg.seed)
    raw_heightmap, visual_heightmap, craters, elevation_lookup = build_heightmap(cfg, rng)
    grid = _build_smoothed_surface(raw_heightmap, cfg)

    rng2 = np.random.default_rng(0)
    max_gap = 0.0
    for _ in range(500):
        r = rng2.uniform(0, cfg.spawn_zone_radius_m)
        theta = rng2.uniform(0, 2 * np.pi)
        x, y = r * np.cos(theta), r * np.sin(theta)
        gap = abs(elevation_lookup(x, y) - _collision_top_z(grid, x, y))
        max_gap = max(max_gap, gap)

    # A generous margin over the ~1.8 cm nearest-cell-rounding noise measured between
    # this test's independent lookup and the production synthesis path - well under
    # the 9 cm wheel radius that made the old (10-35 cm) mismatch visible.
    assert max_gap < 0.05, (
        f"seed {seed}: visual/collision gap {max_gap:.3f} m exceeds the wheel "
        f"radius ({WHEEL_RADIUS_M} m) - the rover would visibly sink into or "
        f"float above the rendered terrain"
    )


@pytest.mark.parametrize("seed", [42, 123, 7, 1, 2])
def test_rendered_png_decodes_back_to_absolute_surface(seed, tmp_path):
    """The saved PNG, decoded the way gz-sim actually renders it, must reproduce the
    absolute-metre visual_heightmap - i.e. the drawn ground must land on the collision
    surface, not a stretched copy of it.

    gz-sim's ogre2 heightmap min/max-normalizes the image: it stretches whatever pixel
    range the PNG contains to fill <size> z, drawing the lowest pixel at <pos> z. It does
    NOT map pixel/65535 -> height linearly. This test replicates that decode from the
    (z_min, z_span) save_heightmap_png reports for <pos> z / <size> z, and checks the
    result equals the source surface to within 16-bit quantization.

    Regression guard for the "rover STILL underground" bug: the previous fix encoded only
    a PARTIAL pixel range and left <size> z at the fixed height_range_m, assuming a linear
    decode; gz's min/max stretch then lifted the rendered ground ~0.2-0.5 m above the
    collision boxes. That mistake reproduces here as a large gap, even though the
    visual/collision surfaces agree in absolute metres (the test above passes)."""
    cfg = TerrainConfig(seed=seed)
    rng = np.random.default_rng(cfg.seed)
    _raw, visual_heightmap, _craters, _lookup = build_heightmap(cfg, rng)

    png_path = tmp_path / "heightmap.png"
    z_min, z_span = save_heightmap_png(visual_heightmap, png_path)

    pixels = np.asarray(Image.open(png_path)).astype(np.float64)
    # gz decode for a full-range PNG: rendered = pos_z + (pixel/65535) * size_z.
    rendered = z_min + (pixels / 65535.0) * z_span

    max_err = float(np.abs(rendered - visual_heightmap).max())
    quantization_m = z_span / 65535.0
    assert max_err < quantization_m + 1e-6, (
        f"seed {seed}: gz-decoded heightmap differs from the absolute surface by "
        f"{max_err:.4f} m (>{quantization_m:.5f} m quantization) - the rendered ground "
        f"would not sit on the collision boxes"
    )


def test_collision_sdf_still_generates_from_raw_heightmap():
    """build_terrain_collision_boxes_sdf must keep taking the RAW heightmap (not
    the synthesized visual one) - feeding it the already-smoothed surface would
    double-smooth and silently drift collision geometry away from what
    build_heightmap's elevation_lookup promises."""
    cfg = TerrainConfig(seed=42)
    rng = np.random.default_rng(cfg.seed)
    raw_heightmap, visual_heightmap, craters, elevation_lookup = build_heightmap(cfg, rng)

    sdf_from_raw = build_terrain_collision_boxes_sdf(raw_heightmap, cfg)
    sdf_from_visual = build_terrain_collision_boxes_sdf(visual_heightmap, cfg)

    assert sdf_from_raw != sdf_from_visual
