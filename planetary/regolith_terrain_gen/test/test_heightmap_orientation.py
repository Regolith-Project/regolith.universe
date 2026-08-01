# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the "rocks float in the air" bug (see PROGRESS.md).

gz maps a heightmap image's first axis to world X and its second to world Y. Every array
in heightmap.py is indexed [row = y, col = x]. So the PNG handed to gz must be the
TRANSPOSE of the array, or the terrain renders mirrored about the x = y diagonal while
collision geometry and elevation_lookup stay in the array's own convention - which is
exactly what left rocks hanging in mid-air over ground that was drawn somewhere else.

The reason this went unnoticed for so long is that every previous check compared arrays
against arrays. The collision boxes, elevation_lookup and the visual array are all built
in this module's convention, so they all agreed with each other and with the rocks; only
the pixels gz actually rendered disagreed. These tests therefore assert on the ENCODED
FILE, which is the one artefact that crosses into gz's convention.
"""

import numpy as np
from PIL import Image

from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.heightmap import build_heightmap, save_heightmap_png


def _decode(path, z_min, z_span):
    """Decode the PNG the way gz does: full pixel range stretched into [z, z + size]."""
    img = np.asarray(Image.open(path)).astype(np.float64)
    lo, hi = img.min(), img.max()
    return z_min + (img - lo) / (hi - lo) * z_span


def test_png_is_the_transpose_of_the_array(tmp_path):
    """An asymmetric surface must come back transposed, not as-written."""
    n = 65
    yy, xx = np.mgrid[0:n, 0:n]
    # Deliberately asymmetric about the diagonal: height depends on x and y differently.
    surface = 3.0 * (xx / (n - 1)) + 11.0 * (yy / (n - 1)) ** 2

    png = tmp_path / "hm.png"
    z_min, z_span = save_heightmap_png(surface, png)
    decoded = _decode(png, z_min, z_span)

    assert np.allclose(decoded, surface.T, atol=1e-3), "PNG is not the transpose of the array"
    assert not np.allclose(decoded, surface, atol=1e-3), "PNG was written untransposed"


def test_a_spike_lands_at_the_right_world_position(tmp_path):
    """The bug in the terms it was actually found in: a spike at (+60, 0) must not
    render at (0, +60). Positions are computed in gz's convention - image axis 0 is
    world X, axis 1 is world Y."""
    cfg = TerrainConfig(seed=42)
    n = cfg.heightmap_resolution_px
    world = cfg.world_size_m
    ppm = (n - 1) / world
    half = world / 2.0

    yy, xx = np.mgrid[0:n, 0:n]
    spike_x, spike_y = 60.0, 0.0
    cx, cy = (spike_x + half) * ppm, (spike_y + half) * ppm
    surface = 25.0 * np.exp(-((np.hypot(xx - cx, yy - cy) / (10.0 * ppm)) ** 2))

    png = tmp_path / "hm.png"
    z_min, z_span = save_heightmap_png(surface, png)
    decoded = _decode(png, z_min, z_span)

    # gz convention: decoded[i, j] is the height at world x = i, y = j.
    i, j = np.unravel_index(np.argmax(decoded), decoded.shape)
    got_x = i / ppm - half
    got_y = j / ppm - half
    assert abs(got_x - spike_x) < 1.0, f"spike rendered at x={got_x:.1f}, expected {spike_x}"
    assert abs(got_y - spike_y) < 1.0, f"spike rendered at y={got_y:.1f}, expected {spike_y}"


def test_rendered_surface_matches_elevation_lookup(tmp_path):
    """The property that actually matters: the ground gz draws under any (x, y) is the
    ground elevation_lookup reports there - which is what rocks are seated on."""
    cfg = TerrainConfig(seed=7)
    rng = np.random.default_rng(cfg.seed)
    _, visual, _, elevation_lookup = build_heightmap(cfg, rng)

    png = tmp_path / "hm.png"
    z_min, z_span = save_heightmap_png(visual, png)
    decoded = _decode(png, z_min, z_span)

    n = cfg.heightmap_resolution_px
    ppm = (n - 1) / cfg.world_size_m
    half = cfg.world_size_m / 2.0

    # Sample at exact heightmap POSTS. elevation_lookup interpolates bilinearly between
    # posts (as gz does when it draws the surface), and this test is about the encoding -
    # orientation, full-range stretch, z mapping - not about interpolation, so comparing
    # at posts isolates the thing under test: on a post, bilinear reduces to the post's
    # own value and any difference left is the encoding.
    worst = 0.0
    rng2 = np.random.default_rng(0)
    for _ in range(400):
        i, j = rng2.integers(1, n - 1, size=2)
        x = i / ppm - half
        y = j / ppm - half
        worst = max(worst, abs(float(decoded[i, j]) - elevation_lookup(x, y)))
    # Same post, same surface: only 16-bit quantisation should remain.
    assert worst < 0.01, f"rendered ground differs from elevation_lookup by up to {worst:.4f} m"
