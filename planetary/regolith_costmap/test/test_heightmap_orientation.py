# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Guards the axis convention where the costmap crosses gz's.

The terrain generator writes the heightmap PNG in gz's axis order (first image
axis = world X); this package indexes [row = y, col = x], as does the planner.
The counterpart test lives in regolith_terrain_gen (test_heightmap_orientation.py)
and asserts the file is WRITTEN that way; this one asserts it is READ back that
way, so the two cannot silently drift apart.

Both are needed because the error is invisible to summary statistics: a transpose
preserves the elevation histogram, so total lethal-cell fraction is identical
whichever way round the array is read. Only per-cell positions differ.
"""

import json

from PIL import Image
import numpy as np
import pytest
from regolith_costmap.costmap_node import build_costmap
from regolith_costmap.costmap_node import load_heightmap

WORLD_SIZE_M = 200.0
HEIGHT_RANGE_M = 10.0
RES_PX = 129


def _write_manifest(tmp_path, surface_yx):
    """Writes surface_yx (indexed [y, x]) out the way the generator does: transposed,
    full-range, with the real (z_min, span) that range decodes to recorded beside it."""
    png_path = tmp_path / "heightmap.png"
    z_min = float(surface_yx.min())
    z_span = max(float(surface_yx.ptp()), 1e-9)
    normalized = (surface_yx - z_min) / z_span
    as_uint16 = (normalized * 65535).astype(np.uint16)
    Image.fromarray(np.ascontiguousarray(as_uint16.T), mode="I;16").save(png_path)

    manifest = {
        "world_size_m": WORLD_SIZE_M,
        "height_range_m": HEIGHT_RANGE_M,
        "heightmap_z_min_m": z_min,
        "heightmap_z_span_m": z_span,
        "heightmap_png": str(png_path),
        "rocks": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _spike_surface(x_m, y_m):
    """A single tall spike at world (x_m, y_m), on otherwise flat ground."""
    surface = np.zeros((RES_PX, RES_PX))
    half = WORLD_SIZE_M / 2.0
    col = int(round((x_m + half) / WORLD_SIZE_M * (RES_PX - 1)))
    row = int(round((y_m + half) / WORLD_SIZE_M * (RES_PX - 1)))
    surface[row, col] = HEIGHT_RANGE_M
    return surface


def test_load_heightmap_returns_row_y_col_x(tmp_path):
    """A spike placed at (+60, 0) must come back at [y=0, x=+60], not [+60, 0]."""
    surface = _spike_surface(60.0, 0.0)
    manifest = _write_manifest(tmp_path, surface)

    loaded = load_heightmap(manifest)

    peak_row, peak_col = np.unravel_index(np.argmax(loaded), loaded.shape)
    expect_row, expect_col = np.unravel_index(np.argmax(surface), surface.shape)
    assert (peak_row, peak_col) == (expect_row, expect_col)
    # And explicitly: not the transposed position, which is the actual failure mode.
    assert (peak_row, peak_col) != (expect_col, expect_row)


def test_asymmetric_surface_survives_the_round_trip(tmp_path):
    """Every cell, not just the peak - an asymmetric ramp so a transpose cannot hide."""
    ys, xs = np.mgrid[0:RES_PX, 0:RES_PX]
    surface = 0.7 * ys / RES_PX * HEIGHT_RANGE_M + 0.3 * xs / RES_PX * HEIGHT_RANGE_M
    manifest = _write_manifest(tmp_path, surface)

    loaded = load_heightmap(manifest)

    # Absolute metres, cell for cell - no rescaling on either side. Only the 16-bit
    # quantization of the PNG separates the two.
    assert np.allclose(loaded, surface, atol=surface.ptp() / 65535 * 2)


def test_lethal_fraction_cannot_detect_a_transpose(tmp_path):
    """Documents WHY the two orientation tests exist: the obvious metric is blind.

    If this ever fails it means a transpose became detectable from the summary
    statistic, which would be good news - but the per-cell tests above are the
    ones that actually guard the convention.
    """
    rng = np.random.default_rng(0)
    surface = rng.normal(size=(RES_PX, RES_PX)).cumsum(axis=0).cumsum(axis=1)
    surface = (surface - surface.min()) / surface.ptp() * HEIGHT_RANGE_M
    manifest = _write_manifest(tmp_path, surface)

    correct, _, _, _ = build_costmap(manifest, surface, 1.0, 0.3, 20.0)
    transposed, _, _, _ = build_costmap(manifest, surface.T, 1.0, 0.3, 20.0)

    lethal_correct = (correct >= 100).mean()
    lethal_transposed = (transposed >= 100).mean()
    assert lethal_correct == pytest.approx(lethal_transposed, abs=0.005)
    # ... while the per-cell verdicts genuinely differ.
    assert ((correct >= 100) != (transposed >= 100)).mean() > 0.001
