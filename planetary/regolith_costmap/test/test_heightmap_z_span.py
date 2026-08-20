# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Pins the vertical decode of the heightmap PNG, and the slope error it used to cause.

`save_heightmap_png` writes the surface FULL-RANGE (0 -> its own minimum, 65535 -> its
own maximum) and hands back the real (z_min, span) those endpoints mean. The costmap used
to ignore that and decode against `height_range_m`, the range the generator was
*configured* to be allowed to use - typically ~8 m of the configured 10 actually gets
occupied. Every elevation was therefore stretched by span/height_range_m and every slope
overstated by the same 1.20-1.25x, running an effective ~16 deg lethal threshold against
the 20 deg configured.

The error was conservative - it over-flagged ground, never under-flagged it - which is
exactly why it survived a full milestone invisibly. These tests make it visible.
See PROGRESS.md "costmap decodes the wrong height span".
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


def _write_manifest(tmp_path, surface_yx, **overrides):
    """Encodes surface_yx exactly as save_heightmap_png does, manifest fields included."""
    png_path = tmp_path / "heightmap.png"
    z_min = float(surface_yx.min())
    z_span = max(float(surface_yx.ptp()), 1e-9)
    as_uint16 = ((surface_yx - z_min) / z_span * 65535).astype(np.uint16)
    Image.fromarray(np.ascontiguousarray(as_uint16.T), mode="I;16").save(png_path)

    manifest = {
        "world_size_m": WORLD_SIZE_M,
        "height_range_m": HEIGHT_RANGE_M,
        "heightmap_z_min_m": z_min,
        "heightmap_z_span_m": z_span,
        "heightmap_png": str(png_path),
        "rocks": [],
    }
    manifest.update(overrides)
    return manifest


def _ramp(rise_m, z_min_m=0.0):
    """A plane rising `rise_m` across the world in +x, so its slope is analytic."""
    _, xs = np.mgrid[0:RES_PX, 0:RES_PX]
    return z_min_m + xs / (RES_PX - 1) * rise_m


def test_decodes_to_true_metres_not_the_configured_range(tmp_path):
    """A surface spanning 8 m of a configured 10 must come back spanning 8, not 10."""
    surface = _ramp(rise_m=8.0)
    manifest = _write_manifest(tmp_path, surface)

    loaded = load_heightmap(manifest)

    assert loaded.ptp() == pytest.approx(8.0, abs=1e-3)
    # The old behaviour, stated explicitly so a regression reads as one.
    assert loaded.ptp() != pytest.approx(HEIGHT_RANGE_M, abs=0.1)


def test_absolute_elevation_is_preserved(tmp_path):
    """z_min is carried, not discarded - terrain 100 m up decodes 100 m up."""
    surface = _ramp(rise_m=8.0, z_min_m=100.0)
    manifest = _write_manifest(tmp_path, surface)

    loaded = load_heightmap(manifest)

    assert loaded.min() == pytest.approx(100.0, abs=1e-3)
    assert loaded.max() == pytest.approx(108.0, abs=1e-3)


def test_slopes_are_not_overstated_across_the_lethal_threshold(tmp_path):
    """The consequence that mattered: ground the rover can cross is not marked lethal.

    An 8 m rise over the 200 m world is a 2.29 deg plane. Decoded against the configured
    10 m it reads as 2.86 deg. With the threshold set between the two, the bug is the
    difference between an entirely traversable map and an entirely lethal one.
    """
    surface = _ramp(rise_m=8.0)
    manifest = _write_manifest(tmp_path, surface)
    true_slope_deg = np.degrees(np.arctan(8.0 / WORLD_SIZE_M))
    stretched_slope_deg = np.degrees(np.arctan(HEIGHT_RANGE_M / WORLD_SIZE_M))
    threshold_deg = (true_slope_deg + stretched_slope_deg) / 2.0

    correct, _, _, _ = build_costmap(manifest, load_heightmap(manifest), 1.0, 0.3, threshold_deg)
    stretched, _, _, _ = build_costmap(
        manifest, surface / 8.0 * HEIGHT_RANGE_M, 1.0, 0.3, threshold_deg
    )

    assert not (correct >= 100).any()
    assert (stretched >= 100).all()


def test_a_manifest_without_the_z_span_fields_is_refused(tmp_path):
    """Loudly, rather than silently falling back to height_range_m and being wrong again."""
    surface = _ramp(rise_m=8.0)
    manifest = _write_manifest(tmp_path, surface)
    del manifest["heightmap_z_span_m"]

    with pytest.raises(KeyError, match="Regenerate the world"):
        load_heightmap(manifest)


def test_the_generator_records_what_the_costmap_reads(tmp_path):
    """The two sides of the contract, joined - no hand-written manifest in between.

    This is the test that would have caught the original bug: it runs the real generator
    and the real decoder against each other, where the hand-rolled fixtures above only
    encode what this file believes the generator does.
    """
    pytest.importorskip("regolith_terrain_gen")
    from regolith_terrain_gen.config import TerrainConfig
    from regolith_terrain_gen.generate import generate_world

    cfg = TerrainConfig(seed=42)
    generate_world(cfg, tmp_path, start_paused=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    loaded = load_heightmap(manifest)

    assert manifest["heightmap_z_span_m"] < cfg.height_range_m
    assert loaded.min() == pytest.approx(manifest["heightmap_z_min_m"], abs=1e-3)
    assert loaded.ptp() == pytest.approx(manifest["heightmap_z_span_m"], abs=1e-3)
