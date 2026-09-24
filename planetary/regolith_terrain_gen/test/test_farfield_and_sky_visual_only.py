# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the "world beyond the world" pass (sky.py / earth.py /
farfield.py): the far-field mesh must join the physics terrain with no visible
seam, and none of the sky/Earth/far-field geometry may ever become visible to
the frozen physics/planning path - not as collision, and not in manifest.json.
"""

from pathlib import Path
import re

import numpy as np
import pytest
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.earth import earth_model_sdf
from regolith_terrain_gen.earth import earth_position_m
from regolith_terrain_gen.earth import earth_radius_m
from regolith_terrain_gen.farfield import _square_ring_xy
from regolith_terrain_gen.farfield import build_farfield_mesh
from regolith_terrain_gen.farfield import farfield_model_sdf
from regolith_terrain_gen.heightmap import build_heightmap
from regolith_terrain_gen.sky import sky_model_sdf

# The OBJ writer formats vertex coordinates at %.3f (millimetre) precision, so
# nothing tighter than that is meaningful; 2 mm gives that a small margin.
BOUNDARY_TOLERANCE_M = 0.002


@pytest.mark.parametrize("seed", [42, 7, 123])
def test_farfield_boundary_matches_physics_terrain(seed, tmp_path):
    """Shell 0 of the far-field mesh must land exactly on the physics terrain's own elevation_lookup, everywhere around the 200 m square boundary - this is what makes the join seamless, not a visual tuning choice."""
    cfg = TerrainConfig(seed=seed)
    rng = np.random.default_rng(cfg.seed)
    _, _, _, elevation_lookup = build_heightmap(cfg, rng)

    out = tmp_path / "farfield.obj"
    build_farfield_mesh(cfg, elevation_lookup, out)

    x0, y0 = _square_ring_xy(cfg.world_size_m / 2.0, cfg.farfield_segments_per_side)
    expected = np.asarray(elevation_lookup(x0, y0), dtype=np.float64)

    verts = [line for line in out.read_text().splitlines() if line.startswith("v ")]
    m = cfg.farfield_segments_per_side
    first_ring_z = np.array([float(line.split()[3]) for line in verts[: 4 * m]])

    max_gap = float(np.max(np.abs(first_ring_z - expected)))
    assert max_gap < BOUNDARY_TOLERANCE_M, (
        f"seed {seed}: far-field mesh disagrees with the physics terrain by "
        f"{max_gap:.4f} m at the 200 m boundary - this would show as a visible seam"
    )


@pytest.mark.parametrize("seed", [42, 7])
def test_farfield_mesh_has_no_degenerate_or_nan_geometry(seed, tmp_path):
    cfg = TerrainConfig(seed=seed)
    rng = np.random.default_rng(cfg.seed)
    _, _, _, elevation_lookup = build_heightmap(cfg, rng)
    out = tmp_path / "farfield.obj"
    stats = build_farfield_mesh(cfg, elevation_lookup, out)

    text = out.read_text()
    verts = np.array(
        [[float(t) for t in line.split()[1:4]] for line in text.splitlines() if line.startswith("v ")]
    )
    assert np.isfinite(verts).all()
    assert stats["triangles"] > 0
    assert verts.shape[0] == stats["vertices"]


def test_farfield_and_sky_and_earth_carry_no_collision():
    """The SDF blocks these modules emit must be visual-only, full stop - collision here would silently change what the costmap/planner see, which the brief explicitly forbids."""
    cfg = TerrainConfig(seed=42)
    sky_sdf = sky_model_sdf(cfg, Path("/tmp/does_not_need_to_exist_for_this_check.png"), Path("/tmp/does_not_need_to_exist_for_this_check.obj"))
    earth_sdf = earth_model_sdf(cfg)
    farfield_sdf = farfield_model_sdf(cfg, Path("/tmp/does_not_need_to_exist_for_this_check.obj"))

    for name, block in [("sky", sky_sdf), ("earth", earth_sdf), ("farfield", farfield_sdf)]:
        # Strip XML comments first: these blocks legitimately mention "collision"
        # in their explanatory <!-- --> comments (that's the whole point), so the
        # real check is on actual SDF elements, not prose.
        stripped = re.sub(r"<!--.*?-->", "", block, flags=re.DOTALL)
        assert "<collision" not in stripped, f"{name} model must not define <collision>"
        assert "<static>true</static>" in block, f"{name} model must be static"


def test_sky_and_earth_are_not_shadow_casters():
    """Both must be unable to cast a shadow onto the terrain - the sky sphere surrounds everything and Earth sits kilometres out, so either casting a shadow would be a rendering bug (see sky.py/earth.py docstrings), not a feature."""
    cfg = TerrainConfig(seed=42)
    sky_sdf = sky_model_sdf(cfg, Path("/tmp/x.png"), Path("/tmp/x.obj"))
    earth_sdf = earth_model_sdf(cfg)
    assert "<cast_shadows>false</cast_shadows>" in sky_sdf
    assert "<cast_shadows>false</cast_shadows>" in earth_sdf


def test_earth_angular_diameter_matches_config():
    cfg = TerrainConfig(seed=42)
    radius = earth_radius_m(cfg)
    x, y, z = earth_position_m(cfg)
    distance = float(np.hypot(np.hypot(x, y), z))
    angular_diameter_deg = np.rad2deg(2.0 * np.arctan(radius / distance))
    assert angular_diameter_deg == pytest.approx(cfg.earth_angular_diameter_deg, abs=1e-6)


def test_earth_position_is_fixed_regardless_of_sun(monkeypatch=None):
    """Earth's sky position must depend only on its own azimuth/elevation/distance fields, never on the sun direction - it does not rise or set, and its phase comes from real-time lighting (see earth.py), not from repositioning it."""
    cfg_a = TerrainConfig(seed=42, sun_azimuth_deg=10.0, sun_elevation_deg=5.0)
    cfg_b = TerrainConfig(seed=42, sun_azimuth_deg=280.0, sun_elevation_deg=40.0)
    assert earth_position_m(cfg_a) == earth_position_m(cfg_b)


def test_shipped_farfield_matches_shipped_terrain_mesh_at_boundary(tmp_path):
    """End-to-end: the farfield.obj generate_world actually writes must meet the actual terrain.obj it also writes, at the 200 m boundary - not just the lower-level elevation_lookup/collision surface (see generate.py's drawn_surface_lookup wiring)."""
    from regolith_terrain_gen.generate import generate_world
    from regolith_terrain_gen.terrain_mesh import load_drawn_surface

    cfg = TerrainConfig(seed=42)
    out = tmp_path / "world"
    generate_world(cfg, out, start_paused=False)

    terrain_lookup = load_drawn_surface(out / "terrain.obj")
    half = cfg.world_size_m / 2.0
    x0, y0 = _square_ring_xy(half, cfg.farfield_segments_per_side)
    expected = np.asarray(terrain_lookup(x0, y0), dtype=np.float64)

    verts = [line for line in (out / "farfield.obj").read_text().splitlines() if line.startswith("v ")]
    m = cfg.farfield_segments_per_side
    first_ring_z = np.array([float(line.split()[3]) for line in verts[: 4 * m]])

    max_gap = float(np.max(np.abs(first_ring_z - expected)))
    assert max_gap < BOUNDARY_TOLERANCE_M, (
        f"shipped farfield.obj disagrees with shipped terrain.obj by {max_gap:.4f} m "
        "at the 200 m boundary"
    )


def test_manifest_unaffected_by_sky_earth_farfield(tmp_path):
    """generate_world's manifest.json must be byte-for-byte identical whether or not sky/earth/farfield are enabled - the costmap/planner read this file, and the brief requires the far-field geometry to be invisible to them."""
    from regolith_terrain_gen.generate import generate_world
    import dataclasses

    cfg_on = TerrainConfig(seed=7)
    cfg_off = dataclasses.replace(
        cfg_on, sky_enabled=False, earth_enabled=False, farfield_enabled=False
    )

    out_on = tmp_path / "on"
    out_off = tmp_path / "off"
    generate_world(cfg_on, out_on, start_paused=False)
    generate_world(cfg_off, out_off, start_paused=False)

    import json

    manifest_on = json.loads((out_on / "manifest.json").read_text())
    manifest_off = json.loads((out_off / "manifest.json").read_text())
    # The three fields below are simply the absolute output_dir, which legitimately
    # differs between the two tmp_path subdirs this test itself chose - nothing to
    # do with sky/earth/farfield. Normalise those, then everything else (every
    # crater, every rock, every spawn/heightmap number) must match exactly.
    for m, out in [(manifest_on, out_on), (manifest_off, out_off)]:
        for key in ("heightmap_png", "terrain_mesh_obj", "world_sdf"):
            m[key] = Path(m[key]).relative_to(out).as_posix()
    assert manifest_on == manifest_off


# Regression test for the "black band of stars between the near terrain and the
# far field" defect: from an elevated camera (the `orbit` render preset, roughly
# reproduced here), the far field's own silhouette - its elevation angle as seen
# from that camera, scanning outward shell by shell - must never give back more
# than a bounded amount in a SINGLE step. A gradual decline is normal and
# expected (real terrain recedes: distance grows faster than height can follow,
# so the angle drops smoothly all the way to the deliberate horizon_drop) - an
# ABRUPT multi-degree drop between adjacent shells is what actually reads as a
# gap of visible sky, because the eye reads a discontinuity, not a gradient.
# Camera position/height matches render_still.py's `orbit` preset, which is
# where this was first found. Checked at 8 evenly-spaced azimuths per seed.
# A first pass checked elevation ANGLE from the `orbit` camera instead of raw
# height, and false-positived on a genuinely smooth, continuous crater rim (a
# steep but real climb-and-descend produces a legitimately large angular swing
# shell to shell purely from geometry/perspective, with no height discontinuity
# at all - confirmed by printing that seed's actual height sequence: monotonic
# climb to a peak, then a smooth, gradual decline, no jump). Height is the
# right quantity to bound directly: it is what farfield.build_farfield_mesh's
# silhouette floor actually constrains, and it does not confuse "steep" with
# "discontinuous" the way a camera-angle projection can. The bug this catches
# produced single-shell height drops of 90-300 m (measured, pre-fix); ordinary
# relief (including steep rims, as above) stays well under 70 m per shell step
# at this mesh resolution.
MAX_SINGLE_STEP_HEIGHT_DROP_M = 70.0


@pytest.mark.parametrize("seed", [42, 7, 123])
def test_farfield_silhouette_has_no_sky_gap_from_elevated_camera(seed):
    """Regression test for the "black band of stars between the near terrain and the far field" defect (seen from an elevated camera, e.g. render_still.py's `orbit` preset - invisible from ground level, which is why it passed earlier ground-level renders).

    Checks the actual mechanism: within the "undropped" zone (see
    farfield.build_farfield_mesh's curve_start - past it,
    farfield_horizon_drop_m deliberately sinks the mesh fast, by design, and a
    large single-shell height drop there is the intended behaviour, not a
    defect), no shell may drop more than MAX_SINGLE_STEP_HEIGHT_DROP_M below
    its immediate predecessor at the same angular position. A sudden height
    cliff between two adjacent, closely-spaced shells is what actually reads as
    a gap of visible sky - a gradual multi-shell decline (ordinary perspective
    recession) or even a steep-but-continuous climb/descent (a crater rim) is
    not.
    """
    from regolith_terrain_gen.farfield import build_farfield_mesh
    import tempfile

    cfg = TerrainConfig(seed=seed)
    rng = np.random.default_rng(cfg.seed)
    _, _, _, elevation_lookup = build_heightmap(cfg, rng)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "farfield.obj"
        stats = build_farfield_mesh(cfg, elevation_lookup, out)
        text = out.read_text()

    verts = np.array(
        [[float(t) for t in line.split()[1:4]] for line in text.splitlines() if line.startswith("v ")]
    )
    n_shells, cols = stats["shells"], 4 * stats["segments_per_side"]
    z = verts[:, 2].reshape(n_shells, cols)

    frac = np.linspace(0.0, 1.0, n_shells)
    undropped = int(np.searchsorted(frac, 0.6, side="right"))

    drop = z[:undropped][:-1] - z[:undropped][1:]  # positive where height falls
    worst = float(drop.max())
    assert worst < MAX_SINGLE_STEP_HEIGHT_DROP_M, (
        f"seed {seed}: far-field height drops {worst:.1f} m in one shell step "
        f"(shell {int(drop.argmax()) // cols + 1}) - reads as a gap of visible "
        f"sky between the near terrain and the far field from an elevated camera"
    )
