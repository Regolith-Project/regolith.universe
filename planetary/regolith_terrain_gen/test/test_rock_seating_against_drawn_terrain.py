# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Do the rocks float? Asked of the SHIPPED ARTEFACTS, not of the generator's helpers.

The "floating rocks" report has now come back four times, and each time the test that
was supposed to cover it passed. The mistakes have a family resemblance: the check kept
sampling a terrain that was not the one being drawn.

  1. It sampled ``elevation_lookup`` - the very function ``scatter.seat_rock_z`` seats
     rocks with. Two things measured through one convention agree no matter how wrong
     that convention is, so it stayed green through the PNG being written transposed.
  2. Repointed at ``heightmap.png``, it went green again - and this time correctly, as
     far as it went: the PNG really did describe a surface every rock sat on. What it
     could not see is that gz did not DRAW that surface at range. A ``<heightmap>``
     visual goes through Ogre-Next's Terra, which point-samples the terrain coarser the
     further it is from the camera. The data was right and the picture was wrong.

The ground is now a mesh (see ``terrain_mesh.py``), which has no level of detail - so
for the first time there is one artefact that is unambiguously both the data and the
picture, and this file reads exactly that:

  * ``terrain.obj`` -> the triangles gz draws the ground from, in world coordinates,
    sampled the way a triangle is and not as a bilinear patch.
  * ``rocks/<variant>.obj`` -> the actual triangles gz renders for each boulder.
  * ``manifest.json`` -> where each rock instance was placed.

Nothing here touches ``elevation_lookup``, ``build_heightmap`` or any in-memory array.

A geometry test still cannot see a rendering artefact, which is the trap this file fell
into last time. That gap is now covered separately and directly, by screenshotting the
real GUI and looking for sky underneath a boulder - see
``test_rendered_terrain_seats_rocks.py``.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.generate import generate_world

SEEDS = [42, 7, 123]

# A rock may not hang in the air at all. The small positive allowance is float noise in
# the OBJ's 4-decimal coordinates, not tolerance for a visible gap.
MAX_FLOAT_M = 0.002


def _drawn_surface(world_dir: Path):
    """The ground gz actually draws, rebuilt from the terrain mesh on disk.

    Parses ``terrain.obj`` without assuming how it was written: vertices are read as
    world coordinates and the grid is recovered from them, so a wrong axis, a wrong
    scale or a dropped row surfaces here instead of being reproduced.
    """
    verts = np.array(
        [
            [float(t) for t in line.split()[1:4]]
            for line in (world_dir / "terrain.obj").read_text().splitlines()
            if line.startswith("v ")
        ]
    )
    assert len(verts), "terrain.obj has no vertices"

    xs = np.unique(verts[:, 0])
    ys = np.unique(verts[:, 1])
    assert len(xs) * len(ys) == len(verts), "terrain.obj is not a regular grid of posts"
    grid = np.full((len(ys), len(xs)), np.nan)
    grid[np.searchsorted(ys, verts[:, 1]), np.searchsorted(xs, verts[:, 0])] = verts[:, 2]
    assert not np.isnan(grid).any(), "terrain.obj leaves holes in its grid"

    def sample(x_m, y_m):
        x = np.clip(np.asarray(x_m, dtype=float), xs[0], xs[-1])
        y = np.clip(np.asarray(y_m, dtype=float), ys[0], ys[-1])
        cx = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
        cy = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, len(ys) - 2)
        u = (x - xs[cx]) / (xs[cx + 1] - xs[cx])
        v = (y - ys[cy]) / (ys[cy + 1] - ys[cy])
        za, zb = grid[cy, cx], grid[cy, cx + 1]
        zc, zd = grid[cy + 1, cx + 1], grid[cy + 1, cx]
        # Each quad is two triangles split on the a->c diagonal, i.e. u == v. A rock
        # rests on a triangle, not on a bilinear patch, and the difference between the
        # two is exactly the sub-post detail that has hidden this bug before.
        return np.where(
            v <= u,
            za + (zb - za) * u + (zc - zb) * v,
            za + (zd - za) * v + (zc - zd) * u,
        )

    return sample


def _obj_vertices(path: Path) -> np.ndarray:
    """The mesh gz renders, parsed straight out of the exported OBJ."""
    verts = [
        [float(v) for v in line.split()[1:4]]
        for line in path.read_text().splitlines()
        if line.startswith("v ")
    ]
    assert verts, f"no vertices in {path}"
    return np.array(verts)


def _rotation(roll, pitch, yaw):
    """SDF pose convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def _clearances(world_dir: Path) -> tuple:
    """Lowest gap between each rock's mesh and the drawn ground under it.

    Positive = the rock hangs in the air. Negative = it is bedded into the regolith.
    """
    manifest = json.loads((world_dir / "manifest.json").read_text())
    sample = _drawn_surface(world_dir)
    meshes = {}

    gaps, scales = [], []
    for rock in manifest["rocks"]:
        name = rock["variant"]
        if name not in meshes:
            meshes[name] = _obj_vertices(world_dir / "rocks" / f"{name}.obj")
        verts = (meshes[name] * rock["scale_m"]) @ _rotation(
            rock["roll_rad"], rock["pitch_rad"], rock["yaw_rad"]
        ).T
        ground = sample(rock["x_m"] + verts[:, 0], rock["y_m"] + verts[:, 1])
        gaps.append(float(np.min(rock["z_m"] + verts[:, 2] - ground)))
        scales.append(rock["scale_m"])
    return np.array(gaps), np.array(scales)


@pytest.mark.parametrize("seed", SEEDS)
def test_no_rock_hangs_above_the_drawn_ground(tmp_path, seed):
    generate_world(TerrainConfig(seed=seed), tmp_path / "world", start_paused=True)
    gaps, _ = _clearances(tmp_path / "world")

    assert len(gaps) > 0
    floating = gaps > MAX_FLOAT_M
    assert not floating.any(), (
        f"seed {seed}: {floating.sum()} of {len(gaps)} rocks hang above the surface gz "
        f"draws (worst {gaps.max():.3f} m). This is the check that measures the shipped "
        f"terrain.obj and rock OBJs rather than the generator's own elevation_lookup - "
        f"if the placement tests still pass, the two surfaces have diverged."
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_rocks_are_bedded_in_but_not_swallowed(tmp_path, seed):
    """Seated, not hovering and not sunk until a boulder reads as a pebble."""
    generate_world(TerrainConfig(seed=seed), tmp_path / "world", start_paused=True)
    gaps, scales = _clearances(tmp_path / "world")

    embed = -gaps
    assert (embed > -MAX_FLOAT_M).all()
    assert (
        embed < 0.68 * scales
    ).all(), (
        f"seed {seed}: deepest embed {embed.max():.2f} m on a {scales[embed.argmax()]:.2f} m rock"
    )


def test_the_check_can_actually_fail(tmp_path):
    """Guards the guard: lift the rocks and this must go red.

    The previous generation of this test could not fail for the bug it was written for,
    so the ability to fail is asserted explicitly rather than assumed.
    """
    generate_world(TerrainConfig(seed=42), tmp_path / "world", start_paused=True)
    manifest_path = tmp_path / "world" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for rock in manifest["rocks"]:
        rock["z_m"] += 0.5
    manifest_path.write_text(json.dumps(manifest))

    gaps, _ = _clearances(tmp_path / "world")
    assert (gaps > MAX_FLOAT_M).all(), "raising every rock by 0.5 m must register as floating"
