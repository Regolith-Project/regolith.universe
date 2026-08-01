# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Do the rocks float? Asked of the SHIPPED ARTEFACTS, not of the generator's helpers.

The "floating rocks" report has now come back three times, and each time the test that
was supposed to cover it passed. The reason is always the same shape of mistake: the
check sampled the terrain through ``elevation_lookup``, which is the very function
``scatter.seat_rock_z`` seats the rocks with. Two things measured through one convention
agree with each other no matter how wrong that convention is - it stayed green through
the PNG being written transposed, and through the seating sampling a different surface
from the one gz interpolates.

So nothing here touches ``elevation_lookup``, ``build_heightmap`` or the in-memory
surface. Everything is read back off what ``generate_world`` actually wrote to disk, and
decoded the way gz decodes it:

  * ``heightmap.png`` + the ``<pos>``/``<size>`` in ``world.sdf`` -> the drawn surface,
    transposed back out of gz's axis order and stretched full-range, then sampled
    BILINEARLY between posts, which is how the renderer fills the space between them.
  * ``rocks/<variant>.obj`` -> the actual triangles gz renders for each boulder.
  * ``manifest.json`` -> where each rock instance was placed.

If any of the encoding, the axis convention, the z mapping, the mesh export or the
seating maths drifts, a rock leaves the ground here and this fails.

Verified against the real renderer while this was written: a top-down camera frame from
30 m shows every boulder's shadow attached to its silhouette, which at the world's 12 deg
sun elevation would separate by 4.7x any gap under the rock.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.generate import generate_world

SEEDS = [42, 7, 123]

# A rock may not hang in the air at all. The small positive allowance is 16-bit PNG
# quantisation over the height span (8 m / 65535 ~ 0.12 mm) plus float noise, not
# tolerance for a visible gap.
MAX_FLOAT_M = 0.002


def _drawn_surface(world_dir: Path):
    """The terrain gz actually draws, decoded from the files exactly as gz decodes them."""
    sdf = (world_dir / "world.sdf").read_text()
    visual = sdf[sdf.index('<visual name="terrain_visual">'):]
    size = [float(v) for v in re.search(r"<size>([-\d.eE ]+)</size>", visual).group(1).split()]
    pos = [float(v) for v in re.search(r"<pos>([-\d.eE ]+)</pos>", visual).group(1).split()]
    world_m, z_span, z_min = size[0], size[2], pos[2]

    from PIL import Image

    # .T undoes gz's axis order (first image axis is world X) so this is [row=y, col=x].
    pixels = np.array(Image.open(world_dir / "heightmap.png")).astype(np.float64).T
    surface = pixels / 65535.0 * z_span + z_min

    n = surface.shape[0]
    half = world_m / 2.0
    per_m = (n - 1) / world_m

    def sample(x_m, y_m):
        fx = np.clip((np.asarray(x_m) + half) * per_m, 0, n - 1)
        fy = np.clip((np.asarray(y_m) + half) * per_m, 0, n - 1)
        x0 = np.floor(fx).astype(int)
        y0 = np.floor(fy).astype(int)
        x1 = np.minimum(x0 + 1, n - 1)
        y1 = np.minimum(y0 + 1, n - 1)
        tx, ty = fx - x0, fy - y0
        return (
            surface[y0, x0] * (1 - tx) * (1 - ty) + surface[y0, x1] * tx * (1 - ty)
            + surface[y1, x0] * (1 - tx) * ty + surface[y1, x1] * tx * ty
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
        f"PNG and OBJ rather than the generator's own elevation_lookup - if the placement "
        f"tests still pass, the two surfaces have diverged."
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_rocks_are_bedded_in_but_not_swallowed(tmp_path, seed):
    """Seated, not hovering and not sunk until a boulder reads as a pebble."""
    generate_world(TerrainConfig(seed=seed), tmp_path / "world", start_paused=True)
    gaps, scales = _clearances(tmp_path / "world")

    embed = -gaps
    assert (embed > -MAX_FLOAT_M).all()
    assert (embed < 0.68 * scales).all(), (
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
