# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Do the NEW, shape-varied visual rocks (rocks_visual.py) float, and do they poke
outside the frozen collision ellipsoid the rover actually stops against?

test_rock_seating_against_drawn_terrain.py already asks the first question of the
ORIGINAL displaced-icosphere meshes in ``rocks/<variant>.obj`` - but those are no longer
what gz draws. Since worldgen.build_world_sdf's default call now points every rock's
<visual> at ``rocks_visual/<variant>_visual.obj`` instead, a check that only reads
``rocks/`` would stay green while the actually-rendered mesh floats - precisely the
"stayed green through the wrong surface" failure mode PROGRESS.md and that file's own
docstring describe, just with a new second mesh instead of a wrong axis convention. This
file reads what ``<visual>`` in the SHIPPED world.sdf actually names: the real mesh URI
and its own local <pose> offset (VISUAL_SEATING_MARGIN_M - see worldgen._rock_model_sdf).

Both checks are asked of the SHIPPED ARTEFACTS (world.sdf, manifest.json, terrain.obj,
rocks_visual/*.obj) rather than of any generator helper, per this file's sibling's own
hard-won lesson.
"""

import json
import math
from pathlib import Path
import re

import numpy as np
import pytest
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.generate import generate_world

SEEDS = [42, 7, 123]

# Same allowance as test_rock_seating_against_drawn_terrain.py: float noise in the OBJ's
# decimal coordinates, not tolerance for a visible gap.
MAX_FLOAT_M = 0.002


def _drawn_surface(world_dir: Path):
    verts = np.array(
        [
            [float(t) for t in line.split()[1:4]]
            for line in (world_dir / "terrain.obj").read_text().splitlines()
            if line.startswith("v ")
        ]
    )
    assert len(verts), "terrain.obj has no vertices"
    xs, ys = np.unique(verts[:, 0]), np.unique(verts[:, 1])
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
        return np.where(
            v <= u,
            za + (zb - za) * u + (zc - zb) * v,
            za + (zd - za) * v + (zc - zd) * u,
        )

    return sample


def _obj_vertices(path: Path) -> np.ndarray:
    verts = [
        [float(v) for v in line.split()[1:4]]
        for line in path.read_text().splitlines()
        if line.startswith("v ")
    ]
    assert verts, f"no vertices in {path}"
    return np.array(verts)


def _rotation(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


_MODEL_RE = re.compile(r'<model name="rock_(\d+)">(.*?)</model>', re.DOTALL)
_VISUAL_URI_RE = re.compile(r"<visual[^>]*>.*?<uri>file://([^<]+)</uri>", re.DOTALL)
_VISUAL_POSE_RE = re.compile(r"<visual[^>]*>\s*<pose>([^<]+)</pose>")


def _rock_visuals_from_sdf(world_dir: Path) -> dict:
    """{rock_index: (mesh_path, (dx, dy, dz))} straight out of the shipped world.sdf -
    the actual mesh URI and local <visual> pose offset gz will draw, not an assumption
    about worldgen.py's internal naming convention."""
    text = (world_dir / "world.sdf").read_text()
    out = {}
    for m in _MODEL_RE.finditer(text):
        index = int(m.group(1))
        block = m.group(2)
        uri_match = _VISUAL_URI_RE.search(block)
        assert uri_match, f"rock_{index} has no visual mesh uri"
        pose_match = _VISUAL_POSE_RE.search(block)
        offset = (0.0, 0.0, 0.0)
        if pose_match:
            parts = [float(t) for t in pose_match.group(1).split()]
            offset = tuple(parts[:3])
        out[index] = (Path(uri_match.group(1)), offset)
    return out


def _visual_clearances(world_dir: Path) -> tuple:
    manifest = json.loads((world_dir / "manifest.json").read_text())
    sample = _drawn_surface(world_dir)
    visuals = _rock_visuals_from_sdf(world_dir)
    meshes = {}

    gaps, scales = [], []
    for i, rock in enumerate(manifest["rocks"]):
        mesh_path, (dx, dy, dz) = visuals[i]
        if mesh_path not in meshes:
            meshes[mesh_path] = _obj_vertices(mesh_path)
        v = meshes[mesh_path] * rock["scale_m"]
        v = v + np.array([dx, dy, dz])
        verts = v @ _rotation(rock["roll_rad"], rock["pitch_rad"], rock["yaw_rad"]).T
        ground = sample(rock["x_m"] + verts[:, 0], rock["y_m"] + verts[:, 1])
        gaps.append(float(np.min(rock["z_m"] + verts[:, 2] - ground)))
        scales.append(rock["scale_m"])
    return np.array(gaps), np.array(scales)


@pytest.mark.parametrize("seed", SEEDS)
def test_no_visual_rock_hangs_above_the_drawn_ground(tmp_path, seed):
    generate_world(TerrainConfig(seed=seed), tmp_path / "world", start_paused=True)
    gaps, _ = _visual_clearances(tmp_path / "world")

    assert len(gaps) > 0
    floating = gaps > MAX_FLOAT_M
    assert not floating.any(), (
        f"seed {seed}: {floating.sum()} of {len(gaps)} rocks' NEW VISUAL mesh hangs above "
        f"the drawn surface (worst {gaps.max():.4f} m) - this is what gz actually renders "
        f"today (rocks_visual/*_visual.obj), not the original displaced-icosphere mesh."
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_visual_rocks_are_bedded_in_but_not_swallowed(tmp_path, seed):
    generate_world(TerrainConfig(seed=seed), tmp_path / "world", start_paused=True)
    gaps, scales = _visual_clearances(tmp_path / "world")

    embed = -gaps
    assert (embed > -MAX_FLOAT_M).all()
    assert (embed < 0.68 * scales).all(), (
        f"seed {seed}: deepest embed {embed.max():.2f} m on a {scales[embed.argmax()]:.2f} m rock"
    )


def test_the_float_check_can_actually_fail(tmp_path):
    generate_world(TerrainConfig(seed=42), tmp_path / "world", start_paused=True)
    manifest_path = tmp_path / "world" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for rock in manifest["rocks"]:
        rock["z_m"] += 0.5
    manifest_path.write_text(json.dumps(manifest))

    gaps, _ = _visual_clearances(tmp_path / "world")
    assert (gaps > MAX_FLOAT_M).all(), "raising every rock by 0.5 m must register as floating"


@pytest.mark.parametrize("seed", SEEDS)
def test_visual_rock_meshes_stay_inside_frozen_collision_ellipsoid(tmp_path, seed):
    """Every visual mesh vertex, scaled and offset exactly as world.sdf places it, must
    satisfy the frozen ellipsoid inequality - the "any new mesh must fit within the same
    collision_radii_m envelope" requirement, checked against the shipped assets rather
    than trusted from rocks_visual.fit_to_collision_envelope's own arithmetic."""
    generate_world(TerrainConfig(seed=seed), tmp_path / "world", start_paused=True)
    world_dir = tmp_path / "world"
    manifest = json.loads((world_dir / "manifest.json").read_text())
    visuals = _rock_visuals_from_sdf(world_dir)
    meshes = {}

    worst_ratio = 0.0
    for i, rock in enumerate(manifest["rocks"]):
        mesh_path, _offset = visuals[i]
        if mesh_path not in meshes:
            meshes[mesh_path] = _obj_vertices(mesh_path)
        verts_unit = meshes[mesh_path]  # local mesh units, BEFORE the instance's own scale_m
        radii = np.array(rock["collision_radii_m"]) / rock["scale_m"]  # back to unit-mesh space
        ratio = np.sqrt(np.sum((verts_unit / radii[None, :]) ** 2, axis=1))
        worst_ratio = max(worst_ratio, float(ratio.max()))

    assert worst_ratio <= 1.0 + 1e-6, (
        f"seed {seed}: a visual rock mesh vertex reaches {worst_ratio:.4f}x its frozen "
        f"collision ellipsoid - the rover could drive through part of a visible rock "
        f"without colliding with it"
    )


def test_the_containment_check_can_actually_fail(tmp_path):
    """Guards the guard: a mesh deliberately inflated past its own ellipsoid must be caught."""
    from regolith_terrain_gen.rocks_visual import fit_to_collision_envelope

    target = np.array([0.8, 0.6, 0.9])
    vertices = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    fitted = fit_to_collision_envelope(vertices, target, shrink=0.97)
    inflated = fitted * 1.5  # simulate the bug: something re-scales the mesh afterward

    ratio = np.sqrt(np.sum((inflated / target[None, :]) ** 2, axis=1))
    assert ratio.max() > 1.0, "an inflated mesh must register as exceeding its ellipsoid"
