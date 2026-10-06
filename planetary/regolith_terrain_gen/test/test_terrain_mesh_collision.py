# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""terrain_collision="mesh": the ground the rover drives on must be the surface its
a-priori map describes.

Asserted on the ARTIFACTS that cross the boundary - the OBJ gz-physics loads and the
PNG the costmap decodes - per point and at asymmetric positions, because this
project has shipped a transposed heightmap past array-only tests before.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.generate import generate_world


@pytest.fixture(scope="module")
def mesh_world(tmp_path_factory):
    out = tmp_path_factory.mktemp("mesh_world")
    cfg = TerrainConfig(seed=31, terrain_collision="mesh", sky_enabled=False,
                        farfield_enabled=False, earth_enabled=False)
    sdf = generate_world(cfg, out, start_paused=False)
    return out, Path(sdf).read_text(), json.loads((out / "manifest.json").read_text())


def _obj_vertices(path):
    v = [line.split()[1:4] for line in path.read_text().splitlines() if line.startswith("v ")]
    return np.array(v, dtype=float)


def test_ground_collides_as_the_mesh_and_not_as_boxes(mesh_world):
    out, sdf, manifest = mesh_world
    assert "terrain_mesh_collision" in sdf
    assert f"file://{out / 'terrain_collision.obj'}" in sdf
    assert "terrain_box_" not in sdf
    assert manifest["terrain_collision"] == "mesh"


def test_collision_mesh_matches_the_map_png_point_by_point(mesh_world):
    from regolith_costmap.costmap_node import load_heightmap

    out, _, manifest = mesh_world
    hm = load_heightmap(manifest)  # what the costmap and the terrain matcher read
    n = hm.shape[0]
    half = manifest["world_size_m"] / 2.0
    res = manifest["world_size_m"] / (n - 1)
    verts = _obj_vertices(out / "terrain_collision.obj")
    rng = np.random.default_rng(0)
    for i in rng.choice(len(verts), 400, replace=False):
        x, y, z = verts[i]
        row, col = int(round((y + half) / res)), int(round((x + half) / res))
        # PNG quantisation is 16-bit over the height span: a few millimetres at most.
        assert abs(hm[row, col] - z) < 0.01, (x, y, z, hm[row, col])


def test_boxes_remain_the_default():
    assert TerrainConfig(seed=1).terrain_collision == "boxes"
