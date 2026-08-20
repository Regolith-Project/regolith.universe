# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Rock COLLISION geometry (see PROGRESS.md).

The rocks' <collision> geometry used to be <mesh>, which is silently a no-op in this
gz-physics install (verified: a probe dropped onto one falls straight through), so the
boulders were decoration the rover drove through while the costmap planned around them.
Swapping them for fitted ellipsoids is a correctness fix only - it costs and saves
essentially nothing.

**Whether rocks sit on the ground is NOT tested here any more.** This file used to carry
a test_no_rock_floats that compared each rock against ``elevation_lookup`` - the same
function scatter.seat_rock_z seats it with. Two things measured through one convention
agree however wrong that convention is, so it stayed green while rocks visibly floated,
twice. It also rebuilt the rock variants with an rng that had not consumed the same
draws as generate_world, so it graded a set of rocks the shipped world never contained.
Seating is now checked in test_rock_seating_against_rendered_png.py, against the PNG and
OBJ files that actually ship.
"""

import numpy as np
import pytest
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.generate import generate_world
from regolith_terrain_gen.rocks import generate_rock_variants

SEEDS = [42, 123, 7]


def test_rock_collision_is_not_a_mesh(tmp_path):
    """<mesh> collision does nothing in this gz-physics install, so a rock whose
    collision geometry is a mesh is an obstacle the rover drives straight through."""
    import json

    world_sdf = generate_world(TerrainConfig(seed=42), tmp_path / "world", start_paused=True)
    text = world_sdf.read_text()
    manifest = json.loads((world_sdf.parent / "manifest.json").read_text())
    rock_count = len(manifest["rocks"])
    assert rock_count > 0

    # Isolate the <collision> blocks and assert none of them contains a <mesh>.
    collision_blocks = [blk.split("</collision>")[0] for blk in text.split("<collision")[1:]]
    meshed = [b for b in collision_blocks if "<mesh>" in b]
    assert (
        not meshed
    ), f"{len(meshed)} collision element(s) still use non-functioning <mesh> geometry"

    # Exactly one working collision proxy per rock...
    assert text.count("<ellipsoid>") == rock_count

    # ...while the rock the user SEES is still the detailed mesh, one visual per rock.
    # The terrain's own visual is a mesh too now (terrain_mesh.py), so match on the
    # rocks/ mesh directory rather than counting every <mesh> visual in the world.
    visual_blocks = [blk.split("</visual>")[0] for blk in text.split("<visual")[1:]]
    rock_visual_meshes = [b for b in visual_blocks if "<mesh>" in b and "/rocks/" in b]
    assert len(rock_visual_meshes) == rock_count


@pytest.mark.parametrize("seed", SEEDS)
def test_collision_ellipsoid_stays_inside_the_visible_rock(tmp_path, seed):
    """The collision proxy must not poke out past the mesh the user sees, or the rover
    would stop against thin air."""
    cfg = TerrainConfig(seed=seed)
    rng = np.random.default_rng(cfg.seed)
    variants = generate_rock_variants(
        tmp_path / "rocks", cfg.rock_variant_count, rng, cfg.rock_subdivisions
    )
    for variant in variants:
        radii = variant.collision_radii
        assert (radii > 0).all()
        # Every ellipsoid semi-axis is realised by an actual vertex of the mesh, so the
        # proxy touches the silhouette without exceeding its bounding box.
        assert (radii <= np.max(np.abs(variant.vertices), axis=0) + 1e-9).all()
