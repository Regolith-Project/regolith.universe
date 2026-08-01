# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the "big floating rocks" report (see PROGRESS.md).

Two separate defects produced it, and both are locked down here:

1. Rocks were placed by sinking the mesh ORIGIN a fixed 0.12 * scale below the ground.
   Meshes are normalized by their bounding RADIUS, but the anisotropic stretch in
   displace_rock leaves each variant's lowest vertex 0.51-1.0 units below its origin, so
   every rock hovered (bottom - 0.12) * scale in the air - up to ~2.1 m under a 2.4 m
   boulder.

2. The rocks' <collision> geometry was <mesh>, which is silently a no-op in this
   gz-physics install (verified: a probe dropped onto one falls straight through), so
   the boulders were decoration the rover drove through. Swapping them for fitted
   ellipsoids is a correctness fix only - it costs and saves essentially nothing.
"""

from pathlib import Path

import numpy as np
import pytest

from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.generate import generate_world
from regolith_terrain_gen.heightmap import build_heightmap
from regolith_terrain_gen.rocks import generate_rock_variants
from regolith_terrain_gen.scatter import _rotation_matrix, scatter_rocks

SEEDS = [42, 123, 7]


def _place(cfg: TerrainConfig, mesh_dir: Path):
    rng = np.random.default_rng(cfg.seed)
    _, _, _, elevation_lookup = build_heightmap(cfg, rng)
    variants = generate_rock_variants(mesh_dir, cfg.rock_variant_count, rng, cfg.rock_subdivisions)
    rocks = scatter_rocks(cfg, rng, variants, elevation_lookup)
    return rocks, {v.name: v for v in variants}, elevation_lookup


def _lowest_clearances(rocks, by_name, elevation_lookup) -> np.ndarray:
    """For each rock, the gap between its lowest vertex and the ground under that vertex.

    Positive means the rock floats; negative means it is embedded.
    """
    out = []
    for rock in rocks:
        verts = (by_name[rock.variant].vertices * rock.scale_m) @ _rotation_matrix(
            rock.roll_rad, rock.pitch_rad, rock.yaw_rad
        ).T
        clearance = [
            rock.z_m + v[2] - elevation_lookup(rock.x_m + v[0], rock.y_m + v[1]) for v in verts
        ]
        out.append(min(clearance))
    return np.array(out)


@pytest.mark.parametrize("seed", SEEDS)
def test_no_rock_floats(tmp_path, seed):
    cfg = TerrainConfig(seed=seed)
    rocks, by_name, lookup = _place(cfg, tmp_path / "rocks")
    clearances = _lowest_clearances(rocks, by_name, lookup)

    assert len(rocks) > 0
    floating = clearances > 0.0
    assert not floating.any(), (
        f"{floating.sum()} of {len(rocks)} rocks float above the terrain "
        f"(worst gap {clearances.max():.3f} m)"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_rocks_are_embedded_but_not_swallowed(tmp_path, seed):
    """Seated rocks should sit slightly INTO the regolith - not hovering, and not sunk
    so far that a boulder reads as a pebble."""
    cfg = TerrainConfig(seed=seed)
    rocks, by_name, lookup = _place(cfg, tmp_path / "rocks")
    clearances = _lowest_clearances(rocks, by_name, lookup)
    scales = np.array([r.scale_m for r in rocks])

    embed_depth = -clearances  # positive = how deep the lowest point is buried
    assert (embed_depth > 0).all()
    # Nothing should be buried by more than a third of its own size.
    assert (embed_depth < 0.34 * scales * 2.0).all(), (
        f"deepest embed {embed_depth.max():.2f} m against scale "
        f"{scales[embed_depth.argmax()]:.2f} m"
    )


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
    assert not meshed, f"{len(meshed)} collision element(s) still use non-functioning <mesh> geometry"

    # Exactly one working collision proxy per rock...
    assert text.count("<ellipsoid>") == rock_count

    # ...while the rock the user SEES is still the detailed mesh, one visual per rock.
    visual_blocks = [blk.split("</visual>")[0] for blk in text.split("<visual")[1:]]
    rock_visual_meshes = [b for b in visual_blocks if "<mesh>" in b]
    assert len(rock_visual_meshes) == rock_count


@pytest.mark.parametrize("seed", SEEDS)
def test_collision_ellipsoid_stays_inside_the_visible_rock(tmp_path, seed):
    """The collision proxy must not poke out past the mesh the user sees, or the rover
    would stop against thin air."""
    cfg = TerrainConfig(seed=seed)
    rng = np.random.default_rng(cfg.seed)
    variants = generate_rock_variants(tmp_path / "rocks", cfg.rock_variant_count, rng, cfg.rock_subdivisions)
    for variant in variants:
        radii = variant.collision_radii
        assert (radii > 0).all()
        # Every ellipsoid semi-axis is realised by an actual vertex of the mesh, so the
        # proxy touches the silhouette without exceeding its bounding box.
        assert (radii <= np.max(np.abs(variant.vertices), axis=0) + 1e-9).all()
