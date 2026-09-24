# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Proves the terrain-visual-realism pass (terrain_detail.py, textures_world.py,
rocks_visual.py) never touches anything the rest of the project has measured navigation
results against - see config.py's "physics is frozen" note and PROGRESS.md.

This does NOT compare against a checked-in snapshot file (which could silently go stale,
or never get regenerated when it should). It compares against the FROZEN functions
themselves, called directly with no realism-pass code in between - exactly the pipeline
that existed before this pass - so what is actually being checked is "does generate_world
route frozen inputs through the frozen functions unchanged", which is the real question of
interest and cannot go stale.

Per this project's verification standard, a check that has never been seen to fail is not
evidence of anything: test_the_check_can_actually_fail deliberately breaks the pipeline
(routes the collision SDF through an extra smoothing pass, the way a careless refactor
might) and asserts this file's own comparison catches it.
"""

import json

import numpy as np
import pytest
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.generate import generate_world
from regolith_terrain_gen.heightmap import build_heightmap
from regolith_terrain_gen.heightmap import build_terrain_collision_boxes_sdf
from regolith_terrain_gen.heightmap import save_heightmap_png
from regolith_terrain_gen.rocks import generate_rock_variants
from regolith_terrain_gen.scatter import scatter_rocks
from regolith_terrain_gen.textures import generate_textures

SEEDS = [42, 7, 123]

ROCK_FROZEN_FIELDS = (
    "x_m",
    "y_m",
    "z_m",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "scale_m",
    "collision_radii_m",
)


def _reference_frozen_outputs(cfg: TerrainConfig, tmp_path):
    """Rebuild everything the hard constraint freezes, calling ONLY the pre-existing
    frozen functions directly - no terrain_detail/textures_world/rocks_visual involved."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    raw_heightmap, visual_heightmap, craters, elevation_lookup = build_heightmap(cfg, rng)
    ref_png = tmp_path / "ref_heightmap.png"
    z_min, z_span = save_heightmap_png(visual_heightmap, ref_png)
    collision_sdf = build_terrain_collision_boxes_sdf(raw_heightmap, cfg)

    # generate.py keeps this exact call (its own docstring explains why: it exists ONLY
    # to consume `rng` in the same position/amount the pre-realism-pass generator did),
    # so the reference pipeline here has to call it too, in the same position, or the two
    # rng streams diverge for a reason that has nothing to do with a real regression.
    generate_textures(
        tmp_path / "ref_textures_legacy", cfg.texture_resolution_px, rng, cfg.texture_tile_size_m
    )

    rock_variants = generate_rock_variants(
        tmp_path / "ref_rocks", cfg.rock_variant_count, rng, cfg.rock_subdivisions
    )
    rocks = scatter_rocks(cfg, rng, rock_variants, elevation_lookup)
    return {
        "heightmap_png_bytes": ref_png.read_bytes(),
        "z_min": z_min,
        "z_span": z_span,
        "collision_sdf": collision_sdf,
        "craters": craters,
        "rocks": rocks,
    }


def _assert_matches_reference(world_dir, manifest, ref) -> None:
    assert (world_dir / "heightmap.png").read_bytes() == ref["heightmap_png_bytes"], (
        "heightmap.png is no longer byte-identical to the frozen pipeline's own output - "
        "the costmap decodes this file, and every navigation result on record assumes it"
    )
    assert manifest["heightmap_z_min_m"] == ref["z_min"]
    assert manifest["heightmap_z_span_m"] == ref["z_span"]

    sdf_text = (world_dir / "world.sdf").read_text()
    assert ref["collision_sdf"] in sdf_text, (
        "the collision box SDF embedded in world.sdf no longer matches "
        "build_terrain_collision_boxes_sdf's own output verbatim"
    )

    assert len(manifest["craters"]) == len(ref["craters"])
    for got, want in zip(manifest["craters"], ref["craters"]):
        assert got["x_m"] == want.x_m
        assert got["y_m"] == want.y_m
        assert got["diameter_m"] == want.diameter_m
        assert got["depth_m"] == want.depth_m
        assert got["rim_height_m"] == want.rim_height_m

    assert len(manifest["rocks"]) == len(ref["rocks"])
    for got, want in zip(manifest["rocks"], ref["rocks"]):
        for field in ROCK_FROZEN_FIELDS:
            got_v = got[field]
            want_v = getattr(want, field)
            if field == "collision_radii_m":
                assert list(got_v) == list(want_v), field
            else:
                assert got_v == want_v, field


@pytest.mark.parametrize("seed", SEEDS)
def test_visual_pass_leaves_frozen_artifacts_byte_identical(tmp_path, seed):
    cfg = TerrainConfig(seed=seed)
    ref = _reference_frozen_outputs(cfg, tmp_path / "ref")

    world_sdf = generate_world(cfg, tmp_path / "world", start_paused=True)
    world_dir = world_sdf.parent
    manifest = json.loads((world_dir / "manifest.json").read_text())

    _assert_matches_reference(world_dir, manifest, ref)


def test_the_check_can_actually_fail(tmp_path, monkeypatch):
    """Guards the guard: if generate.py's collision path were accidentally rerouted
    through an extra transform (the kind of mistake a careless refactor could make),
    this file's own comparison has to catch it - proven by actually breaking it and
    watching the assertion above go red."""
    import regolith_terrain_gen.generate as generate_mod

    real_build = generate_mod.build_terrain_collision_boxes_sdf

    def _broken_build(raw_heightmap, cfg):
        # Simulates a refactor that accidentally smooths the array before building
        # collision boxes from it - collision geometry would silently drift away from
        # what elevation_lookup and every rock's frozen seating promise.
        return real_build(raw_heightmap * 1.0001, cfg)

    monkeypatch.setattr(generate_mod, "build_terrain_collision_boxes_sdf", _broken_build)

    cfg = TerrainConfig(seed=42)
    ref = _reference_frozen_outputs(cfg, tmp_path / "ref")
    world_sdf = generate_world(cfg, tmp_path / "world", start_paused=True)
    world_dir = world_sdf.parent
    manifest = json.loads((world_dir / "manifest.json").read_text())

    with pytest.raises(AssertionError):
        _assert_matches_reference(world_dir, manifest, ref)
