# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Top-level orchestration: seed -> heightmap + textures + rocks + world SDF + manifest."""

from pathlib import Path

import numpy as np
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.farfield import build_farfield_mesh
from regolith_terrain_gen.heightmap import build_heightmap
from regolith_terrain_gen.heightmap import build_terrain_collision_boxes_sdf
from regolith_terrain_gen.heightmap import save_heightmap_png
from regolith_terrain_gen.rocks import generate_rock_variants
from regolith_terrain_gen.rocks_visual import generate_visual_rock_variants
from regolith_terrain_gen.rocks_visual import rock_albedo_tints
from regolith_terrain_gen.rocks_visual import save_pebble_field_obj
from regolith_terrain_gen.scatter import scatter_rocks
from regolith_terrain_gen.sky import save_sky_dome_mesh
from regolith_terrain_gen.sky import save_sky_texture
from regolith_terrain_gen.terrain_detail import build_drawn_surface
from regolith_terrain_gen.terrain_mesh import mesh_surface_lookup
from regolith_terrain_gen.terrain_mesh import save_terrain_mesh_obj
from regolith_terrain_gen.textures import generate_textures
from regolith_terrain_gen.textures_world import generate_world_textures
from regolith_terrain_gen.worldgen import build_world_sdf
from regolith_terrain_gen.worldgen import write_manifest


def _child_rng(seed: int, label: str) -> np.random.Generator:
    """An RNG independent of the shared `rng` `generate_world` uses for
    heightmap/crater/rock placement, derived from (seed, label) so it is still
    deterministic per seed. Every VISUAL-ONLY generator below (terrain detail, rock
    shape variety, pebbles, world-scale textures) must use one of these, never the
    shared `rng` - see config.py's "physics is frozen" note: touching the shared
    generator's draw sequence, even indirectly, shifts every rock/crater placed after
    the touch point and breaks the frozen manifest fields for that seed."""
    import zlib

    return np.random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(label.encode())])


def generate_world(cfg: TerrainConfig, output_dir: Path, start_paused: bool = True) -> Path:
    """Generate all world assets under output_dir and return the path to world.sdf."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    raw_heightmap, heightmap, craters, elevation_lookup = build_heightmap(cfg, rng)
    heightmap_png = output_dir / "heightmap.png"
    # gz min/max-stretches the PNG to fill <size> z; save_heightmap_png hands back the
    # real-world (min, span) that full range maps to so worldgen can pin <pos>/<size> z
    # and the rendered ground lands exactly on the collision surface (see heightmap.py).
    heightmap_z_min, heightmap_z_span = save_heightmap_png(heightmap, heightmap_png)
    terrain_collision_sdf = build_terrain_collision_boxes_sdf(raw_heightmap, cfg)

    # The DRAWN surface, refined from `heightmap` (== elevation_lookup's own surface)
    # within a hard, measured budget - see terrain_detail.py's module docstring. Runs on
    # an independent rng so it cannot perturb the shared `rng` stream below, which is
    # what keeps every rock's frozen manifest fields byte-identical to the pre-existing
    # generator for the same seed (see config.py's "physics is frozen" note).
    rng_visual = _child_rng(cfg.seed, "visual_detail")
    drawn_surface, surface_delta = build_drawn_surface(raw_heightmap, heightmap, cfg, rng_visual)

    # The surface that is actually DRAWN. Written in world coordinates, at a finer
    # stride than elevation_lookup uses (terrain_mesh_stride - kept untouched above) and
    # with a 1:1 world-space UV set (see terrain_mesh.save_terrain_mesh_obj) instead of
    # the old 20 m tile, so the new world-scale texture bake below can align macro
    # features to real terrain instead of repeating. No level of detail, so what the
    # user sees at 200 m is what the tests measure - see terrain_mesh.py.
    terrain_mesh_obj = output_dir / "terrain.obj"
    terrain_mesh_stats = save_terrain_mesh_obj(
        drawn_surface,
        cfg,
        terrain_mesh_obj,
        stride=cfg.terrain_visual_mesh_stride,
        uv_world_size_m=cfg.world_size_m,
    )
    terrain_mesh_stats["surface_delta_max_m"] = float(np.abs(surface_delta).max())
    terrain_mesh_stats["surface_delta_mean_m"] = float(np.abs(surface_delta).mean())

    # UNCHANGED call, at its original position, consuming `rng` exactly as the
    # pre-existing generator did - kept ONLY to preserve the shared rng's draw sequence
    # so generate_rock_variants/scatter_rocks below still draw the same numbers they did
    # before this realism pass, for the same seed. Its own PNG output is immediately
    # superseded by generate_world_textures() further down (independent rng) and is
    # never referenced by the shipped world.sdf. See config.py's texture_world_px note.
    _legacy_texture_pngs = generate_textures(
        output_dir / "textures_legacy", cfg.texture_resolution_px, rng, cfg.texture_tile_size_m
    )

    rock_mesh_dir = output_dir / "rocks"
    rock_variants = generate_rock_variants(
        rock_mesh_dir, cfg.rock_variant_count, rng, cfg.rock_subdivisions
    )
    rocks = scatter_rocks(cfg, rng, rock_variants, elevation_lookup)

    # Everything below is VISUAL ONLY and runs on rngs independent of the shared one
    # above - rocks/craters are already fully placed and frozen by this point.
    rng_rock_visual = _child_rng(cfg.seed, "rock_visual")
    rock_visual_mesh_paths = generate_visual_rock_variants(
        output_dir / "rocks_visual", rock_variants, rng_rock_visual, cfg.rock_visual_shrink
    )
    rock_tints = rock_albedo_tints(len(rocks), rng_rock_visual, cfg.rock_albedo_variation)

    pebble_field_obj = output_dir / "pebbles.obj"
    rng_pebbles = _child_rng(cfg.seed, "pebbles")
    save_pebble_field_obj(pebble_field_obj, cfg, drawn_surface, rng_pebbles)

    rng_textures = _child_rng(cfg.seed, "textures_world")
    texture_pngs = generate_world_textures(
        output_dir / "textures", cfg, drawn_surface, craters, rocks, rng_textures
    )

    # Sky and far-field horizon: both VISUAL ONLY, and both built on their OWN
    # independent rng streams (never the `rng` above) so adding/removing/tuning
    # them can never shift the seed's terrain, craters or rock placement - see
    # sky.py / farfield.py. Neither is written to manifest.json.
    sky_texture_path = None
    sky_mesh_obj = None
    if cfg.sky_enabled:
        sky_texture_path = save_sky_texture(cfg, output_dir / "sky" / "starfield.png")
        sky_mesh_obj = save_sky_dome_mesh(cfg, output_dir / "sky" / "sky_dome.obj")

    farfield_mesh_obj = None
    if cfg.farfield_enabled:
        farfield_mesh_obj = output_dir / "farfield.obj"
        # Matched against drawn_surface (what terrain_mesh_obj actually draws at the
        # boundary), not the coarser elevation_lookup/collision surface: the visual
        # realism pass above (build_drawn_surface) can move the drawn surface up to
        # visual_surface_budget_m off elevation_lookup, and the far-field mesh has to
        # meet whatever is actually ON SCREEN at |x|,|y| == world_size_m/2 to look
        # seamless, not whatever the physics box underneath it happens to be.
        drawn_surface_lookup = mesh_surface_lookup(
            drawn_surface, cfg, stride=cfg.terrain_visual_mesh_stride
        )
        build_farfield_mesh(cfg, drawn_surface_lookup, farfield_mesh_obj)

    world_sdf_path = output_dir / "world.sdf"
    world_sdf_path.write_text(
        build_world_sdf(
            cfg,
            texture_pngs,
            rocks,
            rock_mesh_dir,
            terrain_collision_sdf,
            terrain_mesh_obj,
            # Lets the opening GUI camera be placed relative to the real ground height
            # under it, instead of at a hardcoded absolute z that assumes one seed's
            # terrain elevation (see worldgen._gui_camera_pose).
            elevation_lookup=elevation_lookup,
            start_paused=start_paused,
            sky_texture_path=sky_texture_path,
            sky_mesh_obj=sky_mesh_obj,
            farfield_mesh_obj=farfield_mesh_obj,
            rock_visual_mesh_paths=rock_visual_mesh_paths,
            rock_tints=rock_tints,
            pebble_field_obj=pebble_field_obj,
        )
    )

    spawn_elevation_m = elevation_lookup(*cfg.spawn_zone_center)
    write_manifest(
        output_dir / "manifest.json",
        cfg,
        craters,
        rocks,
        heightmap_png,
        world_sdf_path,
        spawn_elevation_m,
        # Same (min, span) the world SDF decodes with, so the costmap reads the PNG back
        # at the elevations gz renders and the collision boxes use.
        heightmap_z_min_m=heightmap_z_min,
        heightmap_z_span_m=heightmap_z_span,
        terrain_mesh_obj=terrain_mesh_obj,
        terrain_mesh_stats=terrain_mesh_stats,
    )

    return world_sdf_path
