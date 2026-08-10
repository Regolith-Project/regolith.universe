# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Top-level orchestration: seed -> heightmap + textures + rocks + world SDF + manifest."""

from pathlib import Path

import numpy as np

from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.heightmap import build_heightmap, build_terrain_collision_boxes_sdf, save_heightmap_png
from regolith_terrain_gen.rocks import generate_rock_variants
from regolith_terrain_gen.scatter import scatter_rocks
from regolith_terrain_gen.terrain_mesh import save_terrain_mesh_obj
from regolith_terrain_gen.textures import generate_textures
from regolith_terrain_gen.worldgen import build_world_sdf, write_manifest


def generate_world(cfg: TerrainConfig, output_dir: Path, start_paused: bool = True) -> Path:
    """Generates all world assets under output_dir and returns the path to world.sdf."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    raw_heightmap, heightmap, craters, elevation_lookup = build_heightmap(cfg, rng)
    heightmap_png = output_dir / "heightmap.png"
    # gz min/max-stretches the PNG to fill <size> z; save_heightmap_png hands back the
    # real-world (min, span) that full range maps to so worldgen can pin <pos>/<size> z
    # and the rendered ground lands exactly on the collision surface (see heightmap.py).
    heightmap_z_min, heightmap_z_span = save_heightmap_png(heightmap, heightmap_png)
    terrain_collision_sdf = build_terrain_collision_boxes_sdf(raw_heightmap, cfg)

    # The surface that is actually DRAWN. Written in world coordinates and with no
    # level of detail, so what the user sees at 200 m is what the tests measure - see
    # terrain_mesh.py for why a <heightmap> visual could not give that.
    terrain_mesh_obj = output_dir / "terrain.obj"
    terrain_mesh_stats = save_terrain_mesh_obj(heightmap, cfg, terrain_mesh_obj)

    texture_pngs = generate_textures(
        output_dir / "textures", cfg.texture_resolution_px, rng, cfg.texture_tile_size_m
    )

    rock_mesh_dir = output_dir / "rocks"
    rock_variants = generate_rock_variants(rock_mesh_dir, cfg.rock_variant_count, rng, cfg.rock_subdivisions)
    rocks = scatter_rocks(cfg, rng, rock_variants, elevation_lookup)

    world_sdf_path = output_dir / "world.sdf"
    world_sdf_path.write_text(
        build_world_sdf(
            cfg, texture_pngs, rocks, rock_mesh_dir, terrain_collision_sdf,
            terrain_mesh_obj,
            # Lets the opening GUI camera be placed relative to the real ground height
            # under it, instead of at a hardcoded absolute z that assumes one seed's
            # terrain elevation (see worldgen._gui_camera_pose).
            elevation_lookup=elevation_lookup,
            start_paused=start_paused,
        )
    )

    spawn_elevation_m = elevation_lookup(*cfg.spawn_zone_center)
    write_manifest(
        output_dir / "manifest.json", cfg, craters, rocks, heightmap_png, world_sdf_path,
        spawn_elevation_m,
        # Same (min, span) the world SDF decodes with, so the costmap reads the PNG back
        # at the elevations gz renders and the collision boxes use.
        heightmap_z_min_m=heightmap_z_min, heightmap_z_span_m=heightmap_z_span,
        terrain_mesh_obj=terrain_mesh_obj, terrain_mesh_stats=terrain_mesh_stats,
    )

    return world_sdf_path
