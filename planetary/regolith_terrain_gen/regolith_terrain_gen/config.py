# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Tunable parameters for procedural lunar terrain generation."""

from dataclasses import dataclass, field


@dataclass
class TerrainConfig:
    seed: int = 42

    # World / heightmap
    world_size_m: float = 200.0
    heightmap_resolution_px: int = 513  # 2^n + 1, standard heightmap convention
    height_range_m: float = 10.0

    # Base roughness (fractal Brownian motion)
    fbm_octaves: int = 5
    fbm_base_cell_px: float = 96.0
    fbm_lacunarity: float = 2.0
    fbm_gain: float = 0.5
    fbm_weight_m: float = 0.9  # contribution of base roughness before final normalization

    # Regional slope
    regional_slope_deg: float = 1.5

    # Crater field (power-law size-frequency distribution)
    # crater_count and rock_count were raised from their original values (60 / 130)
    # and spawn_zone_radius_m lowered from 12.0: measured against the actual costmap
    # lethality (not just raw obstacle footprints), the original density let a
    # majority of short (10-20 m) straight lines pass completely clear of any
    # obstacle - e.g. the shipped tour_mission.py's fixed 5-leg route crossed an
    # obstacle on only 1 of 5 legs, so the rover barely had to turn. These values
    # were chosen by measuring straight-line-blocked fraction and A* reachability
    # together across several seeds (60-100 m and 10-20 m goals) rather than eyeballed
    # - see PROGRESS.md's "Terrain density increase" note for the actual numbers
    # and the trade-off against a small increase in unreachable-goal risk.
    #
    # Crater sizes start at 6 m, not 2 m, because the surface physically cannot hold
    # anything smaller. Craters are sculpted into the fine heightmap, but the surface
    # that is both RENDERED and COLLIDED is the block-averaged, blurred collision grid
    # (see heightmap.build_heightmap) - so a crater below roughly 2x the collision cell
    # size is averaged clean away. Measured on the previous 8.3 m cells: of 100 craters
    # placed, a mean of 2 survived into the rendered surface at all, and craters under
    # 10 m retained ~0% of their depth (some centres came out very slightly RAISED).
    # That is why the world read as uncratered no matter how high crater_count went.
    # Sub-6 m pitting is now carried by the surface texture instead (textures.py),
    # which costs no physics resolution. See PROGRESS.md "Terrain realism pass".
    crater_count: int = 160
    crater_diameter_min_m: float = 6.0
    crater_diameter_max_m: float = 50.0
    crater_size_exponent: float = 2.0  # N(>D) ~ D^-exponent
    crater_depth_to_diameter: float = 0.055
    crater_rim_height_frac: float = 0.35  # rim height as a fraction of crater depth
    crater_rim_width_frac: float = 0.18  # rim gaussian width as a fraction of crater radius

    # Spawn zone (guaranteed traversable, kept clear of craters/rocks)
    spawn_zone_radius_m: float = 9.0
    spawn_zone_center: tuple = (0.0, 0.0)

    # Collision-box approximation of the terrain (see heightmap.py's
    # build_terrain_collision_boxes_sdf) - also drives the synthesized visual
    # heightmap (build_heightmap), so the rendered ground and the physics ground
    # are the same surface. Kept on cfg rather than as separate keyword defaults
    # on each function so the collision-box builder and the visual synthesizer
    # can never drift out of sync with each other.
    # 40 cells/axis (5.0 m cells, 1764 boxes) rather than the previous 24 (8.3 m).
    # Box count is the dominant physics cost, and this is NOT free - it is bought
    # deliberately. Measured seed 42, interleaved in one session, 3 reps of 3000 steps
    # (absolute RTF on this box drifts between sessions, so only same-session
    # comparisons mean anything):
    #     res24 = 0.479   res32 = 0.388   res40 = 0.269   res48 = 0.206
    # So res40 runs ~1.8x slower than what shipped before. What it buys, across seeds
    # 42/7/123: craters actually present in the rendered surface go 2 -> 32, and slope
    # p95 (the surface really driven) 3.8 deg -> 10.4 deg. res48 would buy 41 craters
    # but costs 2.3x; res40 was chosen as the better point on that curve.
    # Note the rock ellipsoid-collision fix in rocks.py does NOT pay for this - it is a
    # correctness fix, not a performance one. All 190 rocks together cost only ~12%
    # (res24: 0.568 with no rocks, 0.499 with mesh rocks, 0.488 with ellipsoids), and
    # mesh vs ellipsoid is within measurement noise.
    # Finer cells are also what let craters exist at all (see crater_count), and the
    # inter-slab "lip" that drove the original coarse grid scales with cell size, so
    # finer cells partly offset the extra relief: boundaries stepping higher than the
    # 0.09 m wheel radius are 4.6% at res24 with these crater sizes, 1.4% at res40,
    # 0.5% at res48 - against 0.7% for the previously shipped res24 + small craters.
    # 1.4% is a regression on that proxy, so res40 was validated by a real M4 60-100 m
    # acceptance run rather than on the proxy alone - see PROGRESS.md.
    # Smoothing stays at 3 passes: dropping to 2 buys more crater relief (41 visible)
    # but pushes the lip metric to 5.9%, well past what the flip fix established as safe.
    collision_grid_resolution: int = 40
    collision_overlap_frac: float = 0.12
    collision_smoothing_passes: int = 3

    # Rocks
    rock_count: int = 190
    rock_variant_count: int = 4
    rock_scale_min_m: float = 0.3
    rock_scale_max_m: float = 2.4
    rock_subdivisions: int = 1
    # How far a seated rock is sunk BELOW its resting contact with the ground, as a
    # fraction of its scale. This is not the old "0.12 * scale below the mesh origin"
    # constant, which assumed a fixed mesh underside and left every rock floating -
    # see scatter.seat_rock_z for the measurement and the fix.
    rock_embed_frac: float = 0.10
    # Max random roll/pitch on top of yaw, so boulders don't all sit on the same axis.
    rock_tilt_max_rad: float = 0.35

    # Lighting
    sun_elevation_deg: float = 12.0
    sun_azimuth_deg: float = 235.0

    # Surface texture
    texture_resolution_px: int = 512
    # Metres of terrain one texture tile covers. Shared by worldgen (the <texture><size>
    # it writes into the SDF) and textures.py (which needs it to size the sub-resolution
    # crater pits in real metres) - one value so the two cannot drift apart.
    texture_tile_size_m: float = 20.0

    # Visual terrain mesh (terrain_mesh.py): keep every Nth heightmap post as a mesh
    # vertex. The ground is drawn as a <mesh>, not a <heightmap>, because a <heightmap>
    # goes through Ogre-Next's Terra and gets point-sampled coarser with distance -
    # which is what left rocks hanging in the sky at the horizon while every
    # placement test passed. See terrain_mesh.py for the measurement.
    #
    # Stride 1 is the full 513 posts (0.39 m). It is not needed: the drawn surface is
    # piecewise-planar over collision_grid_resolution cells (12 posts each here), so
    # stride 4 still lands a vertex on every cell boundary and reproduces the surface
    # to within a centimetre, for a 16x smaller mesh. Measured, gap opened under the
    # 190 rocks by meshing at each stride, seeds 42/7/123: stride 4 worst +0.01 m,
    # stride 8 worst -0.01 m (still bedded), stride 16 +0.08 m and rocks start
    # visibly lifting. 4 keeps a 4x margin on that and costs 33k triangles.
    terrain_mesh_stride: int = 4

    def __post_init__(self) -> None:
        if self.heightmap_resolution_px % 2 == 0:
            raise ValueError("heightmap_resolution_px should be odd (2^n + 1)")
