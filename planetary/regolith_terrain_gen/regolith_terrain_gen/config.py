# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Tunable parameters for procedural lunar terrain generation."""

from dataclasses import dataclass


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
    # Scene ambient. Near-black by default and deliberately so: with no atmosphere to
    # scatter light, a real lunar shadow is lit only by starlight and earthshine. It is
    # raised (via hello_moon.launch.py's cine_light) only when recording footage, where
    # that honest black swallows the surface texture. Render-only either way - nothing
    # in the heightmap, rock or costmap path reads it.
    scene_ambient: tuple = (0.06, 0.06, 0.07)

    # Sky, Earth, and far-field horizon - all VISUAL ONLY. None of these fields
    # are read by heightmap.py/craters.py/scatter.py, none of the geometry they
    # drive carries collision, and none of it is written to manifest.json - see
    # sky.py / earth.py / farfield.py for the mechanism. Safe to leave on by
    # default: they cost render time, not physics time (see PROGRESS.md's
    # "world beyond the world" entry for the measured RTF delta).
    sky_enabled: bool = True
    # Comfortably inside farfield_outer_radius_m's furthest corner (radius_m *
    # sqrt(2)) and under render_still.py's camera far clip - see that script for
    # the far-clip value this was matched against.
    sky_radius_m: float = 4800.0
    sky_star_count: int = 7000
    sky_texture_resolution: tuple = (4096, 2048)
    sky_milky_way_intensity: float = 0.14

    earth_enabled: bool = True
    # ~1.9 deg is the real Earth-from-Moon angular size; 2.0 deg matches the
    # brief ("roughly four times the Moon seen from Earth", ~0.5 deg).
    earth_angular_diameter_deg: float = 2.0
    earth_distance_m: float = 4000.0
    # Azimuth/elevation in the same convention as sun_azimuth_deg/sun_elevation_deg
    # (0 deg azimuth = +x, elevation up from local horizontal).
    #
    # NOT near the horizon/hero presets' own look direction (~13-80 deg azimuth,
    # see render_still.py's _presets) - and that is a real physical constraint,
    # not an oversight. Earth's phase comes from real per-pixel lighting off the
    # SAME sun direction the terrain uses (see earth.py), and with
    # sun_azimuth_deg=235, an Earth placed anywhere within about +/-90 deg of the
    # camera's own forward azimuth faces almost directly AWAY from the sun as seen
    # from Earth - i.e. it shows its night side, near-black, at EVERY elevation
    # (checked algebraically: the dot product of Earth's position with the sun's
    # direction-of-travel vector stays negative for the whole azimuth range
    # (-35, 145) regardless of elevation). A first pass placed Earth at azimuth 48
    # for exactly that framing reason and it rendered as a flat black disc - not a
    # bug, correct physics for a badly-chosen position. 200/22 instead gives a
    # bright, clearly gibbous Earth (checked: illuminated fraction ~0.8-0.83 under
    # both the shipped 12 deg sun and cine_light's 25 deg one) at the cost of
    # sitting outside the forward-looking presets' frame - see the "earthlight"
    # preset in render_still.py, aimed at this default, for a dedicated view.
    earth_azimuth_deg: float = 200.0
    earth_elevation_deg: float = 22.0

    farfield_enabled: bool = True
    # Half-width of the outer square boundary (matches world_size_m's own
    # half-width convention) - ~2.9 km beyond the 100 m physics edge.
    farfield_outer_radius_m: float = 3000.0
    # 26/56 (a first pass) left large, widely-spaced triangles once relief
    # amplitude picked up (see farfield_relief_ramp_m) - under directional
    # lighting a sparse radial mesh shades as visible flat-faceted "steps"
    # across a hillside, reported as "staircase terracing". Denser sampling,
    # especially in the near-to-mid range where relief is ramping in fastest,
    # removes the faceting without materially changing triangle budget (still
    # a few tens of thousands, trivial next to the near terrain/pebble field).
    farfield_shell_count: int = 42
    farfield_segments_per_side: int = 72
    # Distance outward from the physics boundary over which height blends from
    # "matches the real terrain's edge slope" to "independent far noise" - see
    # farfield.py's module docstring.
    farfield_blend_width_m: float = 45.0
    farfield_relief_m: float = 220.0
    # Distance (past the blend band) over which relief amplitude ramps from
    # _RELIEF_FLOOR up to full farfield_relief_m/crater-rim height - see
    # farfield._relief_ramp. A first pass had full-height (up to ~150 m) hills
    # starting right past the blend band, which loomed as a close-in
    # wall/overhang rather than reading as a receding horizon; ramping pushes
    # full height out to a genuinely distant range. 1100 m (the first value
    # tried) left a long near-flat stretch (only _RELIEF_FLOOR amplitude) that
    # dipped below an elevated camera's sightline to the near terrain's own
    # rim, showing as a gap of visible black sky between the two surfaces;
    # 700 m reaches full height sooner and measurably narrows that gap, but see
    # test_farfield_no_sky_gap_from_elevated_camera - it is not fully closed on
    # every seed yet.
    farfield_relief_ramp_m: float = 700.0
    farfield_crater_count: int = 24
    # How far the outermost shells are pulled down, so the mesh's own far edge
    # sinks below the sightline instead of showing as a hard line against the
    # sky - see farfield.py.
    farfield_horizon_drop_m: float = 320.0
    # Silhouette floor (see farfield.build_farfield_mesh): a shell may dip at
    # most this many metres, PLUS farfield_max_dip_slope_deg of its distance
    # from the boundary, below the running-max height already reached at that
    # same angular position. The distance-scaled part is what lets this apply
    # everywhere with no on/off boundary to hand-tune per seed - a fixed-metres
    # budget just relocates the same visible cliff to wherever the budget runs
    # out (measured: shells pinned flat at the ceiling, then an abrupt ~40 m
    # drop the moment the clamp switched off).
    farfield_max_dip_m: float = 12.0
    farfield_max_dip_slope_deg: float = 2.5
    # CURRENTLY UNUSED - farfield_model_sdf no longer textures the far field at
    # all (see that function's docstring: reusing the near terrain's albedo
    # still read as "crumpled foil" even stretched via this field, so the far
    # field went back to a flat colour). Left in place, harmless, in case a
    # properly distance-blurred/LOD'd texture is worth trying again later - the
    # UV coordinates build_farfield_mesh still writes (inert without a texture
    # to sample) use it.
    farfield_texture_tile_scale: float = 5.0

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

    # --- Visual-realism pass (terrain surface, textures, rocks) ---------------------
    # Everything below governs ONLY what gets drawn - never elevation_lookup, never the
    # collision boxes, never anything that seeds rock/crater placement. See
    # terrain_detail.build_drawn_surface, textures_world.py and rocks_visual.py.
    #
    # HARD BUDGET: the exported terrain.obj (what gz actually draws) may deviate from
    # elevation_lookup - the currently-shipped surface rocks are seated against and the
    # surface the collision boxes approximate - by at most this much, anywhere. Wheel
    # radius is 0.09 m; this is kept well under it so the rover's wheels (which still
    # contact the frozen collision boxes) never visibly float or sink relative to the
    # ground drawn under them. Measured worst-case deviation and rock-seating gaps across
    # seeds 42/7/123 are recorded in PROGRESS.md's terrain-realism-pass note.
    visual_surface_budget_m: float = 0.028
    # Of that budget, how much a bicubic-spline reconstruction of the coarse collision
    # control grid (replacing the piecewise-PLANAR per-cell tangent plane - see
    # terrain_detail.py) is allowed to move the surface. This is what removes the
    # cell-boundary creases and the dead-flat interior of each 5 m cell (the "quilt"),
    # and it gets almost the WHOLE budget, not a fraction of it - see the note below on
    # why an earlier split starved this term almost to nothing.
    #
    # Measured (seed 42): |spline - old| has mean 0.0145 m but p90 0.035 m and a max of
    # 0.51 m at the sharpest crater rims - so even devoting the full 0.028 m budget to
    # this term alone cannot fully close the gap everywhere. What it DOES buy: at
    # visual_macro_budget_m=0.026, 86% of all pixels are corrected EXACTLY to the smooth
    # spline value (no clipping at all), and the clipped 14% (concentrated at crater rims
    # the planner marks lethal and routes around anyway) are still moved 0.026 m toward
    # it, with a mean residual error of only ~0.004 m. That is what actually breaks up
    # the periodic 5 m grid on ordinary ground - see the note below for what happened
    # when this term had to share the budget with the fine-detail one.
    visual_macro_budget_m: float = 0.026
    # A SMALL, ABSOLUTE cap (not a fraction of the raw residual) on the sub-cell detail
    # layer - raw_heightmap minus its own per-cell average, i.e. the real fBm/crater
    # structure the collision-grid averaging discarded (see terrain_detail.py).
    #
    # An earlier version scaled that residual by a FRACTION (0.6) of its own natural
    # amplitude and only THEN clipped it against the shared budget. That was wrong:
    # measured, the raw residual's mean |value| is 0.19 m (crater sub-structure reaches
    # 2.4 m), so even at frac=0.6 it exceeded the whole 0.028 m budget almost everywhere -
    # meaning the final clip(macro + fine, budget) was saturated by the FINE term's sign
    # nearly everywhere, and the macro correction above (which is what actually targets
    # the cell-boundary creases) got crowded out to a few-millimetre nudge in the combined
    # result. Renders still showed the quilt clearly (a faint but unmistakable rectilinear
    # hatch, "corduroy" rather than hard facets) because the surface being drawn was still
    # overwhelmingly the OLD piecewise-planar one. Capping the fine layer's amplitude
    # directly, well under the macro budget, is what actually lets the macro correction
    # through - see PROGRESS.md's terrain-realism-pass note for the before/after.
    visual_fine_detail_amplitude_m: float = 0.006
    # Stride for the EXPORTED/drawn mesh only. elevation_lookup / rock seating keep using
    # terrain_mesh_stride (4) unchanged - this is a separate, finer stride purely for what
    # gz renders, since the visual mesh no longer has to be the identical triangulation
    # elevation_lookup reads. 1 = every post (0.39 m); the mesh has no LOD so this only
    # costs triangles, not correctness. See PROGRESS.md for the measured triangle/RTF cost.
    terrain_visual_mesh_stride: int = 1

    # World-scale texture bake (textures_world.py): one set of albedo/normal/roughness
    # PNGs mapped 1:1 over the whole world_size_m instead of tiling every
    # texture_tile_size_m, so macro features (crater ejecta rays, slope/AO darkening,
    # downslope dust streaking) can be baked ALIGNED to the real heightmap and craters,
    # and repetition (visible every 20 m before) goes away entirely. texture_resolution_px
    # / texture_tile_size_m above are left untouched - they still drive the original
    # generate_textures() call, kept only to preserve the shared RNG's draw sequence so
    # rock/crater placement stays byte-identical (see generate.py); their PNG output is
    # never shipped. See PROGRESS.md for the measured cost (bake time, PNG size, gz load
    # time, RTF) of this resolution vs. alternatives tried.
    texture_world_px: int = 4096

    # Rocks: shape variety. generate_rock_variants/scatter_rocks (rocks.py/scatter.py)
    # are UNCHANGED and still own the frozen collision ellipsoids and the shared RNG
    # draw sequence that places every rock. rocks_visual.py generates NEW, more varied
    # VISUAL-ONLY meshes on an independent RNG and fits each one, per axis, inside the
    # original variant's collision_radii - never the other way around. See rocks_visual.py.
    rock_visual_shrink: float = 0.97  # safety margin so a visual mesh cannot poke outside its frozen ellipsoid
    # Per-instance +/- LIGHTNESS multiplier (see rocks_visual.rock_albedo_tints - one
    # scalar per rock applied equally to r/g/b, never an independent per-channel
    # multiplier, which was tried at 0.28 and measured to produce visibly coloured
    # boulders - purple, teal, maroon - since an independent per-channel spread is a
    # saturation change, not a brightness one. 0.12 (a 0.88-1.12x range on the raised
    # base in worldgen._rock_model_sdf) gives a real, visible grey-to-grey spread -
    # real basalt boulders vary with weathering/dust cover - without ever touching hue.
    rock_albedo_variation: float = 0.12

    # Visual-only pebble field (terrain_detail.py/rocks_visual.py): non-collidable
    # stones welded into one mesh to break up the ground at rover scale, sitting ON TOP
    # of the drawn surface. They carry no collision and are not placed against
    # elevation_lookup, so they are free of the surface budget above - see PROGRESS.md
    # for why that is safe (a non-collidable stone the rover just drives over is not a
    # physics concern, only a visual one).
    #
    # count/radii were originally 4000 / 3-10 cm. Measured against a render (orbit and
    # terrain presets, seed 42): at that size, a pebble is at most a couple of pixels at
    # mid/far range, and antialiasing over a mostly-self-shadowed low-poly facet turns
    # that into a near-black speck - thousands of them read as sensor noise or dust on
    # the lens, not as stones. Fewer and bigger lets each one actually resolve into a
    # lit-and-shadowed shape instead of a single dark pixel.
    #
    # A first revision (900 at 7-22 cm) was still measured reading as small dark dots
    # from the orbit/terrain presets - better than the original but not yet resolving.
    # 500 at 10-28 cm gives each one roughly 2x the previous screen footprint at the same
    # camera distance, while staying under rocks.py's rock_scale_min_m (0.3 m) so pebbles
    # and boulders stay visually distinct categories.
    pebble_count: int = 500
    pebble_radius_min_m: float = 0.10
    pebble_radius_max_m: float = 0.28

    def __post_init__(self) -> None:
        if self.heightmap_resolution_px % 2 == 0:
            raise ValueError("heightmap_resolution_px should be odd (2^n + 1)")
