# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Compose the base terrain, regional slope, and crater field into a heightmap."""

from pathlib import Path

import numpy as np
from PIL import Image

from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.craters import apply_craters, place_craters
from regolith_terrain_gen.noise import fbm
from regolith_terrain_gen.terrain_mesh import mesh_surface_lookup


def build_heightmap(cfg: TerrainConfig, rng: np.random.Generator) -> tuple:
    """Returns (raw_heightmap, visual_heightmap, craters, elevation_lookup).

    raw_heightmap is the full fine-resolution fBm + regional-slope + crater terrain -
    kept around only so build_terrain_collision_boxes_sdf can re-derive the smoothed
    collision surface from it (collision geometry is a deterministic function of this
    array plus cfg).

    visual_heightmap is what actually gets rendered and looked up: the SAME smoothed,
    tilted-slab surface the collision boxes physically are, evaluated at full
    resolution (see _synthesize_visual_heightmap). Previously this function returned
    raw_heightmap for both rendering and lookups, which meant the rendered ground and
    the ground the rover's physics actually rested on were two different surfaces -
    diverging by more than the 0.09 m wheel radius across most of the spawn zone for
    several seeds (seed 42: mean +0.15 m, up to +0.35 m, 80% of the spawn zone exceeding
    the wheel radius) - the rover wasn't broken, it was just resting on an invisible
    surface below the one being drawn, so it visually rendered as sunk into the ground.
    Building visual_heightmap from the identical collision surface makes the gap zero
    by construction, for every seed.

    elevation_lookup(x_m, y_m) intentionally samples visual_heightmap, not
    raw_heightmap: rock placement (scatter.py) and the spawn-point manifest elevation
    both need the surface the rover/rocks actually sit on, or they'd reintroduce this
    exact float/embed mismatch themselves.
    """
    n = cfg.heightmap_resolution_px
    px_per_m = (n - 1) / cfg.world_size_m

    base = fbm(
        (n, n),
        rng,
        octaves=cfg.fbm_octaves,
        base_cell=cfg.fbm_base_cell_px,
        lacunarity=cfg.fbm_lacunarity,
        gain=cfg.fbm_gain,
    )
    heightmap = base * cfg.fbm_weight_m

    # Gentle regional slope in a random direction.
    slope_rad = np.deg2rad(cfg.regional_slope_deg)
    rise_over_world = np.tan(slope_rad) * cfg.world_size_m
    slope_dir = rng.uniform(0.0, 2.0 * np.pi)
    half_world = cfg.world_size_m / 2.0
    lin = np.linspace(-half_world, half_world, n)
    gx, gy = np.meshgrid(lin, lin)
    heightmap += (gx * np.cos(slope_dir) + gy * np.sin(slope_dir)) / cfg.world_size_m * rise_over_world

    craters = place_craters(cfg, rng)
    apply_craters(heightmap, cfg, craters, px_per_m)

    # Normalize to exactly the target height range, preserving relative shape.
    heightmap -= heightmap.min()
    span = heightmap.max()
    if span > 1e-9:
        heightmap *= cfg.height_range_m / span

    grid = _build_smoothed_surface(heightmap, cfg)
    visual_heightmap = np.clip(
        _synthesize_visual_heightmap(heightmap, cfg, grid), 0.0, cfg.height_range_m
    )

    # Evaluated on the triangles of the terrain MESH - the geometry gz actually draws
    # (see terrain_mesh.py for why the ground is a mesh and not a <heightmap>). Everything
    # that seats an object on the ground goes through here, so this has to be the drawn
    # surface and nothing else. Its two predecessors were both a different surface from
    # the drawn one and both showed up as floating rocks: first the NEAREST heightmap
    # post (up to 0.35 m above what gz interpolated between posts), then a BILINEAR
    # sample of every post (correct for a <heightmap>, but a mesh is piecewise-linear
    # over its own, sparser posts).
    elevation_lookup = mesh_surface_lookup(visual_heightmap, cfg)

    return heightmap, visual_heightmap, craters, elevation_lookup


def save_heightmap_png(heightmap: np.ndarray, path: Path) -> tuple:
    """Save as a 16-bit grayscale PNG and return (z_min_m, z_span_m), the vertical
    offset and range worldgen.py must feed into the <heightmap> element's <pos> z and
    <size> z so that gz renders this array back at its true absolute elevations.

    CRITICAL - gz normalizes the heightmap image by its OWN min/max, not linearly.
    gz-sim's ogre2 heightmap stretches whatever pixel range the PNG actually contains
    to fill [0, <size> z] (its lowest pixel renders at <pos> z, its highest at
    <pos> z + <size> z). It does NOT map pixel/65535 -> height*<size>z linearly. So the
    ONLY way to control the absolute rendered elevation is to (a) let the PNG use the
    full 0..65535 range and (b) tell gz, via <pos> z and <size> z, what real-world
    min and span that full range corresponds to. Then:

        rendered(x,y) = pos_z + (pixel-min)/(max-min) * size_z
                      = z_min + (H - z_min)/(z_max - z_min) * (z_max - z_min)
                      = H(x,y)                                     (exact, every pixel)

    A PREVIOUS fix did the opposite - it deliberately encoded only a PARTIAL range
    (heightmap/height_range_m, peaking well below 1.0) on the assumption gz maps pixels
    linearly, so that partial range would land at the right absolute heights. Because gz
    actually min/max-stretches, that partial-range PNG got stretched back up: the rendered
    ground floated ~0.2-0.5 m ABOVE the collision boxes (which are built from these same
    absolute metres), so the rover, correctly resting on the collision surface, rendered
    sunk into / under the visibly-drawn ground. See PROGRESS.md "The rover is STILL
    underground - gz heightmap min/max normalization".

    Collision geometry and elevation_lookup keep using the absolute-metre surface
    unchanged; only the PNG encoding + the <pos>/<size> z that decode it change, so the
    two surfaces now coincide by construction for every seed.

    CRITICAL, SEPARATELY - gz reads the image TRANSPOSED relative to this array. Every
    array here is indexed [row = y, col = x], but gz maps the image's first axis to world
    X and its second to world Y, so handing it the array as-is renders the terrain
    mirrored about the x = y diagonal. That is a purely HORIZONTAL error, invisible to
    every check that compares arrays to arrays (the collision boxes, elevation_lookup and
    the visual array are all built in this module's convention and so all agreed with each
    other), and invisible on the diagonal itself. It was found by rendering a heightmap
    with a single 25 m spike at world (+60, 0) and screenshotting from directly overhead:
    the spike drew at (0, +60). A second spike at (0, -30) drew at (-30, 0). Writing
    `heightmap.T` puts both back where they belong (measured: (54.7, 6.3) and (-5.2,
    -26.1), the residual being the offset of a peak's sunlit face from its apex).

    This is what made rocks visibly float: they are seated on elevation_lookup, which
    matches the COLLISION surface, while the ground being DRAWN was that surface
    transposed - so a rock sat in the air wherever surface(y, x) < surface(x, y), and
    sank wherever it was greater. See PROGRESS.md "Rendered terrain was transposed"."""
    z_min = float(heightmap.min())
    z_max = float(heightmap.max())
    span = z_max - z_min
    if span > 1e-9:
        normalized = (heightmap - z_min) / span
    else:
        # Perfectly flat terrain: any constant renders flat; hand back a unit span so
        # worldgen's <size> z stays positive and <pos> z carries the constant height.
        normalized = np.zeros_like(heightmap)
        span = 1.0
    as_uint16 = (np.clip(normalized, 0.0, 1.0) * 65535).astype(np.uint16)
    # .T (not a flip/rotation) - gz's image axes are (x, y), this module's are (y, x).
    # np.ascontiguousarray because PIL needs a contiguous buffer for I;16.
    Image.fromarray(np.ascontiguousarray(as_uint16.T), mode="I;16").save(path)
    return z_min, span


def _smooth_surface(averaged: np.ndarray, passes: int) -> np.ndarray:
    """Separable [1, 2, 1]/4 blur with edge replication, applied `passes` times.

    Collapses the collision-slab boundary "lips" (see the SMOOTHING note in
    build_terrain_collision_boxes_sdf). The residual lip between two adjacent
    tilted slabs equals the local terrain CURVATURE (the discrete Laplacian of
    the cell averages, /4): a slab centred on its raw cell average has a top
    edge that overshoots its neighbour's wherever the surface bends, even on
    otherwise near-flat ground. Blurring the centre heights before tilting
    removes that curvature term, so neighbouring slab edges very nearly meet.
    Measured on seed 42 at grid_resolution=24: max boundary lip drops from
    1.11 m (raw) to 0.10 m at 3 passes, and the fraction of boundaries with a
    lip taller than the 0.09 m wheel radius from 31 % to 0 % - all with the SAME
    576 boxes, so physics real-time factor is unchanged."""
    a = averaged
    for _ in range(max(0, passes)):
        ap = np.pad(a, ((0, 0), (1, 1)), mode="edge")
        a = (ap[:, :-2] + 2.0 * ap[:, 1:-1] + ap[:, 2:]) / 4.0
        ap = np.pad(a, ((1, 1), (0, 0)), mode="edge")
        a = (ap[:-2, :] + 2.0 * ap[1:-1, :] + ap[2:, :]) / 4.0
    return a


def _build_smoothed_surface(heightmap: np.ndarray, cfg: TerrainConfig) -> dict:
    """The coarse, blurred, per-cell tangent-plane model of the terrain that
    build_terrain_collision_boxes_sdf turns into physics boxes AND
    _synthesize_visual_heightmap turns into the rendered ground - factored out
    so both can never drift out of sync with each other. See
    build_terrain_collision_boxes_sdf's docstring for why each step (block
    averaging, blurring, per-cell tilt) exists.

    cfg.collision_grid_resolution/collision_overlap_frac/collision_smoothing_passes
    were previously separate keyword defaults on build_terrain_collision_boxes_sdf;
    moved onto TerrainConfig so this helper and the box-emitting/visual-synthesis
    callers are structurally guaranteed to use the same values.
    """
    n = heightmap.shape[0]
    grid_resolution = cfg.collision_grid_resolution
    block = max(1, (n - 1) // grid_resolution)
    usable = ((n - 1) // block) * block  # crop to a multiple of block for clean reshaping
    trimmed = heightmap[:usable, :usable]
    rows_blocks, cols_blocks = usable // block, usable // block
    averaged = trimmed.reshape(rows_blocks, block, cols_blocks, block).mean(axis=(1, 3))
    # Blur the cell-average heights so adjacent tilted slabs meet at their shared
    # edge (removes the curvature "lip" - see the SMOOTHING note on
    # build_terrain_collision_boxes_sdf).
    surface = _smooth_surface(averaged, cfg.collision_smoothing_passes)

    half_world = cfg.world_size_m / 2.0
    cell_size_m = cfg.world_size_m * (usable / (n - 1)) / rows_blocks
    xs = -half_world + (np.arange(cols_blocks) + 0.5) * cell_size_m
    ys = -half_world + (np.arange(rows_blocks) + 0.5) * cell_size_m

    # Local surface gradient (metres of height per metre), from central
    # differences of the SMOOTHED cell heights. axis=1 runs along +x (columns),
    # axis=0 along +y (rows). np.gradient uses one-sided differences at edges.
    grad_x = np.gradient(surface, cell_size_m, axis=1)
    grad_y = np.gradient(surface, cell_size_m, axis=0)

    return {
        "block": block,
        "usable": usable,
        "rows_blocks": rows_blocks,
        "cols_blocks": cols_blocks,
        "cell_size_m": cell_size_m,
        "half_world": half_world,
        "xs": xs,
        "ys": ys,
        "surface": surface,
        "grad_x": grad_x,
        "grad_y": grad_y,
    }


def _synthesize_visual_heightmap(heightmap: np.ndarray, cfg: TerrainConfig, grid: dict) -> np.ndarray:
    """Evaluate the SAME tilted collision plane build_terrain_collision_boxes_sdf turns
    into physics boxes, at every full-resolution heightmap pixel - so the rendered
    <heightmap> visual is, cell by cell, the exact surface the rover physically stands
    on (see the mismatch note in build_heightmap). Pixels beyond the cropped ``usable``
    region (a <3 m strip at the +x/+y edge lost to the block-size crop, see
    _build_smoothed_surface) are clamped to their nearest valid cell's plane - that
    strip has no collision coverage at all regardless, a pre-existing edge condition
    well outside the ~9 m spawn zone the rover actually operates in.
    """
    n = heightmap.shape[0]
    px_per_m = (n - 1) / cfg.world_size_m
    rows_blocks, cols_blocks, block = grid["rows_blocks"], grid["cols_blocks"], grid["block"]
    half_world, xs, ys = grid["half_world"], grid["xs"], grid["ys"]
    surface, grad_x, grad_y = grid["surface"], grid["grad_x"], grid["grad_y"]

    row_idx = np.clip(np.arange(n) // block, 0, rows_blocks - 1)
    col_idx = np.clip(np.arange(n) // block, 0, cols_blocks - 1)

    h0 = surface[row_idx[:, None], col_idx[None, :]]
    gx = grad_x[row_idx[:, None], col_idx[None, :]]
    gy = grad_y[row_idx[:, None], col_idx[None, :]]

    coord = -half_world + np.arange(n) / px_per_m
    dx = np.broadcast_to((coord - xs[col_idx])[None, :], (n, n))
    dy = np.broadcast_to((coord - ys[row_idx])[:, None], (n, n))

    return h0 + gx * dx + gy * dy


def build_terrain_collision_boxes_sdf(heightmap: np.ndarray, cfg: TerrainConfig) -> str:
    """Box collision(s) approximating the terrain - <collision> elements to
    embed in the SAME model/link as the visual <heightmap> (see worldgen.py).

    Both gz-sim's native <heightmap> collision geometry AND generic <mesh>
    collision geometry are unimplemented for dartsim in this gz-sim 8 /
    gz-physics 7.8.0 install ("Heightmap/Mesh construction from an SDF has not
    been implemented yet for dartsim" - visible only at -v 4 debug verbosity;
    reproduced identically under the bullet and bullet-featherstone engine
    plugins too, so it's not dartsim-specific or fixable by picking a
    different engine here). Box primitives are the one geometry type
    confirmed to work correctly (wheel and chassis collisions already rely
    on them). This is the pragmatic version of the plan's "static mesh
    instead of heightmap" fallback, substituting boxes since neither
    heightmap nor mesh collision works at all.

    TILTED SLABS (the flip fix - see PROGRESS.md M4/M5 and heightmap.py's own
    prior notes). Earlier versions used AXIS-ALIGNED, FLAT-TOPPED boxes rooted
    at z=0. Because each cell's flat top sits at that cell's average height,
    adjacent cells differ by a vertical CLIFF equal to the height difference
    between neighbours. Measured on seed 42 at grid_resolution=24: those steps
    average 0.31 m and reach 2.27 m, against a 0.09 m wheel radius - 81 % of
    cell boundaries are a step taller than the wheel radius. Driving (and
    especially turning) across such a cliff yields a sudden horizontal contact
    normal that pitches/rolls the chassis - the documented flip.

    Instead of a flat top, each cell's box is TILTED to match the terrain's
    local gradient (a "shingle"), so its top face approximates the true
    tangent plane at the cell centre. Adjacent tilted slabs then very nearly
    meet at their shared edge (both approximate the same underlying smooth
    surface there), collapsing the cliff to a small residual "lip". Slabs are
    also widened by ``overlap_frac`` so neighbours overlap slightly and a wheel
    can't drop into a crack between them; whichever slab is higher at the seam
    simply carries the wheel.

    SMOOTHING (the second and decisive part of the flip fix). Tilting alone was
    NOT enough: with each slab centred on its RAW cell-average height, the
    residual lip between two adjacent tilted slabs equals the local terrain
    CURVATURE (the discrete Laplacian of the cell averages, /4). That is
    non-zero even on near-flat ground wherever the surface gently bends, and it
    was measured up to 1.11 m on seed 42 at grid_resolution=24 (12x the 0.09 m
    wheel radius), with 31 % of all cell boundaries stepping taller than the
    wheel radius. Driving - especially turning - across such a step yields a
    sudden horizontal contact normal that rolls the chassis: this was the
    documented flip, and it reproduced on FLAT terrain (~1.7 deg slope), not
    just crater rims, which is why "avoid steep cells" alone never fixed it.
    The fix is ``_smooth_surface``: blur the cell-average heights with a
    separable [1,2,1] kernel (``smoothing_passes`` times) BEFORE tilting, which
    removes the curvature term so neighbouring slab edges meet. At 3 passes the
    max boundary lip drops to 0.10 m and 0 % of boundaries exceed the wheel
    radius, using the SAME 576 boxes (RTF unchanged - the RTF-vs-resolution
    trade-off recorded under M4 came from box COUNT, and smoothing adds none).
    The blur shifts the collision surface by at most ~1.5 m only at the
    sharpest crater rims (which the planner marks lethal and routes around
    anyway) and <0.15 m on average relative to the pre-blur block average - see
    the VISUAL/COLLISION MATCH note below for why that shift is no longer a
    concern for spawn clearance or anything else that reads elevation_lookup.

    Each grid cell's centre height is the AVERAGE (not a single strided
    sample) of the full-resolution heightmap over that cell's footprint;
    strided sampling would keep sharp single-pixel crater rims as isolated
    spikes. Both the slab centre height and the tilt (gradient) are taken from
    the SMOOTHED surface, so a slab and its neighbour approximate the same
    field at their shared edge and the seams line up (a per-cell least-squares
    plane fit, tried and discarded, minimises in-cell error but NOT the
    inter-cell seam, and measured worse: 0.17 m mean lip).

    Spawn height gotcha (cost a long debugging session, worth recording): a
    slab's centre height is the LOCAL average, which can be several meters
    even near the nominal "origin" spawn point if the regional slope/base
    terrain happens to sit high there for a given seed. Whatever spawns a
    dynamic body into this world MUST look up the actual local elevation (see
    manifest.json's spawn_zone.elevation_m, written by generate.py using the
    same elevation_lookup this module's build_heightmap returns) rather than
    assuming spawn height 0 - spawning below/inside solid collision geometry
    produces erratic, hard-to-diagnose physics (values that look like a
    frozen body, a body falling through everything, or exploding away to
    absurd coordinates, depending on exactly how deep the initial overlap is).

    grid_resolution is still a stability/performance knob (box count is
    grid_resolution**2 and RTF falls off with it), but tilting means the
    default 24 is now genuinely smooth enough to drive rather than a
    reluctant compromise: finer grids help the residual lip only marginally
    for a large RTF cost.

    VISUAL/COLLISION MATCH: the rendered <heightmap> visual (worldgen.py) is no
    longer the raw fine heightmap - it's _synthesize_visual_heightmap evaluating
    this SAME surface/grad_x/grad_y at full resolution (see build_heightmap), so
    the "spawn clearance"/"<0.15 m on average" mismatch this docstring used to
    describe as an accepted trade-off no longer exists: the ground you see and
    the ground the rover stands on are now the identical surface, by construction,
    for every seed.
    """
    grid = _build_smoothed_surface(heightmap, cfg)
    rows_blocks, cols_blocks = grid["rows_blocks"], grid["cols_blocks"]
    cell_size_m, xs, ys = grid["cell_size_m"], grid["xs"], grid["ys"]
    surface, grad_x, grad_y = grid["surface"], grid["grad_x"], grid["grad_y"]

    slab_size_m = cell_size_m * (1.0 + cfg.collision_overlap_frac)  # widen so neighbours overlap (no cracks)
    collisions = []
    for row in range(rows_blocks):
        for col in range(cols_blocks):
            h0 = float(surface[row, col])
            gx = float(grad_x[row, col])
            gy = float(grad_y[row, col])
            norm = float(np.sqrt(1.0 + gx * gx + gy * gy))
            # Orient local +Z along the surface normal n = (-gx, -gy, 1)/norm.
            # For SDF rpy (R = Rz*Ry*Rx) this is: roll = asin(gy/norm),
            # pitch = -atan(gx). Yaw is irrelevant for a square slab.
            roll = float(np.arcsin(gy / norm))
            pitch = float(-np.arctan(gx))
            # Thickness along the normal. A small CONSTANT is deliberate: the
            # rover only ever contacts slab TOPS (there is no ground plane and
            # nothing lives under the terrain), so the slab just needs to be
            # thick enough that (a) it can't be tunnelled through at driving
            # speed and (b) overlapping neighbours still abut vertically across
            # the largest residual lip (~1.1 m on seed 42). 2.5 m covers both.
            # Crucially it must NOT scale with h0: a tall slab (up to ~11 m on
            # high terrain) has a huge vertical AABB, so the rover's broadphase
            # ends up "near" far more slabs than it touches, which measurably
            # tanked RTF (0.29 vs 0.43) for no collision benefit.
            thickness = 2.5
            # Place the box so its TOP-face centre sits at (xc, yc, h0):
            # box_centre = top_centre - (thickness/2) * n.
            cx = xs[col] + (thickness / 2.0) * (gx / norm)
            cy = ys[row] + (thickness / 2.0) * (gy / norm)
            cz = h0 - (thickness / 2.0) * (1.0 / norm)
            collisions.append(
                f'<collision name="terrain_box_{row}_{col}">'
                f"<pose>{cx:.3f} {cy:.3f} {cz:.4f} {roll:.4f} {pitch:.4f} 0</pose>"
                f"<geometry><box><size>{slab_size_m:.4f} {slab_size_m:.4f} {thickness:.4f}</size></box></geometry>"
                f"<surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface>"
                f"</collision>"
            )
    return "\n".join(collisions)
