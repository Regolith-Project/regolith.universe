# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Compose the base terrain, regional slope, and crater field into a heightmap."""

from pathlib import Path

import numpy as np
from PIL import Image

from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.craters import apply_craters, place_craters
from regolith_terrain_gen.noise import fbm


def build_heightmap(cfg: TerrainConfig, rng: np.random.Generator) -> tuple:
    """Returns (heightmap_meters, craters, elevation_lookup) where elevation_lookup(x_m, y_m)
    samples terrain height in meters at a world-frame coordinate."""
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

    def elevation_lookup(x_m: float, y_m: float) -> float:
        px = int(np.clip((x_m + half_world) * px_per_m, 0, n - 1))
        py = int(np.clip((y_m + half_world) * px_per_m, 0, n - 1))
        return float(heightmap[py, px])

    return heightmap, craters, elevation_lookup


def save_heightmap_png(heightmap: np.ndarray, path: Path) -> None:
    """Save as a 16-bit grayscale PNG, the format Gazebo's heightmap geometry expects."""
    normalized = heightmap / heightmap.max() if heightmap.max() > 0 else heightmap
    as_uint16 = (np.clip(normalized, 0.0, 1.0) * 65535).astype(np.uint16)
    Image.fromarray(as_uint16, mode="I;16").save(path)


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


def build_terrain_collision_boxes_sdf(
    heightmap: np.ndarray,
    cfg: TerrainConfig,
    grid_resolution: int = 24,
    overlap_frac: float = 0.12,
    smoothing_passes: int = 3,
) -> str:
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
    anyway) and <0.15 m on average; spawn clearance is preserved (spawn Z comes
    from the fine heightmap via manifest, not from this smoothed grid).

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
    """
    n = heightmap.shape[0]
    block = max(1, (n - 1) // grid_resolution)
    usable = ((n - 1) // block) * block  # crop to a multiple of block for clean reshaping
    trimmed = heightmap[:usable, :usable]
    rows_blocks, cols_blocks = usable // block, usable // block
    averaged = trimmed.reshape(rows_blocks, block, cols_blocks, block).mean(axis=(1, 3))
    # Blur the cell-average heights so adjacent tilted slabs meet at their shared
    # edge (removes the curvature "lip" - see the SMOOTHING note above).
    surface = _smooth_surface(averaged, smoothing_passes)

    half_world = cfg.world_size_m / 2.0
    cell_size_m = cfg.world_size_m * (usable / (n - 1)) / rows_blocks
    xs = -half_world + (np.arange(cols_blocks) + 0.5) * cell_size_m
    ys = -half_world + (np.arange(rows_blocks) + 0.5) * cell_size_m

    # Local surface gradient (metres of height per metre), from central
    # differences of the SMOOTHED cell heights. axis=1 runs along +x (columns),
    # axis=0 along +y (rows). np.gradient uses one-sided differences at edges.
    grad_x = np.gradient(surface, cell_size_m, axis=1)
    grad_y = np.gradient(surface, cell_size_m, axis=0)

    slab_size_m = cell_size_m * (1.0 + overlap_frac)  # widen so neighbours overlap (no cracks)
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
