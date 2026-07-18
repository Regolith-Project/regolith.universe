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


def build_terrain_collision_boxes_sdf(
    heightmap: np.ndarray, cfg: TerrainConfig, grid_resolution: int = 24
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

    Each grid cell's height is the AVERAGE (not a single strided sample) of
    the full-resolution heightmap over that cell's footprint, which matters:
    strided sampling preserves sharp single-pixel features (like crater rims)
    as isolated spikes, producing large steps between adjacent cells that are
    tall relative to the rover's ~0.09 m wheel radius. Averaging acts as a
    low-pass filter, trading exact rim sharpness in the collision shape
    (visual rims stay sharp) for steps small enough to actually drive over.

    Spawn height gotcha (cost a long debugging session, worth recording): a
    box's height here is the LOCAL average, which can be several meters even
    near the nominal "origin" spawn point if the regional slope/base terrain
    happens to sit high there for a given seed. Whatever spawns a dynamic
    body into this world MUST look up the actual local elevation (see
    manifest.json's spawn_zone.elevation_m, written by generate.py using the
    same elevation_lookup this module's build_heightmap returns) rather than
    assuming spawn height 0 - spawning below/inside solid collision geometry
    produces erratic, hard-to-diagnose physics (values that look like a
    frozen body, a body falling through everything, or exploding away to
    absurd coordinates, depending on exactly how deep the initial overlap is).

    Resolution is a genuine stability/performance trade-off (see PROGRESS.md
    M4): coarser grids (e.g. 24) have taller steps at cell boundaries, which
    can flip the rover when crossing one during autonomous driving; finer
    grids (e.g. 64, 4096 total boxes) smooth that out but tank the physics
    step rate badly in this environment (real-time factor dropped to ~0.09,
    vs. ~0.5-0.6 at 24) - all those boxes live in one link now (see the
    module-level note above), so it's not the "many collisions -> freeze"
    issue from M2, just the raw cost of that many collision shapes. 24 is the
    middle-ground default, not a fully validated sweet spot.
    """
    n = heightmap.shape[0]
    block = max(1, (n - 1) // grid_resolution)
    usable = ((n - 1) // block) * block  # crop to a multiple of block for clean reshaping
    cropped = heightmap[: usable + 1, : usable + 1]
    trimmed = cropped[:usable, :usable]
    rows_blocks, cols_blocks = usable // block, usable // block
    averaged = trimmed.reshape(rows_blocks, block, cols_blocks, block).mean(axis=(1, 3))

    half_world = cfg.world_size_m / 2.0
    cell_size_m = cfg.world_size_m * (usable / (n - 1)) / rows_blocks
    xs = -half_world + (np.arange(cols_blocks) + 0.5) * cell_size_m
    ys = -half_world + (np.arange(rows_blocks) + 0.5) * cell_size_m

    collisions = []
    for row in range(rows_blocks):
        for col in range(cols_blocks):
            height = max(float(averaged[row, col]), 0.05)  # avoid zero-thickness boxes
            collisions.append(
                f'<collision name="terrain_box_{row}_{col}">'
                f'<pose>{xs[col]:.3f} {ys[row]:.3f} {height/2.0:.4f} 0 0 0</pose>'
                f"<geometry><box><size>{cell_size_m:.4f} {cell_size_m:.4f} {height:.4f}</size></box></geometry>"
                f"<surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface>"
                f"</collision>"
            )
    return "\n".join(collisions)
