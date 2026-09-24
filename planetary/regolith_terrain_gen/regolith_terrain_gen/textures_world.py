# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""World-scale, per-seed baked terrain textures: albedo/normal/roughness mapped 1:1 over
the whole world instead of tiling every ``texture_tile_size_m`` (see terrain_mesh.py's
``save_terrain_mesh_obj(..., uv_world_size_m=...)``).

WHY A WORLD-SCALE BAKE
-----------------------
A single UV set shared by albedo/normal/roughness, tiled every 20 m, cannot carry macro
features that belong to a SPECIFIC place in the world - a crater's ejecta blanket has to
sit around that crater, not repeat every tile whether or not a crater is there. The old
texture also repeated its sub-resolution crater pitting on the same 20 m period, which
reads as an obviously synthetic pattern once you notice it (see the task brief's defect
3). Mapping the texture once over the whole 200 m world at high resolution lets every
macro feature below be placed at its REAL position, computed from the REAL heightmap and
REAL crater list, and removes tiling repetition at the macro scale entirely.

COST, MEASURED: see PROGRESS.md's terrain-realism-pass note for bake time, PNG size, gz
load time and the RTF impact of texture_world_px, compared against the previous 512 px /
20 m tile (25.6 px/m) scheme. The trade given up is texel density at extreme close range:
texture_world_px / world_size_m is lower than the old tile's px/m for the same
texture_world_px. A first version of the fine-detail layer tried to buy that density back
with a small REPEATING tile (np.tile), on the theory that a sub-metre period would read
as material grain rather than an obvious stamp. Measured wrong: from altitude, enough
repeats are visible at once that the perfectly regular grid read as a woven/crosshatched
fabric pattern - see ``generate_world_textures``'s note on the normal-map layers for what
replaced it (unique, non-tiled noise at several incommensurate scales instead).

RNG: everything here runs on an INDEPENDENT generator the caller creates fresh from the
seed (see generate.py) - never the shared generator build_heightmap/generate_rock_variants/
scatter_rocks consume. Nothing here may perturb that stream: rock and crater placement in
the manifest has to stay byte-identical to the pre-existing generator for the same seed.
"""

from pathlib import Path

import numpy as np
from scipy import ndimage
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.noise import value_noise_2d


def _normalize01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _to_uint8(normalized: np.ndarray) -> np.ndarray:
    return (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)


\
# _tiled_detail (removed) used to generate a small noise pattern once and repeat it
# (np.tile) across the whole texture, on the theory that a sub-metre repeat period would
# read as "material grain" rather than an obvious stamp the way the old 20 m macro tile
# did. Measured against a render (the orbit preset, seed 42), that was wrong: at typical
# orbital/aerial range enough repeats are visible at once that np.tile's perfectly
# regular grid reads immediately as a woven/crosshatched fabric pattern - MORE obviously
# synthetic than the macro tiling this whole module exists to get rid of, not less. Fine
# detail below is now genuinely unique noise evaluated once across the full n_tex canvas
# (bounded by texture_world_px, not by an artificial small-tile resolution), which cannot
# exhibit exact-period repetition because there is no repeated tile to see.


def _crater_ejecta(
    n_tex: int, px_per_m: float, half_world: float, craters: list, rng: np.random.Generator
) -> np.ndarray:
    """Bright ejecta blanket + ray streaks around each crater, positioned at its REAL
    (x, y) - a texture-space analogue of craters.apply_craters, but painting brightness
    instead of sculpting height, and reaching further out (rays are the whole point).

    Each crater gets its own "freshness" (0..1, brighter/rayed the higher) and ray-count/
    phase, so 160 craters don't all look identical - some read as old, muted, ejecta-less
    craters (most of a real lunar surface is exactly that) and a handful read as young,
    bright, rayed ones (like Tycho/Copernicus stand out against Oceanus Procellarum).
    """
    field = np.zeros((n_tex, n_tex), dtype=np.float64)
    for crater in craters:
        freshness = rng.beta(1.5, 4.0)  # skewed toward "old, muted" - most craters are
        if freshness < 0.12:
            continue  # a good fraction of craters get no ejecta signature at all
        n_rays = rng.integers(5, 14)
        ray_phase = rng.uniform(0, 2 * np.pi)
        ray_sharpness = rng.uniform(2.0, 5.0)

        radius_m = crater.diameter_m / 2.0
        reach_m = radius_m * (2.2 + 3.0 * freshness)  # fresher craters throw rays further
        reach_px = int(np.ceil(reach_m * px_per_m))
        cx_px = (crater.x_m + half_world) * px_per_m
        cy_px = (crater.y_m + half_world) * px_per_m
        x0, x1 = max(0, int(cx_px - reach_px)), min(n_tex, int(cx_px + reach_px))
        y0, y1 = max(0, int(cy_px - reach_px)), min(n_tex, int(cy_px + reach_px))
        if x0 >= x1 or y0 >= y1:
            continue

        xs = (np.arange(x0, x1) - cx_px) / px_per_m
        ys = (np.arange(y0, y1) - cy_px) / px_per_m
        gx, gy = np.meshgrid(xs, ys)
        r_norm = np.hypot(gx, gy) / radius_m
        theta = np.arctan2(gy, gx)

        # Bright rim + blanket, fading with distance past the rim.
        blanket = np.exp(-np.clip((r_norm - 1.0), 0.0, None) / (0.7 + 1.3 * freshness))
        blanket = np.where(r_norm >= 0.75, blanket, 0.0)
        # Ray modulation: a directional pattern with n_rays lobes, sharpened so rays read
        # as streaks rather than a smooth donut.
        rays = 0.5 + 0.5 * np.cos(n_rays * theta + ray_phase)
        rays = rays**ray_sharpness
        field[y0:y1, x0:x1] += freshness * blanket * (0.35 + 0.65 * rays)
    return field


def _downslope_streaks(
    hm_tex: np.ndarray, px_per_m: float, rng: np.random.Generator, steps: int = 6, step_m: float = 1.6
) -> np.ndarray:
    """Directional darkening along the local downhill flow direction - dust that has
    visibly migrated downslope, not just a slope-magnitude tint.

    Implemented as an advection blur: a base noise-like field (the height map's own
    high-pass residual, which already has grain) is repeatedly resampled a small step
    further downhill and averaged with the running total, so features smear out along
    flow lines instead of staying isotropic. Cheap relative to a real erosion sim and
    good enough to read as streaking at rover/orbital range.

    A per-pixel angular jitter (drawn from a smooth, low-frequency noise field, not
    independent per pixel - that would just cancel the streaking instead of bending it)
    is added to the advection direction each step. Without it, wherever the local slope
    direction is nearly constant over several metres (gentle ground, e.g. near the
    spawn zone), the repeated straight-line advection produced dead-straight, closely
    parallel lines - measured against a render as a "brushed"/combed look in the near
    foreground, not the irregular dust migration this was meant to be. Real downslope
    creep does not travel in perfectly straight lines either.
    """
    gy, gx = np.gradient(hm_tex, 1.0 / px_per_m)
    slope = np.hypot(gx, gy) + 1e-9
    # Unit downhill direction (steepest descent = -gradient).
    dx, dy = -gx / slope, -gy / slope

    n = hm_tex.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    base = hm_tex - ndimage.uniform_filter(hm_tex, size=max(3, int(0.4 * px_per_m)))
    jitter_angle = value_noise_2d((n, n), 2.0 * px_per_m, rng) * 0.6  # +/- ~0.6 rad wobble
    acc = base.copy()
    cx, cy = xx.copy(), yy.copy()
    step_px = step_m * px_per_m
    for i in range(steps):
        # Wobble strengthens with distance travelled, so streaks curve gently instead of
        # kinking sharply right at the source.
        wobble = jitter_angle * (i + 1)
        c, s = np.cos(wobble), np.sin(wobble)
        dxw = dx * c - dy * s
        dyw = dx * s + dy * c
        cx = cx + dxw * step_px
        cy = cy + dyw * step_px
        sampled = ndimage.map_coordinates(base, [np.clip(cy, 0, n - 1), np.clip(cx, 0, n - 1)], order=1, mode="nearest")
        acc += sampled
    streak = acc / (steps + 1)
    # Weight by slope so flat ground (nothing to slide down) stays unstreaked.
    slope_norm = _normalize01(np.clip(slope, 0, np.percentile(slope, 98)))
    return _normalize01(streak) * slope_norm


def _dust_skirts(n_tex: int, px_per_m: float, half_world: float, rocks: list) -> np.ndarray:
    """Soft brightening halo around each rock's ground footprint - regolith visibly
    mounded/disturbed where a boulder sits, rather than every rock meeting bare,
    untouched ground. Purely a texture effect; costs no geometry or collision."""
    field = np.zeros((n_tex, n_tex), dtype=np.float64)
    for rock in rocks:
        skirt_r_m = rock.scale_m * 1.6
        reach_px = int(np.ceil(skirt_r_m * px_per_m))
        cx_px = (rock.x_m + half_world) * px_per_m
        cy_px = (rock.y_m + half_world) * px_per_m
        x0, x1 = max(0, int(cx_px - reach_px)), min(n_tex, int(cx_px + reach_px))
        y0, y1 = max(0, int(cy_px - reach_px)), min(n_tex, int(cy_px + reach_px))
        if x0 >= x1 or y0 >= y1:
            continue
        xs = (np.arange(x0, x1) - cx_px) / px_per_m
        ys = (np.arange(y0, y1) - cy_px) / px_per_m
        gx, gy = np.meshgrid(xs, ys)
        r_norm = np.hypot(gx, gy) / skirt_r_m
        skirt = np.exp(-((r_norm / 0.55) ** 2)) * np.clip(1.0 - r_norm, 0.0, 1.0)
        field[y0:y1, x0:x1] = np.maximum(field[y0:y1, x0:x1], skirt)
    return field


def _small_pits(n_tex: int, px_per_m: float, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sub-6 m bowl-and-rim pitting (craters too small to survive the collision grid -
    see config.py's crater_count note), scattered at UNIQUE world positions instead of
    stamped once per repeating tile - so the pitting no longer repeats on a visible
    period the way the old tiled texture's did."""
    field = np.zeros((n_tex, n_tex), dtype=np.float64)
    cx_all = rng.uniform(0, n_tex, size=count)
    cy_all = rng.uniform(0, n_tex, size=count)
    r_all = rng.uniform(0.3, 2.2, size=count) / 2.0 * px_per_m
    for cx, cy, radius_px in zip(cx_all, cy_all, r_all):
        reach = int(np.ceil(radius_px * 1.6))
        x0, x1 = max(0, int(cx - reach)), min(n_tex, int(cx + reach))
        y0, y1 = max(0, int(cy - reach)), min(n_tex, int(cy + reach))
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1) - cx
        ys = np.arange(y0, y1) - cy
        gx, gy = np.meshgrid(xs, ys)
        x_norm = np.hypot(gx, gy) / max(radius_px, 1e-6)
        bowl = np.where(x_norm <= 1.0, -(1.0 - x_norm**2), 0.0)
        rim = np.exp(-(((x_norm - 1.0) / 0.35) ** 2))
        field[y0:y1, x0:x1] += bowl + 0.3 * rim
    return field


def generate_world_textures(
    output_dir: Path,
    cfg: TerrainConfig,
    drawn_surface: np.ndarray,
    craters: list,
    rocks: list,
    rng: np.random.Generator,
) -> dict:
    """Bake albedo/normal/roughness once over the whole world and write them to
    output_dir. Returns the same {"albedo": path, "normal": path, "roughness": path}
    shape the old generate_textures() did, so worldgen.py's call site barely changes."""
    n_tex = cfg.texture_world_px
    half_world = cfg.world_size_m / 2.0
    px_per_m = n_tex / cfg.world_size_m

    zoom = n_tex / drawn_surface.shape[0]
    hm_tex = ndimage.zoom(drawn_surface, zoom, order=3)

    gy, gx = np.gradient(hm_tex, 1.0 / px_per_m)
    slope = np.hypot(gx, gy)
    slope_norm = _normalize01(np.clip(slope, 0, np.percentile(slope, 99)))
    curvature = ndimage.gaussian_laplace(hm_tex, sigma=max(1.0, 0.8 * px_per_m))
    concavity = _normalize01(np.clip(-curvature, 0, np.percentile(np.abs(curvature), 95)))

    macro_variation = _normalize01(value_noise_2d((n_tex, n_tex), n_tex / 9.0, rng))
    patch_variation = _normalize01(value_noise_2d((n_tex, n_tex), n_tex / 28.0, rng))

    # --- Albedo -------------------------------------------------------------------
    base_grey = 0.40 + 0.08 * macro_variation + 0.05 * patch_variation
    ejecta = _crater_ejecta(n_tex, px_per_m, half_world, craters, rng)
    streaks = _downslope_streaks(hm_tex, px_per_m, rng)
    skirts = _dust_skirts(n_tex, px_per_m, half_world, rocks)

    brightness = base_grey
    brightness *= 1.0 - 0.16 * slope_norm  # steeper faces read slightly darker
    brightness *= 1.0 - 0.14 * concavity  # crater floors/bowls: ambient-occluded
    brightness *= 1.0 - 0.10 * streaks  # dust migration darkens the streak itself
    brightness += 0.55 * ejecta  # bright ejecta blankets/rays override toward white
    brightness += 0.10 * skirts  # disturbed regolith halo around rocks

    grey = np.clip(brightness, 0.06, 0.92)
    albedo = np.stack([grey, grey, grey * 1.01], axis=-1)

    # --- Normal map -----------------------------------------------------------------
    # A synthetic "detail height" purely for shading - never geometry.
    #
    # An earlier version summed a scaled copy of the real macro elevation (hm_tex) in
    # here too, on the theory that it would "correlate the bake with what the mesh
    # itself already does". Measured mistake: hm_tex varies by METRES over the texture,
    # so its gradient at texture resolution is one to two orders of magnitude larger
    # than any of the fine layers below, and it dominated np.gradient(detail_height)
    # almost everywhere - which is exactly why renders showed "vague smudges" instead of
    # pitting: the normal map was mostly re-deriving the SAME macro slope the mesh's own
    # vertex normals already convey, at the cost of burying the actual micro-detail
    # underneath it. The mesh already carries macro slope; this map's only job is
    # everything the mesh geometry cannot resolve (rover/boot scale and below), so macro
    # elevation is left out entirely now.
    #
    # Four genuinely different spatial scales, all well below what geometry can resolve,
    # and all UNIQUE noise evaluated once across the full n_tex canvas - no np.tile, no
    # repeated pattern of any kind (see the removed _tiled_detail's note above). Cell
    # sizes in pixels are deliberately incommensurate with each other and with
    # collision_grid_resolution's ~5 m period, so no two layers - and nothing here and
    # the terrain mesh's own cell structure - can beat together into a regular pattern:
    #   fine_grain   ~0.2 m  - sand/regolith GRAIN, visible at arm's length
    #   mid_detail   ~1.5 m  - boot/wheel-scale clumping
    #   broad_ripple ~4.4 m  - gentle rolling undulation, breaks up any remaining
    #                          flatness at a scale close to but not aligned with the
    #                          collision cell size
    # plus _small_pits (dense, small, at unique real positions - already non-periodic)
    # and ejecta/skirts so those albedo features carry real relief, not flat paint.
    # Weights and strength are deliberately LOW-CONTRAST: an earlier version at much
    # higher weight/strength read as a dense, perfectly regular crosshatch once the
    # tiling bug above was the dominant visible feature - even with tiling gone, high
    # contrast at a single dominant frequency can still read as "pattern" rather than
    # "texture". This version favours several weaker, differently-scaled layers over one
    # strong one.
    fine_grain = value_noise_2d((n_tex, n_tex), max(2.0, 0.20 * px_per_m), rng)
    mid_detail = value_noise_2d((n_tex, n_tex), 1.5 * px_per_m, rng)
    broad_ripple = value_noise_2d((n_tex, n_tex), 4.4 * px_per_m, rng)
    pits = _small_pits(n_tex, px_per_m, count=9000, rng=rng)

    detail_height = (
        fine_grain * 0.16
        + mid_detail * 0.14
        + broad_ripple * 0.10
        + pits * 0.30
        + ejecta * 0.22
        + skirts * 0.04
    )
    ny_g, nx_g = np.gradient(detail_height)
    strength = 0.75
    nx, ny = -nx_g * strength, -ny_g * strength
    nz = np.ones_like(nx)
    length = np.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny, nz = nx / length, ny / length, nz / length
    normal = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1)

    # --- Roughness --------------------------------------------------------------
    rough_variation = _normalize01(value_noise_2d((n_tex, n_tex), n_tex / 16.0, rng))
    roughness = 0.80 + 0.10 * rough_variation + 0.06 * slope_norm - 0.10 * ejecta
    roughness = np.clip(roughness, 0.55, 0.97)

    output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    paths = {}
    for name, arr in (
        ("albedo", _to_uint8(albedo)),
        ("normal", _to_uint8(normal)),
        ("roughness", _to_uint8(np.stack([roughness] * 3, axis=-1))),
    ):
        path = output_dir / f"{name}.png"
        Image.fromarray(arr, mode="RGB").save(path)
        paths[name] = path
    return paths
