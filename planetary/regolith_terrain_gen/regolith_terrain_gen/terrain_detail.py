# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Build the surface that is actually DRAWN, as a bounded refinement of the frozen
collision/elevation surface - never a replacement for it.

WHY THIS EXISTS
----------------
``heightmap._synthesize_visual_heightmap`` (what ships today) evaluates the collision
boxes' own per-cell tangent PLANE at every pixel: within one collision_grid_resolution
cell (5 m) it is an exact, perfectly flat plane, and two neighbouring cells generally
have different gradients, so the surface has a slope DISCONTINUITY at every cell
boundary. From above this reads as a tiled floor - a quilt of flat facets with creases
between them - and it is the single biggest thing keeping renders from looking like real
ground. See PROGRESS.md's terrain-realism-pass note and the task brief for the measured
"5 m square facets" defect.

The fix has to leave the surface that elevation_lookup, the collision boxes, and every
rock's seated position promise COMPLETELY ALONE - see config.py's "physics is frozen"
note and PROGRESS.md - because every navigation number on record was measured against
today's collision world, and rocks are already seated (in the shipped manifest) against
today's elevation_lookup. So this module never touches build_heightmap's return values;
it only computes a SEPARATE surface for terrain_mesh.save_terrain_mesh_obj to export,
built by refining the same coarse control grid the collision boxes come from, clamped so
it can never drift more than cfg.visual_surface_budget_m away from the surface rocks are
actually seated against.

WHAT IT DOES
------------
1. MACRO: replace the piecewise-PLANAR per-cell reconstruction with a bicubic spline
   through the identical coarse, already-3-pass-smoothed cell-height grid
   (``_build_smoothed_surface``'s ``surface``). A spline through the same control points
   is C2 continuous - no creases at cell boundaries, and no dead-flat interior within a
   cell either - while still tracking the same macro shape (it interpolates the same
   values). Clamped to +/- cfg.visual_macro_budget_m against the old surface.
2. FINE: recover a fraction of the sub-cell relief that block-averaging discarded.
   raw_heightmap minus its own per-cell average is exactly that discarded detail - real
   fBm/crater structure at native (0.39 m) resolution, not synthetic noise - scaled down
   by cfg.visual_fine_residual_frac and folded in. This is what breaks up the
   still-rather-smooth spline surface into something with actual boot-scale texture.
3. Both deltas are added and the TOTAL is clamped to
   +/- cfg.visual_surface_budget_m, which is the number this module actually promises
   and the one the rock-seating gap should be measured against.

Returns (drawn_surface, total_delta) - both [row=y, col=x] arrays at raw_heightmap's
full resolution, in the same absolute-metre convention as everything else in this
package. total_delta is returned so callers can report the real measured deviation
rather than assuming the configured budget was reached.
"""

import numpy as np
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.heightmap import _build_smoothed_surface


def _bicubic_spline_reconstruction(raw_heightmap: np.ndarray, cfg: TerrainConfig, grid: dict) -> np.ndarray:
    """Evaluate a C2 bicubic spline through the coarse control grid at full resolution.

    Falls back to plain bilinear (still C0-continuous in VALUE, still much smoother than
    the tangent-plane reconstruction which is only C0 in value and discontinuous in
    slope) if scipy's spline ever rejects the grid (e.g. too few control points for a
    degenerate tiny world) - this module must never crash generation over an unavailable
    smoothness upgrade.
    """
    xs, ys, surface = grid["xs"], grid["ys"], grid["surface"]
    n = raw_heightmap.shape[0]
    per_m = (n - 1) / cfg.world_size_m
    half = cfg.world_size_m / 2.0
    fine_coords = -half + np.arange(n) / per_m
    fine_coords = np.clip(fine_coords, xs[0], xs[-1])  # spline domain is [xs[0], xs[-1]]

    try:
        from scipy.interpolate import RectBivariateSpline

        k = min(3, len(xs) - 1, len(ys) - 1)
        spline = RectBivariateSpline(ys, xs, surface, kx=max(1, k), ky=max(1, k))
        return spline(fine_coords, fine_coords)
    except Exception:
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator((ys, xs), surface, method="linear", bounds_error=False, fill_value=None)
        gy, gx = np.meshgrid(fine_coords, fine_coords, indexing="ij")
        return interp((gy, gx))


def _fine_residual(raw_heightmap: np.ndarray, grid: dict) -> np.ndarray:
    """raw_heightmap minus its own per-cell block average, at full resolution - the
    sub-cell relief _build_smoothed_surface's averaging step discards, recovered as an
    array the same shape as raw_heightmap so it can be added directly."""
    block = grid["block"]
    usable = grid["usable"]
    n = raw_heightmap.shape[0]
    trimmed = raw_heightmap[:usable, :usable]
    rb, cb = usable // block, usable // block
    block_avg = trimmed.reshape(rb, block, cb, block).mean(axis=(1, 3))
    upsampled_avg = np.repeat(np.repeat(block_avg, block, axis=0), block, axis=1)

    residual = np.zeros_like(raw_heightmap)
    residual[:usable, :usable] = trimmed - upsampled_avg
    if usable < n:
        # Edge strip beyond the crop (see _build_smoothed_surface): reuse the nearest
        # valid row/column of residual rather than leaving it exactly zero, so the strip
        # doesn't read as a hard-edged flat border.
        residual[usable:, :] = residual[usable - 1 : usable, :]
        residual[:, usable:] = residual[:, usable - 1 : usable]
    return residual


def build_drawn_surface(
    raw_heightmap: np.ndarray, visual_heightmap: np.ndarray, cfg: TerrainConfig, rng
) -> tuple:
    """Return (drawn_surface, total_delta), both [row=y, col=x], full resolution.

    `rng` must be an INDEPENDENT generator (not the shared one build_heightmap/
    generate_rock_variants/scatter_rocks consume) - see generate.py. Currently unused
    (the fine layer is sourced entirely from real heightmap structure, not noise) but
    kept in the signature so a future noise-based detail layer has nowhere else to
    reach for randomness than here.

    BUDGET SPLIT, and why it is NOT macro-gets-a-slice/fine-gets-the-rest: an earlier
    version pre-clipped macro_delta to a small sub-budget, scaled the fine residual by a
    FRACTION of its own (much larger) natural amplitude, summed the two, and clipped the
    sum to the total budget. Because the raw fine residual's typical magnitude (mean
    0.19 m - see config.py's visual_fine_detail_amplitude_m note) is far bigger than the
    whole budget even after that fraction, the final clip ended up saturated by the fine
    term's SIGN almost everywhere, and the macro correction - the only one of the two
    that is structurally tied to the cell-boundary creases making the "quilt" - was
    crowded out to a few millimetres' effective contribution. The drawn surface stayed
    overwhelmingly the OLD piecewise-planar one and rendered that way.

    This version gives the macro term nearly the whole budget UNCLIPPED-then-capped (not
    pre-shrunk), and caps the fine layer to a small ABSOLUTE amplitude well under the
    macro budget, so summing the two and clipping once to the total budget lets both
    through rather than letting one drown the other.
    """
    del rng  # reserved; see docstring

    grid = _build_smoothed_surface(raw_heightmap, cfg)
    spline_surface = _bicubic_spline_reconstruction(raw_heightmap, cfg, grid)
    macro_delta = np.clip(
        spline_surface - visual_heightmap, -cfg.visual_macro_budget_m, cfg.visual_macro_budget_m
    )

    fine_delta = np.clip(
        _fine_residual(raw_heightmap, grid),
        -cfg.visual_fine_detail_amplitude_m,
        cfg.visual_fine_detail_amplitude_m,
    )

    total_delta = np.clip(
        macro_delta + fine_delta, -cfg.visual_surface_budget_m, cfg.visual_surface_budget_m
    )
    drawn_surface = visual_heightmap + total_delta
    return drawn_surface, total_delta
