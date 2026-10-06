# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Deliberate degradation of the a-priori terrain map, shared by every consumer of it.

Lives here, not in `terrain_relative_node.py`, because the terrain matcher is no
longer the only thing that reads the map: the planner's costmap can be built from
the same degraded DEM (docs/TRL4_BASELINE.md, gap G9 - the costmap used to be a
perfect hazard oracle). Both must degrade through ONE function with the same seed
and the same shift convention, otherwise the rover would localise against one map
and plan on another, and "the rover's a-priori map" would stop being a single thing.
"""

import math


def shift_vector_m(shift_m, shift_bearing_deg=None):
    """Return the registration offset (dx_m, dy_m) the map is displaced by.

    `shift_bearing_deg` None is the legacy behaviour, kept so every recorded run
    still means what it did: shift_m on BOTH axes, a fixed diagonal of magnitude
    shift_m * sqrt(2). Given a bearing, shift_m is the MAGNITUDE and the offset
    points along it, 0 deg = +x, 90 deg = +y, counter-clockwise. The diagonal
    meant registration direction was never a variable; it has to be one to ask
    whether the error's direction matters relative to the route.
    """
    if shift_bearing_deg is None:
        return float(shift_m), float(shift_m)
    theta = math.radians(shift_bearing_deg)
    return shift_m * math.cos(theta), shift_m * math.sin(theta)


def bearing_from_param(value):
    """ROS parameter value -> bearing in degrees, or None. NaN means "not set"."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def degrade_dem(dem, resolution_m, post_m=0.0, noise_m=0.0, shift_m=0.0, noise_seed=0,
                prefilter_m=0.0, shift_bearing_deg=None):
    """Return a deliberately worsened copy of the a-priori DEM. Identity when unset.

    The shipped DEM is the terrain generator's own heightmap, read exactly - a
    map no mission has. This is the knob that takes that idealisation away, and
    it exists so the live stack can be run on a map with a mission's defects
    rather than only reasoned about.

    It is the SAME computation `scripts/terrain_relative_dem_quality.py` replays
    offline, and deliberately shared with it rather than reimplemented: that
    script's published table is the prediction this parameter lets us test live,
    and two copies of a degradation model drift apart silently.

    Three defects, applied in this order because that is the order a real
    pipeline acquires them:

      post_m   posts resampled coarser and back onto the same grid. Removes fine
               relief - most of what the matcher reads. Information LOST.
      noise_m  elevation error, spatially correlated at the post scale, RENORMALISED
               to this rms after smoothing. The renormalisation is not a detail:
               without it a smoothing kernel several cells wide cuts the drawn
               sigma by more than an order of magnitude (0.15 m requested came
               out as 0.008 m realised on a 0.39 m-post map), so the arm labelled
               "0.1 m of elevation noise" was in truth testing coarsening plus a
               trace - which is exactly why its replayed score was identical to
               the coarsening-only arm. Correlated error is the right MODEL for a
               DEM; its amplitude still has to be the amplitude claimed. The node
               logs the realised rms either way, so the claim stays checkable.
      shift_m  the map is internally correct but sits in the wrong place. A pure
               BIAS, and the one the replay says is fatal: the matcher faithfully
               reports where the rover is on a map that is itself displaced, at
               full confidence, because nothing about the cost surface is wrong.
               With `shift_bearing_deg` None this is the legacy fixed diagonal
               (shift_m on BOTH axes, magnitude shift_m * sqrt(2)); with a bearing,
               shift_m is the magnitude and the offset points along it (see
               `shift_vector_m`). A feature at (x, y) on the true map appears at
               (x + dx, y + dy) on the degraded one.

    `prefilter_m` is NOT a defect. It is the mitigation the three above imply, and
    it belongs to the rover rather than to the map: smooth the DEM you were given
    before differentiating it. Since slope error goes as sigma/L, lengthening L
    costs signal linearly and suppresses noise-induced slope error by the same
    factor - and the degradation table already shows that trade is a good one,
    because coarsening is the cheap defect and vertical error is the expensive
    one. It is applied last, on the map as received, which is the only thing a
    real rover could do.
    """
    import numpy as _np
    from scipy import ndimage

    dem = _np.asarray(dem, dtype=float)
    if post_m and post_m > resolution_m:
        factor = post_m / resolution_m
        small = ndimage.zoom(dem, 1.0 / factor, order=1)
        dem = ndimage.zoom(small, _np.array(dem.shape) / _np.array(small.shape), order=1)
    realised_noise_m = 0.0
    if noise_m:
        rng = _np.random.default_rng(noise_seed)
        field = ndimage.gaussian_filter(
            rng.normal(0.0, noise_m, dem.shape),
            sigma=max(post_m or resolution_m, resolution_m) / resolution_m,
        )
        spread = float(_np.std(field))
        if spread > 0.0:
            field *= noise_m / spread
        realised_noise_m = float(_np.std(field))
        dem = dem + field
    if shift_m:
        dx_m, dy_m = shift_vector_m(shift_m, shift_bearing_deg)
        # Array axes are [row = y, col = x], hence (dy, dx).
        dem = ndimage.shift(dem, (dy_m / resolution_m, dx_m / resolution_m), order=1,
                            mode="nearest")
    if prefilter_m:
        dem = ndimage.gaussian_filter(dem, sigma=prefilter_m / resolution_m)
    return dem, realised_noise_m
