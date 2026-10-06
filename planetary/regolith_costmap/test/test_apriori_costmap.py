# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""The shared a-priori-map degradation, and the costmap that is built from it.

Two things have to hold. `degrade_dem` moved out of terrain_relative_node.py and
must still compute exactly what it did (every recorded degraded-map run depends on
that), and the costmap's rock layer must move with the DEM: a registration shift
that displaced the terrain but left the rocks where they were would hand the planner
a map that is inconsistent with itself (docs/TRL4_BASELINE.md, gap G9).
"""

import math

import numpy as np
import pytest
from regolith_costmap.costmap_node import build_apriori_costmap
from regolith_costmap.costmap_node import build_costmap
from regolith_costmap.dem_degradation import bearing_from_param
from regolith_costmap.dem_degradation import degrade_dem
from regolith_costmap.dem_degradation import shift_vector_m

# Verbatim copy of the function as it was in terrain_relative_node.py before the move.
def _legacy_degrade_dem(dem, resolution_m, post_m=0.0, noise_m=0.0, shift_m=0.0, noise_seed=0,
                prefilter_m=0.0):
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
        offset = shift_m / resolution_m
        dem = ndimage.shift(dem, (offset, offset), order=1, mode="nearest")
    if prefilter_m:
        dem = ndimage.gaussian_filter(dem, sigma=prefilter_m / resolution_m)
    return dem, realised_noise_m


@pytest.mark.parametrize("kwargs", [
    {},
    {"post_m": 4.0},
    {"noise_m": 0.2, "noise_seed": 7},
    {"shift_m": 2.0},
    {"shift_m": -1.5},
    {"prefilter_m": 2.0},
    {"post_m": 3.0, "noise_m": 0.3, "shift_m": 1.0, "noise_seed": 3, "prefilter_m": 1.0},
])
def test_unchanged_for_existing_arguments(kwargs):
    rng = np.random.default_rng(11)
    dem = rng.normal(0.0, 1.0, (65, 65)).cumsum(axis=0).cumsum(axis=1) * 0.01
    want, want_noise = _legacy_degrade_dem(dem, 0.5, **kwargs)
    got, got_noise = degrade_dem(dem, 0.5, **kwargs)
    assert np.array_equal(got, want)
    assert got_noise == want_noise


def test_bearing_none_is_the_legacy_diagonal():
    assert shift_vector_m(3.0, None) == (3.0, 3.0)


@pytest.mark.parametrize("bearing, expected", [
    (0.0, (2.0, 0.0)), (90.0, (0.0, 2.0)), (180.0, (-2.0, 0.0)), (-90.0, (0.0, -2.0)),
])
def test_bearing_convention_is_x_then_ccw(bearing, expected):
    dx, dy = shift_vector_m(2.0, bearing)
    assert dx == pytest.approx(expected[0], abs=1e-12)
    assert dy == pytest.approx(expected[1], abs=1e-12)


def test_bearing_shift_magnitude_is_shift_m():
    dx, dy = shift_vector_m(5.0, 37.0)
    assert math.hypot(dx, dy) == pytest.approx(5.0)


def test_bearing_param_nan_means_unset():
    assert bearing_from_param(float("nan")) is None
    assert bearing_from_param(None) is None
    assert bearing_from_param(45) == 45.0


def _peak(n=65, row=30, col=20):
    dem = np.zeros((n, n))
    dem[row, col] = 1.0
    return dem


def test_degrade_bearing_moves_a_feature_along_the_bearing():
    dem = _peak()
    out, _ = degrade_dem(dem, 1.0, shift_m=5.0, shift_bearing_deg=0.0)
    assert np.unravel_index(np.argmax(out), out.shape) == (30, 25)  # +x = +col
    out, _ = degrade_dem(dem, 1.0, shift_m=5.0, shift_bearing_deg=90.0)
    assert np.unravel_index(np.argmax(out), out.shape) == (35, 20)  # +y = +row


def test_degrade_bearing_none_equals_legacy_diagonal():
    dem = _peak()
    a, _ = degrade_dem(dem, 1.0, shift_m=3.0)
    b, _ = degrade_dem(dem, 1.0, shift_m=3.0, shift_bearing_deg=None)
    assert np.array_equal(a, b)
    assert np.unravel_index(np.argmax(a), a.shape) == (33, 23)


# ---- costmap ----------------------------------------------------------------

WORLD_M = 64.0


def _manifest(rocks=()):
    return {"world_size_m": WORLD_M, "rocks": list(rocks)}


def _rock(x, y, radii=(0.5, 0.4), scale=1.0):
    rock = {"x_m": x, "y_m": y, "scale_m": scale}
    if radii is not None:
        rock["collision_radii_m"] = list(radii)
    return rock


def _bump(x_m, y_m, n=65):
    """A steep, narrow bump (lethal slope ring) centred on world (x_m, y_m); 1 m posts."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    cx, cy = x_m + WORLD_M / 2.0, y_m + WORLD_M / 2.0
    return 12.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 1.5**2))


def _centroid(lethal):
    rows, cols = np.nonzero(lethal)
    return rows.mean(), cols.mean()


def _lethal(manifest, heightmap, **kwargs):
    grid, *_ = build_apriori_costmap(manifest, heightmap, 1.0, 0.3, 20.0, **kwargs)
    return np.asarray(grid) >= 100


def test_defaults_give_the_identical_costmap():
    rng = np.random.default_rng(5)
    heightmap = rng.normal(0.0, 1.0, (65, 65)).cumsum(axis=0).cumsum(axis=1) * 0.05
    rocks = [_rock(-10.0, 5.0), _rock(12.3, -7.7, radii=None, scale=2.0), _rock(0.0, 0.0)]
    manifest = _manifest(rocks)
    want = build_costmap(manifest, heightmap, 1.0, 0.3, 20.0)
    got = build_apriori_costmap(manifest, heightmap, 1.0, 0.3, 20.0)
    for a, b in zip(want, got):
        assert np.array_equal(a, b)
    assert np.array_equal(want[0], build_costmap(
        manifest, heightmap, 1.0, 0.3, 20.0, rock_shift_xy_m=(0.0, 0.0),
        rock_min_detect_diameter_m=0.0)[0])


@pytest.mark.parametrize("shift_m, bearing", [
    (6.0, 0.0), (6.0, 90.0), (6.0, 210.0), (4.0, None), (5.0, 33.0),
])
def test_rock_and_bump_land_in_the_same_cell_after_a_shift(shift_m, bearing):
    # Same true point for both; one is stored as terrain, the other as a rock. Shifting
    # the map must carry both to the same costmap cell, or the registration error is
    # applied with opposite signs to the two layers.
    px, py = -5.0, 8.0
    terrain_only = _lethal(
        _manifest(), _bump(px, py), dem_shift_m=shift_m, dem_shift_bearing_deg=bearing)
    rock_only = _lethal(
        _manifest([_rock(px, py, radii=(0.5, 0.5))]), np.zeros((65, 65)),
        dem_shift_m=shift_m, dem_shift_bearing_deg=bearing)
    t_row, t_col = _centroid(terrain_only)
    r_row, r_col = _centroid(rock_only)
    assert abs(t_row - r_row) <= 1.0
    assert abs(t_col - r_col) <= 1.0
    # and the shift really happened, in the expected direction
    dx, dy = shift_vector_m(shift_m, bearing)
    unshifted = _centroid(_lethal(_manifest([_rock(px, py, radii=(0.5, 0.5))]),
                                  np.zeros((65, 65))))
    assert r_col - unshifted[1] == pytest.approx(dx, abs=1.0)
    assert r_row - unshifted[0] == pytest.approx(dy, abs=1.0)


def test_small_rocks_are_dropped_by_footprint_diameter():
    rocks = [_rock(-10.0, 0.0, radii=(0.5, 0.2)),  # footprint 0.5 m -> diameter 1.0
             _rock(10.0, 0.0, radii=(0.25, 0.1))]  # diameter 0.5
    flat = np.zeros((65, 65))
    both = _lethal(_manifest(rocks), flat)
    only_big = _lethal(_manifest(rocks), flat, rock_min_detect_diameter_m=0.8)
    none = _lethal(_manifest(rocks), flat, rock_min_detect_diameter_m=1.5)
    assert both.sum() > only_big.sum() > none.sum() == 0
    # exactly at the threshold the rock is kept (>=)
    assert _lethal(_manifest(rocks), flat, rock_min_detect_diameter_m=1.0).sum() == only_big.sum()


def test_rock_without_radii_falls_back_to_scale():
    rock = _rock(0.0, 0.0, radii=None, scale=1.5)  # diameter 3.0
    flat = np.zeros((65, 65))
    assert _lethal(_manifest([rock]), flat, rock_min_detect_diameter_m=2.9).any()
    assert not _lethal(_manifest([rock]), flat, rock_min_detect_diameter_m=3.1).any()


def test_degraded_dem_reaches_the_slope_layer():
    manifest = _manifest()
    assert _lethal(manifest, _bump(0.0, 0.0)).any()
    assert not _lethal(manifest, _bump(0.0, 0.0), dem_post_m=16.0).any()

