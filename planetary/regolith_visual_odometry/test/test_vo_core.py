# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests the recovered motion against a scene whose answer is known exactly.

No simulator and no ROS. A textured ground plane is ray-traced from two camera
poses a known rigid motion apart, which gives geometrically exact image and
depth pairs - so any error these tests report is the estimator's, not a
renderer's or a physics engine's.

The case that matters most is test_pure_lateral_slide. That motion - the rover
sliding sideways while pointing straight ahead - is the one a differential-drive
odometry model cannot represent at all and an IMU cannot observe, and it is what
put M4's arrival error at 3.1-13.1 m. If this package cannot see that, it has no
reason to exist, and the test asserts on vy directly rather than on a norm that a
large vx could hide inside.
"""

import numpy as np
import pytest
from regolith_visual_odometry.vo_core import OPTICAL_TO_BODY
from regolith_visual_odometry.vo_core import VoConfig
from regolith_visual_odometry.vo_core import detect_features
from regolith_visual_odometry.vo_core import estimate_motion
from regolith_visual_odometry.vo_core import sample_depth
from regolith_visual_odometry.vo_core import usable_depth_mask

WIDTH, HEIGHT = 640, 480
HFOV_RAD = 1.4
CAMERA_PITCH_RAD = 0.25
CAMERA_OFFSET_M = np.array([0.20, 0.0, 0.195])
# The keyframe baseline the node actually uses. Not an arbitrary test constant:
# at 0.1 s the camera moves 2 cm, which is ~1 px of flow against 0.74 px of
# tracker noise, and the estimate is meaningless. See vo_core.py's header.
DT_S = 0.4
# Worst-case velocity error measured over 4 seeds x 4 motion types at these
# settings. The tests assert against the accuracy this estimator actually has,
# not an aspirational one.
VELOCITY_TOL_MPS = 0.06


def _intrinsics():
    focal = (WIDTH / 2.0) / np.tan(HFOV_RAD / 2.0)
    return np.array([[focal, 0.0, WIDTH / 2.0], [0.0, focal, HEIGHT / 2.0], [0.0, 0.0, 1.0]])


def _rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _random_field(seed, cells=400, extent_m=60.0):
    """A fixed, bilinearly-sampled random field over world (x, y) in metres.

    Blobby rather than white noise, because Lucas-Kanade needs structure with a
    scale to it - and so does real regolith under a low sun.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.random((cells, cells))

    def sample(x_m, y_m):
        u = (x_m + extent_m / 2.0) / extent_m * (cells - 1)
        v = (y_m + extent_m / 2.0) / extent_m * (cells - 1)
        u = np.clip(u, 0, cells - 1.001)
        v = np.clip(v, 0, cells - 1.001)
        u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
        du, dv = u - u0, v - v0
        top = coarse[v0, u0] * (1 - du) + coarse[v0, u0 + 1] * du
        bottom = coarse[v0 + 1, u0] * (1 - du) + coarse[v0 + 1, u0 + 1] * du
        return top * (1 - dv) + bottom * dv

    return sample


# The ground has relief, and it has to. A perfectly flat plane makes every
# landmark coplanar, which is a textbook degenerate configuration for PnP - the
# pose is ambiguous and the recovered translation comes out scaled wrong. On a
# plane this estimator reports 0.139 m/s for a true 0.200 m/s and invents 0.09 m/s
# of sideways drift out of a pure turn; with a few centimetres of relief both
# resolve. That is a property of the geometry, not of this implementation, and it
# is the reason the test scene is a rough surface: real regolith - craters, rocks,
# the roughness the costmap measures - is never the degenerate case, so a flat
# test plane would have been testing a situation the rover is never in.
RELIEF_AMPLITUDE_M = 0.10


def _render(base_position_m, base_yaw_rad, k_matrix, texture, relief):
    """Ray-traces a rough ground surface into a geometrically exact (gray, depth) pair.

    Depth is the optical-frame z of each intersection, which is what a depth
    camera reports; rays that escape above the horizon come back as +inf, the way
    gz writes unreturned rays.
    """
    world_from_base = _rot_z(base_yaw_rad)
    world_from_camera_body = world_from_base @ _rot_y(CAMERA_PITCH_RAD)
    world_from_optical = world_from_camera_body @ OPTICAL_TO_BODY
    camera_position = np.asarray(base_position_m, float) + world_from_base @ CAMERA_OFFSET_M

    us, vs = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    # Rays with optical z = 1, so the ray parameter IS the reported depth.
    rays_optical = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], axis=-1)
    rays_world = rays_optical @ world_from_optical.T

    def height(x_m, y_m):
        return (relief(x_m, y_m) - 0.5) * 2.0 * RELIEF_AMPLITUDE_M

    # Fixed-point intersection against the heightfield: start on the mean plane,
    # then re-solve against the surface height found there. Converges in a few
    # passes for relief this gentle.
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = -camera_position[2] / rays_world[..., 2]
        for _ in range(6):
            valid = np.isfinite(depth) & (depth > 0.0)
            points = camera_position + rays_world * np.where(valid, depth, 0.0)[..., None]
            surface = height(points[..., 0], points[..., 1])
            depth = (surface - camera_position[2]) / rays_world[..., 2]
    depth[~np.isfinite(depth) | (depth <= 0.0)] = np.inf

    hit = np.isfinite(depth)
    points = camera_position + rays_world * np.where(hit, depth, 0.0)[..., None]
    gray = np.zeros((HEIGHT, WIDTH), np.float64)
    gray[hit] = texture(points[..., 0][hit], points[..., 1][hit])
    return (gray * 255).astype(np.uint8), depth.astype(np.float32)


def _estimate(velocity_mps, yaw_rate_rps=0.0, seed=0, cfg=None):
    """Renders a keyframe/current pair for a known base_link motion and estimates it.

    Takes VELOCITY, and moves the rover by velocity * DT_S, so the assertions read
    in the units the estimator reports and stay correct if the baseline changes.
    """
    delta_position_m = np.asarray(velocity_mps, float) * DT_S
    delta_yaw_rad = yaw_rate_rps * DT_S
    k_matrix = _intrinsics()
    texture, relief = _random_field(seed), _random_field(seed + 100, cells=120)
    prev_gray, prev_depth = _render(np.zeros(3), 0.0, k_matrix, texture, relief)
    cur_gray, _ = _render(delta_position_m, delta_yaw_rad, k_matrix, texture, relief)
    return estimate_motion(
        prev_gray,
        prev_depth,
        cur_gray,
        k_matrix,
        DT_S,
        _rot_y(CAMERA_PITCH_RAD),
        CAMERA_OFFSET_M,
        cfg or VoConfig(),
    )


def test_pure_forward_motion():
    """0.2 m/s straight ahead: vx recovered, and no phantom sideways velocity."""
    estimate = _estimate([0.2, 0.0, 0.0])

    assert estimate.valid, estimate.reason
    assert estimate.linear_mps[0] == pytest.approx(0.2, abs=VELOCITY_TOL_MPS)
    assert estimate.linear_mps[1] == pytest.approx(0.0, abs=VELOCITY_TOL_MPS)
    assert estimate.angular_rps[2] == pytest.approx(0.0, abs=0.05)


def test_pure_lateral_slide():
    """The motion this whole package exists for: sideways, with the heading fixed.

    Wheel odometry reports this as zero - a differential-drive model has no vy
    term - and the IMU sees no rotation, so nothing in the previous sensor suite
    contradicted "we went nowhere sideways". VO has to report it.
    """
    estimate = _estimate([0.0, 0.1, 0.0])

    assert estimate.valid, estimate.reason
    assert estimate.linear_mps[1] == pytest.approx(0.1, abs=VELOCITY_TOL_MPS)
    # And the direction must be right - a sign error here would drive the EKF's
    # correction the wrong way, which is worse than not measuring it at all.
    assert estimate.linear_mps[1] > 0.0
    assert estimate.linear_mps[0] == pytest.approx(0.0, abs=VELOCITY_TOL_MPS)


def test_lateral_slide_the_other_way():
    """Same magnitude, opposite sign - pins the axis convention, not just the axis."""
    estimate = _estimate([0.0, -0.1, 0.0])

    assert estimate.valid, estimate.reason
    assert estimate.linear_mps[1] == pytest.approx(-0.1, abs=VELOCITY_TOL_MPS)


def _mean_lateral(lateral_mps, seeds=12):
    """Mean recovered vy over independent scenes, all with the same true motion."""
    return np.mean([_estimate([0.2, lateral_mps, 0.0], seed=s).linear_mps[1] for s in range(seeds)])


def test_slipping_while_driving_is_resolved_by_averaging_not_by_one_estimate():
    """The realistic case: mostly forward, ~10% of the motion sideways.

    That ratio is the measured one from the M4 error budget (8.7-11.1% of all
    motion across the three acceptance seeds), so it is the case the milestone
    turns on - and it is deliberately NOT asserted on a single estimate, because
    a single estimate genuinely cannot see it. One estimate's lateral noise is
    sd 0.011-0.026 m/s against a slip signal of 0.020 m/s, so an assertion like
    `approx(0.02, abs=0.06)` would pass just as happily on a dead vy channel.

    What makes this work is that the noise is unbiased and the EKF integrates
    thousands of these. Averaged over independent scenes the slip is recovered to
    better than a millimetre per second, and a no-slip scene reads as zero - so
    this asserts the two are distinguishable, which is the property the fusion
    actually depends on.
    """
    without_slip = _mean_lateral(0.0)
    with_slip = _mean_lateral(0.02)

    assert without_slip == pytest.approx(0.0, abs=0.010)
    assert with_slip == pytest.approx(0.02, abs=0.010)
    # And they are separated - the assertion a dead or sign-flipped vy channel fails.
    assert with_slip - without_slip > 0.010


def test_forward_speed_reads_slightly_low_and_that_is_recorded():
    """Pins a real, measured bias rather than leaving it to be rediscovered.

    Averaged over 12 scenes the estimator recovers 0.183-0.191 m/s for a true
    0.200 - about 5% low. The cause is the nearest-pixel depth lookup on a
    steeply oblique ground plane, where a sub-pixel feature position and the
    depth sampled for it disagree slightly, biasing reconstructed landmarks. It
    is small, it is toward under-reporting distance (the safe direction for
    arrival), and the EKF also has wheel odometry's vx to weigh against it - so
    it is documented rather than chased. This test exists so that if it ever
    grows, something says so.
    """
    forward = np.mean([_estimate([0.2, 0.0, 0.0], seed=s).linear_mps[0] for s in range(12)])

    assert forward == pytest.approx(0.19, abs=0.02)
    assert forward < 0.2


def test_pure_yaw_does_not_read_as_translation():
    """Turning in place must not manufacture velocity at base_link.

    The camera is 0.2 m ahead of base_link, so yawing genuinely translates the
    CAMERA. If the lever arm were dropped, an ordinary turn would inject a
    fictitious sideways velocity into the filter on every single turn - which
    would be worse than the drift this package is meant to remove.
    """
    yaw_rate = 0.2
    estimate = _estimate([0.0, 0.0, 0.0], yaw_rate_rps=yaw_rate)

    assert estimate.valid, estimate.reason
    assert estimate.angular_rps[2] == pytest.approx(yaw_rate, abs=0.05)
    assert estimate.linear_mps[0] == pytest.approx(0.0, abs=VELOCITY_TOL_MPS)
    assert estimate.linear_mps[1] == pytest.approx(0.0, abs=VELOCITY_TOL_MPS)


def test_featureless_scene_is_refused_not_guessed():
    """Uniform ground yields no corners: the estimate must be invalid, not zero.

    Publishing a confident zero here would be a zero-velocity update every time
    the camera faced a smooth patch, silently telling the EKF the rover had
    stopped while it was driving.
    """
    k_matrix = _intrinsics()
    flat_gray = np.full((HEIGHT, WIDTH), 128, np.uint8)
    depth = np.full((HEIGHT, WIDTH), 3.0, np.float32)

    estimate = estimate_motion(
        flat_gray,
        depth,
        flat_gray,
        k_matrix,
        DT_S,
        _rot_y(CAMERA_PITCH_RAD),
        CAMERA_OFFSET_M,
        VoConfig(),
    )

    assert not estimate.valid
    assert estimate.velocity_sigma_mps == float("inf")


def test_missing_depth_is_refused():
    """Plenty of texture but no range anywhere: no metric motion is possible.

    This surfaces as "too few tracked features" rather than a depth complaint,
    and that is the depth mask doing its job - the detector is never allowed to
    look at pixels without range, so a frame with no range has nowhere to search.
    Asserted on the outcome rather than the wording, because the wording is an
    implementation detail and the refusal is not.
    """
    k_matrix = _intrinsics()
    texture, relief = _random_field(0), _random_field(100, cells=120)
    gray, _ = _render(np.zeros(3), 0.0, k_matrix, texture, relief)
    no_depth = np.full((HEIGHT, WIDTH), np.inf, np.float32)

    estimate = estimate_motion(
        gray,
        no_depth,
        gray,
        k_matrix,
        DT_S,
        _rot_y(CAMERA_PITCH_RAD),
        CAMERA_OFFSET_M,
        VoConfig(),
    )

    assert not estimate.valid
    assert estimate.n_inliers == 0
    assert estimate.velocity_sigma_mps == float("inf")


def test_features_are_only_taken_where_depth_exists():
    """The masking rule directly: the sky must never contribute a feature.

    This is the defect that made the first live run refuse every single frame
    pair. The strongest corners in a lunar scene are on the skyline, where
    sunlit terrain meets a near-black sky, and the sky has no range - so an
    unmasked detector spent the whole budget on unusable points (25 corners
    found, 8 with depth, against an inlier floor of 20).
    """
    k_matrix = _intrinsics()
    texture, relief = _random_field(0), _random_field(100, cells=120)
    gray, depth = _render(np.zeros(3), 0.0, k_matrix, texture, relief)

    # This scene genuinely has sky: rays above the horizon return no range.
    assert not np.isfinite(depth).all(), "test scene must contain range-less sky"

    corners = detect_features(gray, VoConfig(), usable_depth_mask(depth, VoConfig()))
    sampled = sample_depth(depth, corners)

    assert len(corners) > VoConfig.min_inliers
    assert np.isfinite(sampled).all()


def test_reported_sigma_is_finite_and_sane_on_a_good_estimate():
    """The covariance the EKF will weight this by must be a real, small number."""
    estimate = _estimate([0.2, 0.0, 0.0])

    assert estimate.valid, estimate.reason
    assert np.isfinite(estimate.velocity_sigma_mps)
    assert 0.0 < estimate.velocity_sigma_mps < 1.0
    assert estimate.n_inliers >= VoConfig.min_inliers


def test_a_zero_interval_is_not_a_measurement():
    """dt = 0 would divide displacement by nothing; refuse rather than emit inf."""
    k_matrix = _intrinsics()
    texture, relief = _random_field(0), _random_field(100, cells=120)
    gray, depth = _render(np.zeros(3), 0.0, k_matrix, texture, relief)

    estimate = estimate_motion(
        gray,
        depth,
        gray,
        k_matrix,
        0.0,
        _rot_y(CAMERA_PITCH_RAD),
        CAMERA_OFFSET_M,
        VoConfig(),
    )

    assert not estimate.valid


def test_a_frame_with_no_range_anywhere_says_so_specifically():
    """ "No depth here" and "no texture here" are different problems.

    They used to produce the same message, and that cost real time: an M4
    acceptance seed refused 2759 consecutive frame pairs for "too few tracked
    features", which reads as a barren landscape but is equally consistent with a
    camera pressed against a boulder seeing nothing but out-of-range near ground.
    Telling them apart is the difference between tuning the detector and tuning
    the depth band.
    """
    k_matrix = _intrinsics()
    texture, relief = _random_field(0), _random_field(100, cells=120)
    gray, _ = _render(np.zeros(3), 0.0, k_matrix, texture, relief)
    everything_too_close = np.full((HEIGHT, WIDTH), 0.02, np.float32)

    estimate = estimate_motion(
        gray,
        everything_too_close,
        gray,
        k_matrix,
        DT_S,
        _rot_y(CAMERA_PITCH_RAD),
        CAMERA_OFFSET_M,
        VoConfig(),
    )

    assert not estimate.valid
    assert "no usable depth" in estimate.reason
    assert estimate.mask_fraction < 0.01


def test_a_physically_impossible_speed_is_refused():
    """The rover cruises at 0.2 m/s; a solver artifact reporting metres per second
    is not a measurement of it.

    The reprojection gate catches most bad solves but not all - the catastrophic
    cases measured on real frames ran to 46 m/s. A sensor that knows what vehicle
    it is bolted to can refuse the impossible outright, so this pins that it does.
    Driven through the real code path by making the frame interval implausibly
    short, which inflates any recovered displacement into a huge velocity.
    """
    k_matrix = _intrinsics()
    texture, relief = _random_field(0), _random_field(100, cells=120)
    prev_gray, prev_depth = _render(np.zeros(3), 0.0, k_matrix, texture, relief)
    cur_gray, _ = _render(np.array([0.08, 0.0, 0.0]), 0.0, k_matrix, texture, relief)

    plausible = estimate_motion(
        prev_gray,
        prev_depth,
        cur_gray,
        k_matrix,
        0.4,
        _rot_y(CAMERA_PITCH_RAD),
        CAMERA_OFFSET_M,
        VoConfig(),
    )
    absurd = estimate_motion(
        prev_gray,
        prev_depth,
        cur_gray,
        k_matrix,
        0.004,
        _rot_y(CAMERA_PITCH_RAD),
        CAMERA_OFFSET_M,
        VoConfig(),
    )

    assert plausible.valid, plausible.reason
    assert not absurd.valid
    assert "impossible" in absurd.reason
