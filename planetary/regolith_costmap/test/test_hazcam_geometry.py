# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""The hazcam detector on depth images rendered analytically from known scenes.

Each scene is ray-cast exactly (ground plane, box, hemisphere) through a camera
with the rover's real intrinsics and mounting, so the expected answer is known
to the centimetre. The mounting comes from regolith_rover.urdf.xacro:
  * base_link sits 0.055 m above the ground (wheel radius 0.09, wheel centre
    0.035 above base_link);
  * camera_link is at (chassis_length / 2, 0, ground_clearance + chassis_height / 2
    + 0.03) = (0.20, 0, 0.195) in base_link, pitched 0.25 rad down;
  * 640x480, hfov 1.4 rad, so fx = fy = 320 / tan(0.7) = 380.9 px.
"""

import math
import time

import numpy as np
import pytest

from regolith_costmap.hazcam import back_project
from regolith_costmap.hazcam import depth_image_to_array
from regolith_costmap.hazcam import detect_obstacles
from regolith_costmap.hazcam import HazcamConfig
from regolith_costmap.hazcam import level_rotation
from regolith_costmap.hazcam import level_to_odom
from regolith_costmap.hazcam import OPTICAL_TO_BODY
from regolith_costmap.hazcam import optical_to_level
from regolith_costmap.hazcam import roll_pitch_yaw

WIDTH, HEIGHT = 640, 480
FX = 320.0 / math.tan(0.7)
K = np.array([[FX, 0.0, 320.0], [0.0, FX, 240.0], [0.0, 0.0, 1.0]])
BASE_ABOVE_GROUND_M = 0.055
MOUNT_PITCH = 0.25
MOUNT_ROTATION = np.array(
    [
        [math.cos(MOUNT_PITCH), 0.0, math.sin(MOUNT_PITCH)],
        [0.0, 1.0, 0.0],
        [-math.sin(MOUNT_PITCH), 0.0, math.cos(MOUNT_PITCH)],
    ]
)
MOUNT_OFFSET = np.array([0.20, 0.0, 0.195])
COSTMAP_RES_M = 1.0


def _yaw_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def render(base_xyz, roll, pitch, yaw, ground_slope_xy=(0.0, 0.0), boxes=(), domes=()):
    """Planar depth image of: ground z = sx * x + sy * y, axis-aligned boxes
    ((xmin, ymin, xmax, ymax, height) standing on z = 0 ground), and hemispheres
    ((cx, cy, radius) centred on z = 0). Rays that hit nothing are +inf, as gz's sky."""
    r_world_base = _yaw_matrix(yaw) @ level_rotation(roll, pitch)
    r_world_optical = r_world_base @ MOUNT_ROTATION @ OPTICAL_TO_BODY
    origin = np.asarray(base_xyz, dtype=float) + r_world_base @ MOUNT_OFFSET

    v, u = np.mgrid[0:HEIGHT, 0:WIDTH].astype(float)
    rays_opt = np.stack([(u - K[0, 2]) / FX, (v - K[1, 2]) / FX, np.ones_like(u)], axis=-1)
    d = rays_opt @ r_world_optical.T  # world direction, scaled so t == planar depth
    best = np.full(u.shape, np.inf)

    sx, sy = ground_slope_xy
    # Ground: (o + t d).z = sx (o + t d).x + sy (o + t d).y
    denom = d[..., 2] - sx * d[..., 0] - sy * d[..., 1]
    numer = sx * origin[0] + sy * origin[1] - origin[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = numer / denom
    best = np.where((t > 0) & np.isfinite(t), np.minimum(best, t), best)

    for xmin, ymin, xmax, ymax, height in boxes:
        lo = np.array([xmin, ymin, -1.0])
        hi = np.array([xmax, ymax, height])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - origin) / d
            t2 = (hi - origin) / d
        t_near = np.nanmax(np.minimum(t1, t2), axis=-1)
        t_far = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (t_near <= t_far) & (t_far > 0)
        best = np.where(hit, np.minimum(best, np.maximum(t_near, 0.0)), best)

    for cx, cy, radius in domes:
        oc = origin - np.array([cx, cy, 0.0])
        a = np.einsum("...i,...i", d, d)
        b = 2.0 * np.einsum("...i,i", d, oc)
        c = oc @ oc - radius * radius
        disc = b * b - 4 * a * c
        with np.errstate(invalid="ignore"):
            t = (-b - np.sqrt(disc)) / (2 * a)
        z_hit = origin[2] + t * d[..., 2]
        hit = (disc >= 0) & (t > 0) & (z_hit >= 0.0)
        best = np.where(hit, np.minimum(best, t), best)
    return best.astype(np.float32)


def flat_pose(x=0.0, y=0.0, yaw=0.0):
    return (x, y, BASE_ABOVE_GROUND_M), 0.0, 0.0, yaw


def detect(depth, roll, pitch, cfg=HazcamConfig()):
    return detect_obstacles(depth, K, MOUNT_ROTATION, MOUNT_OFFSET, roll, pitch, cfg)


def rover_frame_to_world(base_xy, yaw, forward_m, left_m):
    c, s = math.cos(yaw), math.sin(yaw)
    return base_xy[0] + c * forward_m - s * left_m, base_xy[1] + s * forward_m + c * left_m


def costmap_cell(xy):
    return int(math.floor(xy[0] / COSTMAP_RES_M)), int(math.floor(xy[1] / COSTMAP_RES_M))


# ---------------------------------------------------------------- conventions


def test_flat_ground_back_projects_to_the_ground():
    """The renderer and back_project agree on planar depth and on the mounting:
    every ground point lands 0.055 m below base_link."""
    base, roll, pitch, yaw = flat_pose()
    depth = render(base, roll, pitch, yaw)
    pts = optical_to_level(back_project(depth, K, HazcamConfig()), MOUNT_ROTATION,
                           MOUNT_OFFSET, roll, pitch)
    near = np.hypot(pts[:, 0], pts[:, 1]) < 6.0
    assert near.sum() > 10000
    assert np.allclose(pts[near, 2], -BASE_ABOVE_GROUND_M, atol=1e-3)
    # Nearest visible ground: the bottom image row, 0.81 rad below horizontal.
    assert pts[near, 0].min() == pytest.approx(0.20 + 0.25 / math.tan(0.25 + math.atan(240 / FX)),
                                               abs=0.02)


def test_roll_pitch_yaw_round_trip():
    for roll, pitch, yaw in [(0.01, -0.05, 2.0), (-0.2, 0.3, -1.0)]:
        r = _yaw_matrix(yaw) @ level_rotation(roll, pitch)
        # matrix -> quaternion (w > 0 branch is fine for these small tilts)
        w = math.sqrt(1.0 + np.trace(r)) / 2.0
        q = ((r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w),
             (r[1, 0] - r[0, 1]) / (4 * w), w)
        assert roll_pitch_yaw(*q) == pytest.approx((roll, pitch, yaw), abs=1e-9)


def test_level_to_odom():
    out = level_to_odom(np.array([[1.0, 0.0], [0.0, 2.0]]), 10.0, -5.0, math.pi / 2)
    assert out == pytest.approx(np.array([[10.0, -4.0], [8.0, -5.0]]))


def test_depth_image_decoding():
    raw = np.array([[1.5, np.inf], [np.nan, 4.25]], dtype="<f4")
    padded = np.zeros((2, 3), dtype="<f4")  # row step wider than the image
    padded[:, :2] = raw
    out = depth_image_to_array("32FC1", 2, 2, 12, False, padded.tobytes())
    assert out[0, 0] == 1.5 and out[1, 1] == 4.25 and np.isinf(out[0, 1]) and np.isnan(out[1, 0])
    mm = np.array([[1500, 0]], dtype=">u2")
    out = depth_image_to_array("16UC1", 1, 2, 4, True, mm.tobytes())
    assert out[0, 0] == pytest.approx(1.5) and np.isnan(out[0, 1])
    with pytest.raises(ValueError):
        depth_image_to_array("rgb8", 1, 1, 3, False, b"\0\0\0")


# ---------------------------------------------------------------- no rock, no detection


@pytest.mark.parametrize("roll_err_deg,pitch_err_deg", [
    (0.0, 0.0), (0.35, 0.0), (0.0, -0.35), (0.35, 0.35), (1.0, -1.0),
])
def test_flat_ground_has_no_detections_under_tilt_error(roll_err_deg, pitch_err_deg):
    """The RC-1 IMU's tilt bias is about 0.35 deg sigma; 1 deg is three sigma."""
    base, roll, pitch, yaw = flat_pose()
    depth = render(base, roll, pitch, yaw)
    det = detect(depth, math.radians(roll_err_deg), math.radians(pitch_err_deg))
    assert len(det.xy_level) == 0


@pytest.mark.parametrize("slope_deg,on_slope,bias_deg", [
    (3.0, False, 0.0),   # flat under the rover, 3 deg rise ahead; EKF-style zero tilt
    (3.0, True, 0.35),   # standing on it, IMU pitch biased
    (-3.0, True, -0.35),
    (15.0, False, 0.0),  # crater-wall stress case, well above the 2-3 deg regional slope
])
def test_gentle_slope_has_no_detections(slope_deg, on_slope, bias_deg):
    s = math.tan(math.radians(slope_deg))
    pitch_true = -math.radians(slope_deg) if on_slope else 0.0
    base = (0.0, 0.0, BASE_ABOVE_GROUND_M / math.cos(math.radians(slope_deg)))
    depth = render(base, 0.0, pitch_true, 0.0, ground_slope_xy=(s, 0.0))
    det = detect(depth, 0.0, pitch_true + math.radians(bias_deg))
    assert len(det.xy_level) == 0


def test_side_slope_has_no_detections():
    s = math.tan(math.radians(3.0))
    roll_true = math.radians(3.0)  # ground rising to +y: rover rolled left side up
    base = (0.0, 0.0, BASE_ABOVE_GROUND_M)
    depth = render(base, roll_true, 0.0, 0.0, ground_slope_xy=(0.0, s))
    assert len(detect(depth, roll_true + math.radians(0.35), 0.0).xy_level) == 0


# ---------------------------------------------------------------- rocks


def _box_at(base_xy, yaw, forward_m, left_m, size_m, height_m):
    cx, cy = rover_frame_to_world(base_xy, yaw, forward_m, left_m)
    h = size_m / 2.0
    return (cx - h, cy - h, cx + h, cy + h, height_m), (cx, cy)


def test_0p3m_rock_at_5m_detected_in_the_right_odom_cell():
    base_xy, yaw = (12.3, -7.6), 0.6
    box, centre = _box_at(base_xy, yaw, 5.0, 0.0, 0.4, 0.3)
    base, roll, pitch, _ = flat_pose(*base_xy, yaw)
    det = detect(render(base, roll, pitch, yaw, boxes=[box]), roll, pitch)
    assert len(det.xy_level) > 0
    odom = level_to_odom(det.xy_level, base_xy[0], base_xy[1], yaw)
    rock_cell = costmap_cell(centre)
    cells = {costmap_cell(p) for p in odom}
    assert all(abs(c[0] - rock_cell[0]) <= 1 and abs(c[1] - rock_cell[1]) <= 1 for c in cells)
    # Every detection sits on or right beside the rock: no spurious far-field cells.
    assert np.max(np.hypot(odom[:, 0] - centre[0], odom[:, 1] - centre[1])) < 0.45
    assert det.step_m.max() == pytest.approx(0.3, abs=0.03)


def test_hemisphere_rock_off_axis_detected():
    base_xy, yaw = (-3.0, 4.0), -2.2
    cx, cy = rover_frame_to_world(base_xy, yaw, 4.5, 1.5)
    base, roll, pitch, _ = flat_pose(*base_xy, yaw)
    det = detect(render(base, roll, pitch, yaw, domes=[(cx, cy, 0.3)]), roll, pitch)
    assert len(det.xy_level) > 0
    odom = level_to_odom(det.xy_level, base_xy[0], base_xy[1], yaw)
    assert np.max(np.hypot(odom[:, 0] - cx, odom[:, 1] - cy)) < 0.45


def test_rock_on_a_slope_with_tilt_bias_detected_and_nothing_else():
    s = math.tan(math.radians(3.0))
    pitch_true = -math.radians(3.0)
    yaw = 0.0
    base = (0.0, 0.0, BASE_ABOVE_GROUND_M)
    # Ground at x = 4 m is 4 * tan(3 deg) = 0.21 m up; the box shows 0.3 m above it.
    box = (3.8, -0.2, 4.2, 0.2, 4.0 * s + 0.3)
    depth = render(base, 0.0, pitch_true, yaw, ground_slope_xy=(s, 0.0), boxes=[box])
    det = detect(depth, math.radians(0.35), pitch_true + math.radians(0.35))
    assert len(det.xy_level) > 0
    assert np.max(np.hypot(det.xy_level[:, 0] - 4.0, det.xy_level[:, 1])) < 0.45


def test_nan_and_inf_pixels_are_ignored():
    rng = np.random.default_rng(3)
    base, roll, pitch, yaw = flat_pose()
    flat = render(base, roll, pitch, yaw)
    box, _ = _box_at((0.0, 0.0), 0.0, 4.0, -0.5, 0.4, 0.3)
    rock = render(base, roll, pitch, yaw, boxes=[box])
    for image in (flat, rock):
        mask = rng.random(image.shape) < 0.15
        junk = rng.choice(np.array([np.nan, np.inf, -np.inf, 0.0, -1.0], dtype=np.float32),
                          size=int(mask.sum()))
        image[mask] = junk
    assert len(detect(flat, roll, pitch).xy_level) == 0
    det = detect(rock, roll, pitch)
    assert len(det.xy_level) > 0
    assert np.max(np.hypot(det.xy_level[:, 0] - 4.0, det.xy_level[:, 1] + 0.5)) < 0.45


def test_all_invalid_image_is_harmless():
    depth = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    det = detect(depth, 0.0, 0.0)
    assert len(det.xy_level) == 0 and det.n_points == 0


def test_rock_beyond_max_range_is_not_reported():
    base, roll, pitch, yaw = flat_pose()
    box, _ = _box_at((0.0, 0.0), 0.0, 8.0, 0.0, 0.6, 0.5)
    assert len(detect(render(base, roll, pitch, yaw, boxes=[box]), roll, pitch).xy_level) == 0


def test_object_below_step_threshold_is_not_reported():
    """Documents the stated blind spot: 0.10 m is under the 0.15 m step threshold."""
    base, roll, pitch, yaw = flat_pose()
    box, _ = _box_at((0.0, 0.0), 0.0, 3.0, 0.0, 0.4, 0.10)
    assert len(detect(render(base, roll, pitch, yaw, boxes=[box]), roll, pitch).xy_level) == 0


def test_cpu_per_frame_is_small():
    """Budget: a few percent of a core at 2 frames per simulated second."""
    base, roll, pitch, yaw = flat_pose()
    box, _ = _box_at((0.0, 0.0), 0.0, 4.0, 0.0, 0.4, 0.3)
    depth = render(base, roll, pitch, yaw, boxes=[box])
    detect(depth, roll, pitch)
    started = time.perf_counter()
    for _ in range(10):
        detect(depth, roll, pitch)
    per_frame_s = (time.perf_counter() - started) / 10
    print(f"hazcam detect_obstacles: {per_frame_s * 1e3:.1f} ms per 640x480 frame")
    assert per_frame_s < 0.1
