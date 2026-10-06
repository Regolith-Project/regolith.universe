# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Onboard hazard detection from the forward depth camera: the geometry, without ROS.

docs/TRL4_BASELINE.md gap G10. In the reference configuration the planner knows
only the rocks an orbiter could resolve (>= 1.5 m across); on a typical world 66
of 190 rocks are missing from its map and the rover finds them by driving into
them. Every real planetary rover carries hazard cameras for exactly this. This
rover already carries a depth camera (camera_link, 640x480 at 10 Hz, hfov 1.4 rad,
pitched 0.25 rad down, about 0.25 m above the ground on flat terrain), so the
hazard layer is software only.

THE TEST IS A LOCAL STEP, NOT A HEIGHT ABOVE AN ASSUMED GROUND PLANE. Points are
binned into a rover-local, gravity-levelled elevation grid; a cell is an obstacle
when its highest point stands at least `step_height_m` above the lowest point of
any cell within `step_radius_cells`. A rock is a local height discontinuity; a
gentle slope is not. The alternative - height above the plane the rover thinks it
stands on - turns every attitude error into a range-proportional height error:
the 0.35 deg (1 sigma) tilt bias of the RC-1 IMU model is 0.037 m at 6 m, and the
EKF runs in two_d_mode, so an EKF-only attitude would be off by the full terrain
slope under the wheels (2-3 deg regionally, more on a crater wall: 0.3 m at 6 m).
The step test reads attitude only through the local grade, where a tilt error of
d degrees moves the apparent grade by d degrees: irrelevant against a test that
fires at 25 deg (below).

THRESHOLDS, AND WHY.
  * step_height_m = 0.15. The chassis clears 0.11 m and the wheels are 0.09 m in
    radius (regolith_rover.urdf.xacro), so anything at or above 0.15 m can neither
    be straddled nor climbed. Every unmapped rock on seeds 42/7/123 has a
    collision ellipsoid at least 0.39 m tall (2 x min vertical radius, before the
    10% embedding), so the threshold sits well under the smallest real target.
    Missed by design: objects 0.09-0.15 m tall, which could still high-centre the
    rover. None exist in the generated worlds (no rock under 0.3 m), but a real
    site would have them.
  * step_radius_cells = 2 at cell_m = 0.1, a disc footprint (cell offsets with
    di^2 + dj^2 <= 4). The farthest two points that can be compared are in cells
    (2, 0) apart: at most 0.3 m by 0.1 m, 0.32 m horizontally. A 0.15 m step so
    needs an average grade of at least atan(0.15 / 0.32) = 25 deg - steeper than
    the planner's 20 deg lethal slope. Terrain the a-priori map calls traversable
    cannot trigger it, with any tilt error the IMU model produces (sub-degree).
    Real rocks trigger it from their own face, which rises from the ground to the
    rock's top over a few centimetres to a few decimetres of horizontal distance.
  * max_range_m = 6.0 (horizontal, from base_link). At range r one image row
    covers r^2 / h * (1 / fy) of ground (h = 0.25 m camera height, fy = 381 px):
    0.26 m at 5 m, 0.38 m at 6 m, 0.67 m at 8 m. Beyond about 6 m the ground
    between rows is unsampled and a rock's face spans few rows. 6 m is also
    30 s of driving at the 0.2 m/s cruise speed, far more warning than the
    planner needs. The simulated depth has no noise at all; a real stereo hazcam's
    range error grows as r^2 (about 0.1 m at 5 m for a 0.1 m baseline), so 6 m is
    optimistic for hardware - MER's hazcams were used to about 3 m.
  * min_points_obstacle = 3, min_points_ground = 2. A cell's height is believed
    only when several pixels agree, so one stray pixel cannot create a step.

DEPTH CONVENTION. Planar depth along the optical axis, the convention
regolith_visual_odometry's back_project already uses on these same frames.
Non-finite, zero and negative pixels (sky, out of clip) are dropped.

CPU. A 640x480 frame is subsampled to every 4th column and every row: 76,800
pixels, of which about two thirds see ground. Every row is kept because far
range is where row spacing runs out (see max_range_m); the column stride costs
nothing, a 4-pixel stride is 6 cm laterally at 6 m. Per-cell min/max use a sort
rather than numpy's ufunc.at, which on this machine's numpy 1.21 runs at about a
microsecond per element.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage

# Optical frame (x right, y down, z forward) to the camera link's body frame
# (x forward, y left, z up). gz cameras look along their link's +X. Same matrix
# as regolith_visual_odometry.vo_core.OPTICAL_TO_BODY.
OPTICAL_TO_BODY = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


@dataclass(frozen=True)
class HazcamConfig:
    """Tunables; the node's parameters map onto these 1:1. Reasoning in the module docstring."""

    stride_u: int = 4
    stride_v: int = 1
    min_depth_m: float = 0.1
    max_range_m: float = 6.0
    cell_m: float = 0.1
    step_height_m: float = 0.15
    step_radius_cells: int = 2
    min_points_obstacle: int = 3
    min_points_ground: int = 2


@dataclass
class ElevationGrid:
    """Rover-local, gravity-levelled grid.

    Index [ix, iy]; cell centre is (x_min + (ix + 0.5) * cell_m, y_min + (iy + 0.5) * cell_m).
    """

    z_max: np.ndarray
    z_min: np.ndarray
    count: np.ndarray
    x_min: float
    y_min: float
    cell_m: float


@dataclass
class Detections:
    """Obstacle cells of one frame: centres in the levelled base frame, and step heights."""

    xy_level: np.ndarray  # (N, 2)
    step_m: np.ndarray  # (N,)
    n_points: int  # valid points that entered the grid (diagnostic)


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Rotation matrix of a unit quaternion."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def roll_pitch_yaw(x: float, y: float, z: float, w: float) -> tuple:
    """ZYX Euler angles of a quaternion (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def level_rotation(roll: float, pitch: float) -> np.ndarray:
    """base_link -> yaw-free gravity-levelled frame: Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return ry @ rx


def back_project(depth: np.ndarray, k_matrix: np.ndarray, cfg: HazcamConfig) -> np.ndarray:
    """Subsampled depth image -> (N, 3) points in the OPTICAL frame. Invalid pixels dropped.

    Depth is planar (along the optical axis). NaN, +/-inf, zero and anything under
    min_depth_m are not measurements - gz returns inf for sky beyond the far clip -
    and are removed before any arithmetic touches them.
    """
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    height, width = depth.shape[:2]
    v = np.arange(0, height, cfg.stride_v, dtype=np.float64)
    u = np.arange(0, width, cfg.stride_u, dtype=np.float64)
    d = np.asarray(depth[:: cfg.stride_v, :: cfg.stride_u], dtype=np.float64)
    # A planar depth beyond max_range / cos(hfov/2 + margin) cannot be within
    # max_range horizontally; cap generously and let the horizontal cut be exact.
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(d) & (d >= cfg.min_depth_m) & (d <= 3.0 * cfg.max_range_m)
    vv, uu = np.nonzero(valid)
    z = d[vv, uu]
    return np.column_stack([(u[uu] - cx) / fx * z, (v[vv] - cy) / fy * z, z])


def optical_to_level(
    points_optical: np.ndarray,
    mount_rotation: np.ndarray,
    mount_offset: np.ndarray,
    roll: float,
    pitch: float,
) -> np.ndarray:
    """Optical-frame points -> the levelled base frame (origin base_link, x along heading, z up).

    mount_rotation / mount_offset: camera_link expressed in base_link (from TF).
    roll / pitch: the rover's attitude, from the IMU.
    """
    rotation = level_rotation(roll, pitch) @ mount_rotation @ OPTICAL_TO_BODY
    offset = level_rotation(roll, pitch) @ np.asarray(mount_offset, dtype=np.float64)
    return points_optical @ rotation.T + offset


def build_elevation_grid(points_level: np.ndarray, cfg: HazcamConfig) -> ElevationGrid:
    """Bin points into per-cell max/min height and count, within max_range_m horizontally."""
    n_x = int(math.ceil(cfg.max_range_m / cfg.cell_m))
    n_y = 2 * n_x
    x_min, y_min = 0.0, -n_x * cfg.cell_m
    z_max = np.full((n_x, n_y), -np.inf)
    z_min = np.full((n_x, n_y), np.inf)
    count = np.zeros((n_x, n_y), dtype=np.int64)
    if len(points_level) == 0:
        return ElevationGrid(z_max, z_min, count, x_min, y_min, cfg.cell_m)

    x, y, z = points_level[:, 0], points_level[:, 1], points_level[:, 2]
    keep = np.hypot(x, y) <= cfg.max_range_m
    ix = np.floor((x - x_min) / cfg.cell_m).astype(np.int64)
    iy = np.floor((y - y_min) / cfg.cell_m).astype(np.int64)
    keep &= (ix >= 0) & (ix < n_x) & (iy >= 0) & (iy < n_y) & np.isfinite(z)
    if not keep.any():
        return ElevationGrid(z_max, z_min, count, x_min, y_min, cfg.cell_m)
    flat = ix[keep] * n_y + iy[keep]
    zk = z[keep]

    # Sorted by (cell, z): each cell's run starts at its min and ends at its max.
    order = np.lexsort((zk, flat))
    flat_s, z_s = flat[order], zk[order]
    starts = np.flatnonzero(np.r_[True, flat_s[1:] != flat_s[:-1]])
    ends = np.r_[starts[1:], len(flat_s)] - 1
    cells = flat_s[starts]
    z_min.flat[cells] = z_s[starts]
    z_max.flat[cells] = z_s[ends]
    count.flat[cells] = ends - starts + 1
    return ElevationGrid(z_max, z_min, count, x_min, y_min, cfg.cell_m)


def detect_steps(grid: ElevationGrid, cfg: HazcamConfig) -> tuple:
    """Return (obstacle mask, step heights).

    An obstacle is a cell whose top stands step_height_m above the lowest believed
    point within step_radius_cells (itself included).
    """
    ground = np.where(grid.count >= cfg.min_points_ground, grid.z_min, np.inf)
    r = cfg.step_radius_cells
    yy, xx = np.mgrid[-r: r + 1, -r: r + 1]
    footprint = (xx * xx + yy * yy) <= r * r
    lowest_near = ndimage.minimum_filter(ground, footprint=footprint, mode="constant", cval=np.inf)
    with np.errstate(invalid="ignore"):
        step = grid.z_max - lowest_near
        obstacle = (grid.count >= cfg.min_points_obstacle) & np.isfinite(step) & (
            step >= cfg.step_height_m
        )
    return obstacle, np.where(obstacle, step, 0.0)


def detect_obstacles(
    depth: np.ndarray,
    k_matrix: np.ndarray,
    mount_rotation: np.ndarray,
    mount_offset: np.ndarray,
    roll: float,
    pitch: float,
    cfg: HazcamConfig = HazcamConfig(),
) -> Detections:
    """One depth frame -> obstacle cell centres in the levelled base frame."""
    points = optical_to_level(
        back_project(depth, k_matrix, cfg), mount_rotation, mount_offset, roll, pitch
    )
    grid = build_elevation_grid(points, cfg)
    obstacle, step = detect_steps(grid, cfg)
    ix, iy = np.nonzero(obstacle)
    xy = np.column_stack(
        [grid.x_min + (ix + 0.5) * grid.cell_m, grid.y_min + (iy + 0.5) * grid.cell_m]
    )
    return Detections(xy, step[ix, iy], int(grid.count.sum()))


def level_to_odom(xy_level: np.ndarray, x_m: float, y_m: float, yaw: float) -> np.ndarray:
    """Levelled base-frame (x, y) -> odom (x, y), given the EKF's base_link pose in odom."""
    xy_level = np.asarray(xy_level, dtype=np.float64).reshape(-1, 2)
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = xy_level[:, 0], xy_level[:, 1]
    return np.column_stack([x_m + c * x - s * y, y_m + s * x + c * y])


def ground_edge_bias(points_level: np.ndarray, near_m=(1.0, 2.0)) -> float:
    """Median height of near ground at the image edges minus at the centre (m), NaN if unknown.

    A live diagnostic for the one convention synthetic tests cannot settle: if gz's
    depth were distance along the ray rather than planar, every off-axis point would
    be pushed outward along its ray and edge ground would read low (about -0.03 m at
    1.5 m and 0.5 rad bearing). Planar depth on ground of a few degrees reads ~0.
    """
    if len(points_level) == 0:
        return float("nan")
    x, y, z = points_level[:, 0], points_level[:, 1], points_level[:, 2]
    rng = np.hypot(x, y)
    band = (rng >= near_m[0]) & (rng <= near_m[1])
    bearing = np.abs(np.arctan2(y, x))
    centre = band & (bearing < 0.15)
    edge = band & (bearing > 0.45)
    if centre.sum() < 10 or edge.sum() < 10:
        return float("nan")
    return float(np.median(z[edge]) - np.median(z[centre]))


def depth_image_to_array(encoding: str, height: int, width: int, step: int,
                         is_bigendian: bool, data: bytes) -> np.ndarray:
    """sensor_msgs/Image bytes -> float32 metres, without cv_bridge.

    32FC1 is metres (what the gz bridge publishes); 16UC1 is millimetres with 0 as
    no-return (the ROS depth convention), converted so 0 becomes NaN.
    """
    if encoding == "32FC1":
        dtype = np.dtype(np.float32)
    elif encoding == "16UC1":
        dtype = np.dtype(np.uint16)
    else:
        raise ValueError(f"unsupported depth encoding '{encoding}' (expected 32FC1 or 16UC1)")
    dtype = dtype.newbyteorder(">" if is_bigendian else "<")
    row_items = step // dtype.itemsize
    image = np.frombuffer(data, dtype=dtype, count=height * row_items).reshape(height, row_items)
    image = image[:, :width]
    if encoding == "16UC1":
        out = image.astype(np.float32) / 1000.0
        out[image == 0] = np.nan
        return out
    return image.astype(np.float32, copy=False)
