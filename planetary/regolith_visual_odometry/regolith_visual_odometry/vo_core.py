# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""The geometry behind regolith_visual_odometry, with no ROS in it.

Kept ROS-free on purpose: every claim this makes about recovered motion is then
testable against a synthetic scene with a known answer, without a simulator, a
node, or a clock. See test/test_vo_core.py.

WHY THIS PACKAGE EXISTS. M4's acceptance was 0/3 with an arrival error of
3.1-13.1 m, and the mechanism was pinned by measurement rather than argued: on
every seed the true error equalled the EKF's drift plus the stopping tolerance
to within centimetres. The rover arrives exactly where it believes the goal is.
About 10% of its motion over this terrain is LATERAL slide, and a differential
-drive odometry model cannot represent lateral motion at all - it assumes
vy = 0 by construction - while an IMU measures heading correctly and so never
contradicts it. Nothing in a wheel+IMU stack observes the error, so it
accumulates as a random walk that nothing corrects. Feeding the EKF an absolute
position reference at 0.5 m / 1 Hz turned the same build from 0/3 into 3/3,
which is what identified the estimator, and only the estimator, as the gap.

WHAT THIS DOES AND DOES NOT PROMISE. Visual odometry is a RELATIVE sensor. It
is not the oracle that produced that 3/3: the oracle handed the filter absolute
position and drove EKF divergence to 0.00 m, which nothing onboard can do. What
VO does is observe the term that was structurally invisible - it watches the
ground move sideways past the camera - so the drift rate drops rather than the
drift vanishing. Published visual odometry on planetary rovers runs on the
order of 1-2% of distance travelled; over this acceptance's ~110 m traverses
that is 1-2 m against a 1.5 m bar. Close, not comfortable, and the honest
number is whatever the acceptance run measures.

WHY DEPTH. A single camera cannot observe scale - monocular VO recovers the
shape of a trajectory but not its size, which is useless for correcting a
metre-scale error. The depth camera added alongside the RGB one (same link,
same intrinsics, see regolith_rover.urdf.xacro) is what makes the recovered
translation metric.

WHY IT DOES NOT RUN AT FRAME RATE, WHICH IS THE ONE COUNTER-INTUITIVE PART.
Estimating between adjacent frames is worse than useless here, and the reason is
parallax, not compute. At the rover's 0.2 m/s cruise a 0.1 s interval moves the
camera 2 cm, which on this geometry is about 1 px of optical flow - while the
feature tracker's own noise, measured against the exact known motion of a
synthetic scene, is 0.74 px. Signal and noise are the same size and the estimate
is meaningless: measured 0.056 m/s for a true 0.200 m/s. Waiting for a real
baseline fixes it, and the sweep that set the defaults below (4 seeds x forward /
lateral / combined / yaw, worst case over all of them) is:

    baseline    0.1 s       0.2 s       0.3 s       0.5 s       0.8 s
    flow        1.05 px     2.41 px     2.93 px     5.74 px    10.15 px
    recovered   0.056       0.200       0.192       0.224       0.201   (true 0.200)

So the node holds a KEYFRAME and estimates against it only once ~0.4 s of camera
motion has accumulated, giving ~2.5 velocity updates per second at a worst-case
error of 0.051 m/s. That is ample: this exists to correct a drift that grows over
minutes, not to close a control loop.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Optical frame (x right, y down, z forward - what OpenCV's projection assumes)
# to ROS body frame (x forward, y left, z up). v_body = OPTICAL_TO_BODY @ v_optical.
OPTICAL_TO_BODY = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


@dataclass(frozen=True)
class VoConfig:
    """Tunables, all in one place so the node's parameters map onto them 1:1."""

    max_features: int = 400
    feature_quality: float = 0.01
    min_feature_distance_px: float = 12.0
    lk_window_px: int = 21
    lk_pyramid_levels: int = 3
    # Local contrast equalization before anything else. Measured on real frames
    # from this world: the onboard camera spans pixel values 10-124 with a
    # standard deviation of 9.8 - a dim, flat image, because lunar regolith at a
    # low sun angle is uniformly dark grey and the sky is black. Shi-Tomasi
    # scores corners RELATIVE to the strongest corner in the frame, so on a raw
    # frame it found 25 corners against a 400 budget.
    clahe_clip_limit: float = 3.0
    clahe_grid: int = 8
    # Depth outside this band is dropped. The near bound rejects the chassis and
    # lens-adjacent noise; the far bound matters more than it looks - distant
    # points carry almost no parallax, so their depth error dominates the metric
    # scale while contributing nothing to observing translation.
    min_depth_m: float = 0.4
    max_depth_m: float = 15.0
    # Below this many RANSAC inliers the estimate is not trusted at all. Six is
    # the algebraic minimum for PnP; this sits well above it because near the
    # minimum RANSAC will happily find a consensus among noise.
    min_inliers: int = 20
    ransac_reprojection_px: float = 1.5
    ransac_iterations: int = 200
    # A solved pose whose own inliers still reproject this badly is not a pose.
    # PnP's final refinement occasionally diverges, and when it does it does not
    # fail - it returns a confident, wildly wrong answer. Measured over 136 real
    # frame pairs: the 4% that were catastrophic (up to 46 m/s reported for a
    # 0.2 m/s rover) had a median reprojection RMS of 87 px, against 0.67 px for
    # the other 96%. Two populations three orders of magnitude apart, so the
    # threshold sits far from both. Without this gate those 4% swing the mean
    # velocity error from 0.05 m/s to 1.1 m/s - a filter fed one of them takes a
    # correction worse than the drift it was there to remove.
    max_reprojection_rms_px: float = 3.0
    # Floor on the reported velocity sigma, so a scene that happens to fit well
    # can never claim to be better than the sensor model actually is.
    min_velocity_sigma_mps: float = 0.02


@dataclass(frozen=True)
class VoEstimate:
    """Body-frame motion over one frame interval, plus what it was derived from.

    `valid` false means the geometry was refused, not that the rover was still -
    the caller must publish this as "no information" (a huge covariance), never
    as a measured zero. Reporting a refused estimate as zero velocity would feed
    the EKF a confident ZUPT every time the camera looked at a featureless
    stretch, which is precisely the corruption wheel_slip_node.py exists to stop.
    """

    valid: bool
    linear_mps: np.ndarray  # (vx, vy, vz) of the camera, in base_link axes
    angular_rps: np.ndarray  # (wx, wy, wz), in base_link axes
    velocity_sigma_mps: float
    n_tracked: int
    n_inliers: int
    reprojection_rms_px: float
    reason: str = ""


def _invalid(reason: str, n_tracked: int = 0, n_inliers: int = 0) -> VoEstimate:
    return VoEstimate(
        valid=False,
        linear_mps=np.zeros(3),
        angular_rps=np.zeros(3),
        velocity_sigma_mps=float("inf"),
        n_tracked=n_tracked,
        n_inliers=n_inliers,
        reprojection_rms_px=float("nan"),
        reason=reason,
    )


def back_project(pixels_xy: np.ndarray, depths_m: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    """Pixels + metric depth -> 3D points in the camera's OPTICAL frame."""
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    u, v = pixels_xy[:, 0], pixels_xy[:, 1]
    return np.column_stack([(u - cx) / fx * depths_m, (v - cy) / fy * depths_m, depths_m])


def sample_depth(depth_image: np.ndarray, pixels_xy: np.ndarray) -> np.ndarray:
    """Nearest-pixel depth lookup. Out-of-bounds and non-finite come back as NaN."""
    height, width = depth_image.shape[:2]
    cols = np.rint(pixels_xy[:, 0]).astype(int)
    rows = np.rint(pixels_xy[:, 1]).astype(int)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    out = np.full(len(pixels_xy), np.nan)
    sampled = depth_image[rows[inside], cols[inside]].astype(np.float64)
    # gz writes unreturned rays as +inf (and some drivers as 0); both mean "no range".
    sampled[~np.isfinite(sampled) | (sampled <= 0.0)] = np.nan
    out[inside] = sampled
    return out


def enhance(gray: np.ndarray, cfg: VoConfig) -> np.ndarray:
    """Local contrast equalization (CLAHE), applied to every frame before use.

    Not cosmetic. See VoConfig.clahe_clip_limit: raw frames from this world are
    dim and flat, and corner detection is relative, so without this the detector
    finds a few dozen features where it should find hundreds. Applied to BOTH
    frames of a pair so the tracker compares like with like.
    """
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=(cfg.clahe_grid,) * 2)
    return clahe.apply(gray)


def usable_depth_mask(depth_image: np.ndarray, cfg: VoConfig) -> np.ndarray:
    """Where features are allowed to be found: pixels with real, in-band range.

    This mask is the difference between visual odometry working here and not.
    The strongest corners in a lunar scene sit on the SKYLINE - bright sunlit
    terrain against a near-black sky - and the sky has no range at all, so an
    unmasked detector spends its whole feature budget on points that can never
    be given a 3D position. Measured on real captured frames: of 25 corners
    found, 8 had usable depth, which is below the inlier floor, so every single
    frame pair in the first live run was refused.

    Eroded by a few pixels so features do not land on a depth discontinuity,
    where the nearest-pixel range lookup may belong to whichever surface the
    corner is not on.
    """
    usable = (
        np.isfinite(depth_image)
        & (depth_image >= cfg.min_depth_m)
        & (depth_image <= cfg.max_depth_m)
    ).astype(np.uint8)
    return cv2.erode(usable, np.ones((5, 5), np.uint8))


def detect_features(gray: np.ndarray, cfg: VoConfig, mask: np.ndarray | None = None) -> np.ndarray:
    """Shi-Tomasi corners as an (N, 2) float array of (x, y) pixels."""
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=cfg.max_features,
        qualityLevel=cfg.feature_quality,
        minDistance=cfg.min_feature_distance_px,
        mask=mask,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def track_features(prev_gray: np.ndarray, cur_gray: np.ndarray, prev_xy: np.ndarray, cfg: VoConfig):
    """Lucas-Kanade forward/backward track. Returns the surviving (prev, cur) pairs.

    The backward check is what keeps this honest on repetitive ground: regolith
    at a low sun angle is full of near-identical shadowed pits, and a forward-only
    track will happily slide a corner onto its neighbour. Requiring the reverse
    track to land back within a pixel of where it started discards those.
    """
    if len(prev_xy) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    lk_params = dict(
        winSize=(cfg.lk_window_px, cfg.lk_window_px),
        maxLevel=cfg.lk_pyramid_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    forward, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, prev_xy.reshape(-1, 1, 2), None, **lk_params
    )
    backward, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
        cur_gray, prev_gray, forward, None, **lk_params
    )

    forward = forward.reshape(-1, 2)
    backward = backward.reshape(-1, 2)
    ok = (status_fwd.ravel() == 1) & (status_bwd.ravel() == 1)
    ok &= np.linalg.norm(backward - prev_xy, axis=1) < 1.0
    return prev_xy[ok], forward[ok]


def estimate_motion(
    prev_gray: np.ndarray,
    prev_depth: np.ndarray,
    cur_gray: np.ndarray,
    k_matrix: np.ndarray,
    dt_s: float,
    base_from_camera_rotation: np.ndarray,
    camera_offset_in_base_m: np.ndarray,
    cfg: VoConfig,
) -> VoEstimate:
    """Recovers body-frame velocity from one frame pair.

    The chain, because each step is a place a sign can silently invert:

      1. Track corners from the previous image into the current one.
      2. Give the PREVIOUS frame's corners metric 3D positions from its depth
         image. These are landmarks in the previous optical frame.
      3. solvePnPRansac fits the current camera's pose to those landmarks, so it
         returns (R, t) with p_cur = R @ p_prev + t. That is the transform of
         POINTS, which is the inverse of the motion of the CAMERA - the single
         easiest sign error to make here, and the reason the camera displacement
         below is -R.T @ t rather than t.
      4. Rotate optical -> camera body -> base_link.
      5. Subtract the lever arm. The camera sits 0.2 m ahead of and 0.2 m above
         base_link, so while the rover yaws, the camera translates even though
         base_link does not: v_base = v_camera - omega x r.
    """
    if dt_s <= 0.0:
        return _invalid("non-positive dt")

    prev_view, cur_view = enhance(prev_gray, cfg), enhance(cur_gray, cfg)
    # Look for features only where the depth camera has something to say - see
    # usable_depth_mask, without which the detector spends its budget on the sky.
    corners = detect_features(prev_view, cfg, usable_depth_mask(prev_depth, cfg))
    prev_xy, cur_xy = track_features(prev_view, cur_view, corners, cfg)
    n_tracked = len(prev_xy)
    if n_tracked < cfg.min_inliers:
        return _invalid("too few tracked features", n_tracked=n_tracked)

    depths = sample_depth(prev_depth, prev_xy)
    usable = np.isfinite(depths) & (depths >= cfg.min_depth_m) & (depths <= cfg.max_depth_m)
    prev_xy, cur_xy, depths = prev_xy[usable], cur_xy[usable], depths[usable]
    if len(prev_xy) < cfg.min_inliers:
        return _invalid("too few features with usable depth", n_tracked=n_tracked)

    object_points = back_project(prev_xy, depths, k_matrix)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points.astype(np.float64),
        cur_xy.astype(np.float64),
        k_matrix.astype(np.float64),
        None,
        iterationsCount=cfg.ransac_iterations,
        reprojectionError=cfg.ransac_reprojection_px,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None or len(inliers) < cfg.min_inliers:
        n_in = 0 if inliers is None else len(inliers)
        return _invalid("PnP found no consensus", n_tracked=n_tracked, n_inliers=n_in)

    inliers = inliers.ravel()
    projected, _ = cv2.projectPoints(object_points[inliers], rvec, tvec, k_matrix, None)
    residuals = np.linalg.norm(projected.reshape(-1, 2) - cur_xy[inliers], axis=1)
    reprojection_rms = float(np.sqrt(np.mean(residuals**2)))
    if reprojection_rms > cfg.max_reprojection_rms_px:
        return _invalid(
            "solved pose does not reproject its own inliers",
            n_tracked=n_tracked,
            n_inliers=len(inliers),
        )

    rotation_points, _ = cv2.Rodrigues(rvec)
    # Motion of the camera, not of the points - see step 3 above.
    displacement_optical = (-rotation_points.T @ tvec).ravel()
    rotation_camera = rotation_points.T

    displacement_body = OPTICAL_TO_BODY @ displacement_optical
    rotation_body = OPTICAL_TO_BODY @ rotation_camera @ OPTICAL_TO_BODY.T

    displacement_base = base_from_camera_rotation @ displacement_body
    rotvec_base = base_from_camera_rotation @ cv2.Rodrigues(rotation_body)[0].ravel()

    angular_rps = rotvec_base / dt_s
    camera_linear_mps = displacement_base / dt_s
    # Lever arm: the camera is not at base_link, so yaw alone moves it.
    linear_mps = camera_linear_mps - np.cross(angular_rps, camera_offset_in_base_m)

    return VoEstimate(
        valid=True,
        linear_mps=linear_mps,
        angular_rps=angular_rps,
        velocity_sigma_mps=velocity_sigma(
            reprojection_rms, float(np.mean(depths[inliers])), k_matrix, len(inliers), dt_s, cfg
        ),
        n_tracked=n_tracked,
        n_inliers=len(inliers),
        reprojection_rms_px=reprojection_rms,
    )


def velocity_sigma(
    reprojection_rms_px: float,
    mean_depth_m: float,
    k_matrix: np.ndarray,
    n_inliers: int,
    dt_s: float,
    cfg: VoConfig,
) -> float:
    """Propagates the measured pixel residual into a velocity sigma.

    Deliberately derived rather than tuned, so it moves with the scene instead
    of being a constant that happens to work on one seed. A residual of e pixels
    at depth Z with focal length f is a metric error of about e * Z / f; two
    frames contribute, so sqrt(2); averaging over n independent inliers divides
    by sqrt(n); and a displacement error over dt is a velocity error over dt.

    It captures the geometry, not everything: it says nothing about a consensus
    that is confidently wrong (a moving object filling the frame, say), which is
    why min_inliers gates admission separately rather than just widening sigma.
    """
    focal_px = float(k_matrix[0, 0])
    position_sigma = np.sqrt(2.0) * reprojection_rms_px * mean_depth_m / focal_px
    sigma = position_sigma / max(np.sqrt(n_inliers), 1.0) / dt_s
    return float(max(sigma, cfg.min_velocity_sigma_mps))
