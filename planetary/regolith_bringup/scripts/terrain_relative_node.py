#!/usr/bin/env python3
# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Terrain-relative navigation: an EARNED absolute position fix, from onboard sensors.

This is the node `absolute_reference_relay.py` was a stand-in for. That relay
publishes `/absolute_reference/pose` from ground truth - an oracle - to test one
falsifiable claim: that localisation is the only thing between this stack and
M4. It passed 3/3 where the unaided stack passes 1/3. This node publishes the
same topic, in the same units, at a comparable rate and accuracy, WITHOUT ever
reading ground truth.

WHAT IT MEASURES. The rover's attitude is an absolute measurement: gravity does
not drift. Driving over known terrain, the sequence of roll/pitch the IMU reports
is a signature of where on that terrain the rover actually is. The a-priori DEM
(the same terrain heightmap `regolith_costmap` already loads - on a real mission,
an orbital DEM) predicts roll/pitch for any candidate position. Sliding the
recent trajectory over the DEM and scoring predicted against measured attitude
gives an absolute position fix. This is TERCOM - terrain contour matching - the
technique cruise missiles and planetary landers use for exactly this reason: it
needs no beacon, no landmark catalogue, and no external signal, which is the
situation on the Moon.

WHAT IT DOES NOT USE. No ground truth, no GPS, no camera. Wheel odometry and an
IMU supply the trajectory shape and the attitude; the DEM is a file loaded before
the run. Every input is something a real rover has.

WHY IT IS NOT DEAD RECKONING. The fix is absolute, so its error does not grow
with distance. Dead reckoning's does: the rover slides sideways ~10% of its
motion, a differential-drive model cannot represent lateral velocity and an IMU
cannot observe it, so that error accumulates uncorrected and permanently. That is
the measured cause of M4's arrival error on every failing seed.

MEASURED, before this node ever ran live. Replayed through 25 recorded runs
across seeds 7, 55 and 123 - no new simulation time - maintaining a running
correction the way the EKF would, with ground truth used only to score and never
inside the loop:

    final EKF error, median over 25 runs:   2.97 m -> 0.71 m
    runs finishing outside M4's 1.5 m bar:  18 of 25 -> 8 of 25
    seed 123, the drift-limited seed:       11.18 m -> 1.64 m

`absolute_reference_relay.py`'s docstring names a 0.5 m-sigma, ~1 Hz fix as
"what a real terrain-relative or visual-odometry fix would deliver". That was
written as an assumption. This is the measurement of it, and it lands in the
same range from a slower, lumpier update.

WINDOW LENGTH is the parameter that matters most, and 12, 15 and 20 m all land
within 0.53-0.71 m median in that replay while 30 m gives 1.80 m and 40 m gives
2.15 m. Shorter wins because the fix assumes ONE offset for the whole window,
and the estimate drifts while the window fills - over 30 m on seed 123 that
drift is metres, and no single offset can represent it. 15 m is the middle of
the flat part of that curve rather than its argmin, deliberately: the difference
between 12 and 20 m is inside what 25 runs can resolve.

HONEST LIMITS, stated before any result is claimed:

- The DEM is the generator's own heightmap, read exactly, so the map is perfect.
  A real orbital DEM carries registration error and coarser resolution. Degrading
  it deliberately and re-measuring is the obvious next experiment and it has NOT
  been run - there is no parameter for it yet. Until there is, every number above
  is a best case on map quality.
- The replay above is a model of the EKF, not the EKF: it applies each accepted
  fix to a running correction, where the real filter weights it against its own
  covariance and keeps drifting between fixes. It is evidence that the fix
  carries real information, not a prediction of the live number.
- This simulator's IMU is configured with no noise model, so its attitude is
  effectively exact. A real IMU's levelling error would enter the match directly.
  The EKF this feeds already fuses that same IMU, so the idealisation is not new
  here, but it is not free either.
- Attitude is informative only where terrain has relief. On flat ground the match
  is ambiguous - which is what `min_margin` rejects rather than papers over.
- Terrain matching cannot observe heading. Only x and y are published, and the
  EKF config fuses only x and y.
"""

import json
import math
from pathlib import Path

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

from regolith_costmap.costmap_node import load_heightmap


def terrain_gradients(manifest: dict) -> tuple:
    """Return (gx, gy, world_size_m, resolution_m) for the a-priori DEM.

    `load_heightmap` is reused rather than reimplemented on purpose: it carries
    two decodes that are easy to get wrong and were both live defects once - the
    `heightmap_z_min_m`/`heightmap_z_span_m` vertical decode (using
    `height_range_m` instead overstates every slope by 20-25%) and the transpose
    into [row = y, col = x] (without it the whole slope field is mirrored about
    the x = y diagonal). A gradient field is exactly as sensitive to both as the
    costmap is, and matching against a mirrored DEM would fail in a way that
    looks like "terrain matching does not work here" rather than like a bug.
    """
    dem = load_heightmap(manifest)
    world_size_m = float(manifest["world_size_m"])
    resolution_m = world_size_m / (dem.shape[0] - 1)
    gy, gx = np.gradient(dem, resolution_m)
    return gx, gy, world_size_m, resolution_m


def bilinear(field: np.ndarray, x, y, world_size_m: float, resolution_m: float):
    """Sample `field` (indexed [row = y, col = x]) at world coordinates x, y.

    Coordinates are clamped to the DEM rather than masked: the world is 200 m
    across and the mission runs well inside it, so an out-of-bounds sample means
    a wild candidate offset, and clamping scores it against the edge (a poor
    match) instead of introducing NaNs into the cost surface.
    """
    n = field.shape[0]
    col = np.clip((np.asarray(x) + world_size_m / 2.0) / resolution_m, 0.0, n - 1.001)
    row = np.clip((np.asarray(y) + world_size_m / 2.0) / resolution_m, 0.0, n - 1.001)
    c0 = np.floor(col).astype(int)
    r0 = np.floor(row).astype(int)
    fc = col - c0
    fr = row - r0
    return (
        field[r0, c0] * (1 - fr) * (1 - fc)
        + field[r0, c0 + 1] * (1 - fr) * fc
        + field[r0 + 1, c0] * fr * (1 - fc)
        + field[r0 + 1, c0 + 1] * fr * fc
    )


def predicted_attitude(gx, gy, world_size_m, resolution_m, x, y, yaw):
    """Roll and pitch a rover at (x, y, yaw) would have on this DEM, in radians.

    The terrain gradient projected onto the heading is pitch (nose down on a
    downslope, hence the sign) and onto the left axis is roll. Small-slope
    territory - `arctan` is kept anyway because it costs nothing and keeps the
    prediction honest on crater walls.
    """
    grad_x = bilinear(gx, x, y, world_size_m, resolution_m)
    grad_y = bilinear(gy, x, y, world_size_m, resolution_m)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    pitch = -np.arctan(grad_x * cos_yaw + grad_y * sin_yaw)
    roll = np.arctan(-grad_x * sin_yaw + grad_y * cos_yaw)
    return roll, pitch


def match_offset(
    gx,
    gy,
    world_size_m,
    resolution_m,
    xs,
    ys,
    yaws,
    rolls,
    pitches,
    search_m: float = 6.0,
    step_m: float = 0.25,
):
    """Find the (dx, dy) that best explains measured attitude along this trajectory.

    Returns (dx, dy, margin). `margin` is the median candidate cost divided by
    the best candidate's - how much better the winner is than a typical wrong
    answer. It is the ambiguity measure the node gates on: on featureless ground
    every candidate scores alike and margin approaches 1, which is the matcher
    saying "this terrain does not tell you where you are" rather than guessing.

    Scored as plain squared error on absolute roll and pitch. Mean-removed and
    Huber variants were both tried on the same 99 recorded windows and both were
    worse (median 0.96/0.97 m against 0.82 m, and markedly worse on seed 123):
    the DC level of attitude is not a nuisance to be normalised away, it is an
    absolute measurement of the local slope and it carries real information.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    yaws = np.asarray(yaws, dtype=float)
    rolls = np.asarray(rolls, dtype=float)
    pitches = np.asarray(pitches, dtype=float)

    offsets = np.arange(-search_m, search_m + 1e-9, step_m)
    k = len(offsets)
    # Candidate grid, evaluated in one vectorised pass over (candidates, samples)
    # rather than a Python loop per candidate: at 49x49 candidates and ~75
    # samples this is ~50 ms instead of ~1 s, which is what makes it affordable
    # inside a node whose executor also has to keep publishing.
    # Row index is dy, column index is dx - so the flattened candidate order has
    # dy varying slowly (repeat) and dx fast (tile), matching the `grid[i, j]`
    # reshape below. Getting this pair the wrong way round returns (dy, dx) as
    # (dx, dy), which on isotropic terrain still looks like a plausible fix and
    # localises the rover to a mirrored position; the synthetic-offset test in
    # test_terrain_relative.py exists for exactly this.
    dys = np.repeat(offsets, k)
    dxs = np.tile(offsets, k)
    cand_x = xs[None, :] + dxs[:, None]
    cand_y = ys[None, :] + dys[:, None]
    pred_roll, pred_pitch = predicted_attitude(
        gx, gy, world_size_m, resolution_m, cand_x, cand_y, yaws[None, :]
    )
    cost = np.mean(
        (pred_roll - rolls[None, :]) ** 2 + (pred_pitch - pitches[None, :]) ** 2, axis=1
    )
    best = int(np.argmin(cost))
    grid = cost.reshape(k, k)
    i, j = divmod(best, k)

    # Parabolic interpolation about the winner, so the fix is not quantised to
    # the 0.25 m search step. Guarded on a positive denominator: at a plateau or
    # a grid edge the fit is meaningless and the raw cell is used instead.
    def refine(index, along_rows):
        lo = max(index - 1, 0)
        hi = min(index + 1, k - 1)
        c_mid = grid[i, j]
        c_lo = grid[lo, j] if along_rows else grid[i, lo]
        c_hi = grid[hi, j] if along_rows else grid[i, hi]
        denom = c_lo - 2.0 * c_mid + c_hi
        if denom <= 0.0:
            return 0.0
        return float(np.clip(0.5 * (c_lo - c_hi) / denom, -1.0, 1.0)) * step_m

    dy = float(offsets[i]) + refine(i, True)
    dx = float(offsets[j]) + refine(j, False)
    margin = float(np.median(cost) / max(cost[best], 1e-12))
    return dx, dy, margin


def shift_samples(samples, dx: float, dy: float) -> list:
    """Move a buffered window into the frame the filter is about to be in.

    Called after a fix is published. Only the position pair moves - attitude and
    the travelled-distance stamp are frame-independent, and shifting either would
    be a bug that a match still absorbs silently.
    """
    return [
        (x + dx, y + dy, yaw, roll, pitch, travelled)
        for x, y, yaw, roll, pitch, travelled in samples
    ]


def rpy_from_quaternion(q) -> tuple:
    """Roll, pitch, yaw from a geometry_msgs Quaternion (same convention as m4_acceptance)."""
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


class TerrainRelativeNode(Node):
    def __init__(self):
        super().__init__("regolith_terrain_relative")
        self.declare_parameter("manifest_path", "")
        # A window is measured in metres travelled, not seconds: what the match
        # needs is terrain traversed, and this rover's speed varies by an order
        # of magnitude between cruising and grinding through a recovery.
        self.declare_parameter("window_m", 15.0)
        self.declare_parameter("update_m", 6.0)
        self.declare_parameter("search_m", 6.0)
        self.declare_parameter("step_m", 0.25)
        self.declare_parameter("min_speed_mps", 0.03)
        self.declare_parameter("min_samples", 30)
        self.declare_parameter("sample_stride_m", 0.2)
        self.declare_parameter("min_margin", 2.0)
        self.declare_parameter("consistency_m", 1.5)
        self.declare_parameter("position_variance", 1.0)

        self._window_m = float(self.get_parameter("window_m").value)
        self._update_m = float(self.get_parameter("update_m").value)
        self._search_m = float(self.get_parameter("search_m").value)
        self._step_m = float(self.get_parameter("step_m").value)
        self._min_speed = float(self.get_parameter("min_speed_mps").value)
        self._min_samples = int(self.get_parameter("min_samples").value)
        self._stride_m = float(self.get_parameter("sample_stride_m").value)
        self._min_margin = float(self.get_parameter("min_margin").value)
        self._consistency_m = float(self.get_parameter("consistency_m").value)
        self._variance = float(self.get_parameter("position_variance").value)

        manifest_path = Path(self.get_parameter("manifest_path").value)
        try:
            manifest = json.loads(manifest_path.read_text())
            self._gx, self._gy, self._world_m, self._res_m = terrain_gradients(manifest)
        except (OSError, KeyError, ValueError) as error:
            self.get_logger().error(
                f"Failed to load terrain manifest '{manifest_path}': {error!r}. Terrain-relative "
                "navigation cannot run without the a-priori DEM; the launch files generate it via "
                "regolith_terrain_gen."
            )
            raise SystemExit(1)

        # Ring of (x, y, yaw, roll, pitch, travelled_m) in the ESTIMATOR's frame,
        # one per `sample_stride_m` of travel. Distance-spaced rather than
        # time-spaced so a stationary rover cannot flood the window with
        # duplicates of one spot.
        #
        # Publishing a fix makes the EKF jump, and samples already in the window
        # were recorded before that jump - so the buffer is SHIFTED by the
        # published correction in `_attempt_fix` and never straddles one. That is
        # exactly what the offline replay did (`ex[win] + cx`: the whole window
        # moves with the running correction), which is the point - the live node
        # and the validated model have to be the same computation.
        #
        # An earlier version stored these in the wheel-odometry frame instead,
        # reasoning that /odom is never jumped by fusion, and anchored the window
        # onto the current EKF pose with a translation. That was wrong and it cost
        # a live run to find out: /odom and the estimator's frame differ by a
        # ROTATION as well (measured at ~47 deg on seed 123 - odom integrates
        # wheel-derived heading from spawn, the EKF's yaw comes from the IMU), so
        # the matcher was handed a path rotated off the one the rover drove. It
        # then returned a confident, wrong fix - margin 4.66, correction +5.26 m
        # where the true error was 0.15 m - and the run's divergence went to 4 m
        # within seconds of the first published fix. A rotated path still looks
        # like a perfectly good path; nothing in the cost surface can tell.
        self._samples = []
        self._travelled_m = 0.0
        self._ekf_xy = None
        self._last_xy = None
        self._last_fix_at_m = 0.0
        self._last_offset = None
        self._attitude = None
        self._speed = 0.0
        self._fixes_published = 0
        self._fixes_rejected_margin = 0
        self._fixes_rejected_consistency = 0

        self._pub = self.create_publisher(
            PoseWithCovarianceStamped, "/absolute_reference/pose", 10
        )
        self.create_subscription(Imu, "/imu", self._on_imu, 20)
        self.create_subscription(Odometry, "/odom", self._on_odom, 20)
        self.create_subscription(Odometry, "/odometry/filtered", self._on_estimate, 20)
        self.get_logger().info(
            f"Terrain-relative navigation active: {self._window_m:.0f} m window, fix every "
            f"{self._update_m:.0f} m, +-{self._search_m:.0f} m search at {self._step_m:.2f} m, "
            f"DEM {self._res_m:.3f} m/px from {manifest_path}. Onboard sensors only - no ground "
            "truth is read by this node."
        )

    def _on_imu(self, msg: Imu) -> None:
        """Attitude from the IMU, which is what a real rover has; gravity does not drift.

        Only roll and pitch are taken. The IMU's yaw is a gyro integration with
        no absolute reference, and the estimator already fuses it - reading it
        back here would be circular. The heading used for matching comes from
        the EKF, which is the stack's best heading estimate.
        """
        roll, pitch, _ = rpy_from_quaternion(msg.orientation)
        self._attitude = (roll, pitch)

    def _on_odom(self, msg: Odometry) -> None:
        """Wheel odometry, used ONLY to measure distance travelled.

        Its pose is not used for anything else - see the buffer comment in
        __init__ for what happened when it was. Distance comes from here rather
        than from the EKF because a published fix moves the EKF by up to
        `search_m` in one step, and that jump is not travel: counted as travel it
        would bring the next fix forward and make the node fix more often the
        worse it is doing.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self._last_xy is not None:
            self._travelled_m += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)

    def _on_estimate(self, msg: Odometry) -> None:
        if self._attitude is None or self._last_xy is None:
            return
        _, _, yaw = rpy_from_quaternion(msg.pose.pose.orientation)
        self._speed = abs(msg.twist.twist.linear.x)
        self._ekf_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        x, y = self._ekf_xy

        # Only sample while genuinely driving. A wedged or reversing rover's
        # attitude is set by the boulder it is on, not by the terrain the DEM
        # knows about, and those samples are noise in the match, not signal.
        if self._speed < self._min_speed:
            return
        if self._samples and self._travelled_m - self._samples[-1][5] < self._stride_m:
            return
        roll, pitch = self._attitude
        self._samples.append((x, y, yaw, roll, pitch, self._travelled_m))
        while self._samples and self._travelled_m - self._samples[0][5] > self._window_m:
            self._samples.pop(0)

        if self._travelled_m - self._last_fix_at_m < self._update_m:
            return
        if len(self._samples) < self._min_samples:
            return
        self._last_fix_at_m = self._travelled_m
        self._attempt_fix(msg)

    def _attempt_fix(self, msg: Odometry) -> None:
        arr = np.array(self._samples, dtype=float)
        dx, dy, margin = match_offset(
            self._gx,
            self._gy,
            self._world_m,
            self._res_m,
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],
            arr[:, 3],
            arr[:, 4],
            search_m=self._search_m,
            step_m=self._step_m,
        )

        if margin < self._min_margin:
            self._fixes_rejected_margin += 1
            self.get_logger().info(
                f"Terrain fix rejected: margin {margin:.2f} < {self._min_margin:.2f} - this "
                f"stretch of terrain does not determine position ({len(self._samples)} samples "
                f"over {self._window_m:.0f} m)"
            )
            self._last_offset = (dx, dy)
            return

        # Two independent windows have to agree before the filter is told
        # anything. A single window's gross mismatch - a crater rim that looks
        # like the next crater rim - is the failure mode that would actively harm
        # the estimate, and on the recorded runs this gate cut the p90 error from
        # 3.91 m to 1.56 m for the cost of rejecting a third of the fixes.
        if self._last_offset is not None:
            disagreement = math.hypot(dx - self._last_offset[0], dy - self._last_offset[1])
            if disagreement > self._consistency_m:
                self._fixes_rejected_consistency += 1
                self.get_logger().info(
                    f"Terrain fix rejected: disagrees with the previous window by "
                    f"{disagreement:.2f} m (> {self._consistency_m:.2f}); offset now "
                    f"({dx:+.2f}, {dy:+.2f}), was ({self._last_offset[0]:+.2f}, "
                    f"{self._last_offset[1]:+.2f})"
                )
                self._last_offset = (dx, dy)
                return
        else:
            # Nothing to agree with yet. Hold the first fix back rather than
            # publish an ungated one into a filter that has no defence against it.
            self._last_offset = (dx, dy)
            self.get_logger().info(
                f"Terrain fix held: first window, offset ({dx:+.2f}, {dy:+.2f}), margin "
                f"{margin:.2f} - waiting for a second window to confirm"
            )
            return

        self._last_offset = (dx, dy)
        # Move the buffered window into the frame the filter is about to be in, so
        # the next match sees one continuous path rather than one with a step in
        # it. The EKF blends rather than jumping the whole way, so this over- or
        # under-shoots slightly; that residual is a placement error the next fix
        # simply measures again, which is the behaviour the replay validated.
        self._samples = shift_samples(self._samples, dx, dy)
        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "odom"  # the frame the EKF estimates in
        out.pose.pose.orientation = msg.pose.pose.orientation
        out.pose.pose.position.x = self._ekf_xy[0] + dx
        out.pose.pose.position.y = self._ekf_xy[1] + dy
        out.pose.pose.position.z = msg.pose.pose.position.z
        covariance = [0.0] * 36
        covariance[0] = self._variance
        covariance[7] = self._variance
        covariance[14] = 1e6  # z, roll, pitch, yaw: not observed by terrain matching
        covariance[21] = 1e6
        covariance[28] = 1e6
        covariance[35] = 1e6
        out.pose.covariance = covariance
        self._pub.publish(out)
        self._fixes_published += 1
        self.get_logger().info(
            f"Terrain fix #{self._fixes_published} at {self._travelled_m:.1f} m travelled: "
            f"correction ({dx:+.2f}, {dy:+.2f}) m, margin {margin:.2f}, {len(self._samples)} "
            f"samples (rejected so far: {self._fixes_rejected_margin} ambiguous, "
            f"{self._fixes_rejected_consistency} inconsistent)"
        )


def main() -> None:
    rclpy.init()
    node = TerrainRelativeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
