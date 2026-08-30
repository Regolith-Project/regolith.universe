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
from typing import NamedTuple

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


class Sample(NamedTuple):
    """One buffered observation. Named because the tuple indices were a live bug.

    Renumbering this buffer once already left a stale `[5]` in the stride check
    that would have raised on the second sample of the next run; the fields are
    named now so a reader and the interpreter both catch that instead of neither.

    The displacement is carried as a world-frame VECTOR rather than as a distance
    to be re-projected along a heading later. Re-projecting is wrong twice over
    on this rover: it lays reverse motion (every escape manoeuvre) forwards, and
    it assumes the heading was constant across the step. Differencing the node's
    own track costs nothing and is exact for both.
    """

    dx_m: float            # world displacement since the previous sample
    dy_m: float
    yaw: float             # world heading, from the estimator (IMU-driven)
    roll: float            # measured attitude, from the IMU
    pitch: float
    travelled_m: float     # odometer reading at this sample


def reconstruct_window(dxs, dys, anchor_x: float, anchor_y: float):
    """Lay the recent path out behind `anchor`, from per-sample world displacements.

    `dxs[i]`/`dys[i]` are the displacement between sample i-1 and i (index 0 is
    ignored). Returns (xs, ys) with the NEWEST sample sitting exactly on the
    anchor.

    This is what makes the loop stable, and it took three live failures to get
    here. The window has to be expressed relative to the CURRENT estimate:

    - Buffering absolute estimator positions and shifting them by each published
      correction over-shifts (the filter absorbs only part of what it is told),
      the match then reports ~zero, and the node certifies a wrong position as
      correct once a second.
    - Buffering them and NOT shifting leaves the window stale, so the same
      correction is measured and re-applied every tick and the estimate runs away
      geometrically - seed 123 reached 1432 m.

    Anchoring sidesteps both: when the filter moves, the whole window moves with
    it for free, so each match measures the residual that is actually left. The
    shape comes from the odometer and the IMU's heading, neither of which the fix
    can feed back into - wheel-integrated HEADING would be no good over this
    distance (43 m of error over 107 m in M4's error budget, against 3.5 m for
    the IMU's), but wheel-integrated DISTANCE is accurate to about 1% once the
    slip gate is in, and distance is all that is taken from it.

    Accumulate segment by segment. Multiplying a total backward distance by each
    sample's own heading is only correct on a dead-straight path and silently
    mangles every turn - it scored 3.19 m against 0.42 m in replay.
    """
    rel_x = np.concatenate([[0.0], np.cumsum(np.asarray(dxs, dtype=float)[1:])])
    rel_y = np.concatenate([[0.0], np.cumsum(np.asarray(dys, dtype=float)[1:])])
    return anchor_x + rel_x - rel_x[-1], anchor_y + rel_y - rel_y[-1]


def clamp_correction(dx: float, dy: float, max_step_m: float) -> tuple:
    """Limit how far one fix may ask the filter to move.

    A fix is a slow pull, not a teleport. Unlimited, a wrong lock compounds: in
    replay the worst run ended 37.4 m out and six of seven seed-7 runs took an
    ~11.7 m excursion mid-run. Capped, the same runs stay under 8.8 m and every
    one of the 25 improves. Genuine error is still corrected - just over tens of
    seconds instead of instantly, which is fast against drift that took minutes
    to accumulate.
    """
    if max_step_m <= 0.0:
        return dx, dy
    magnitude = math.hypot(dx, dy)
    if magnitude <= max_step_m:
        return dx, dy
    scale = max_step_m / magnitude
    return dx * scale, dy * scale


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
        # Publishing is TIME-paced, not distance-paced, and that is the whole
        # lesson of the first live run. A fix every 6 m of travel is a fix every
        # several MINUTES on this rover, and a single isolated absolute update
        # left the EKF's velocity state corrupted with nothing absolute to
        # contradict it until the next one - divergence went 0.17 m -> 29.6 m
        # from one 0.86 m correction. The oracle publishes at ~1 Hz and never
        # showed this: a dense stream gives an injected error no room to
        # integrate. See PROGRESS.md.
        #
        # Density costs accuracy nothing and buys stability: in the replay,
        # matching every 0.25 / 0.5 / 1 / 2 / 3 m all land at 0.33-0.42 m median
        # against 0.77 m at 6 m, and the curve is flat below ~2 m because the
        # window barely changes. So the rate is chosen for the filter, not for
        # the matcher.
        self.declare_parameter("publish_hz", 1.0)
        self.declare_parameter("search_m", 6.0)
        self.declare_parameter("step_m", 0.25)
        self.declare_parameter("min_speed_mps", 0.03)
        self.declare_parameter("min_samples", 30)
        self.declare_parameter("sample_stride_m", 0.2)
        self.declare_parameter("min_margin", 2.0)
        self.declare_parameter("consistency_m", 1.5)
        # How far one publish may ask the filter to move. See clamp_correction.
        self.declare_parameter("max_step_m", 0.10)
        # Wider than the matcher's measured 0.8 m accuracy, and deliberately so:
        # at 1 Hz over a 15 m window, consecutive fixes share almost all of their
        # samples, so they are nowhere near the independent measurements the
        # filter assumes. Publishing 0.8 m sigma sixty times a minute would make
        # the filter far more certain than the evidence supports. 2.0 m sigma is
        # a judgement about that correlation, not a measurement of it - it keeps
        # the anchor firm while making any single fix unable to yank the
        # estimate, which is what went wrong live.
        self.declare_parameter("position_variance", 4.0)

        self._window_m = float(self.get_parameter("window_m").value)
        self._publish_hz = float(self.get_parameter("publish_hz").value)
        self._search_m = float(self.get_parameter("search_m").value)
        self._step_m = float(self.get_parameter("step_m").value)
        self._min_speed = float(self.get_parameter("min_speed_mps").value)
        self._min_samples = int(self.get_parameter("min_samples").value)
        self._stride_m = float(self.get_parameter("sample_stride_m").value)
        self._min_margin = float(self.get_parameter("min_margin").value)
        self._consistency_m = float(self.get_parameter("consistency_m").value)
        self._max_step_m = float(self.get_parameter("max_step_m").value)
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
        self._track = None      # this node's own absolute position estimate
        self._yaw = None
        self._last_odom_stamp = None
        self._pending_dx = 0.0   # track displacement since the last buffered sample
        self._pending_dy = 0.0
        self._last_xy = None
        self._last_offset = None
        self._last_offset_at_m = None
        self._last_orientation = None
        self._last_stamp = None
        self._attitude = None
        self._odom_vx = 0.0
        self._fixes_published = 0
        self._fixes_rejected_margin = 0
        self._fixes_rejected_consistency = 0
        self._report_every_m = 5.0
        self._last_report_m = -1e9

        self._pub = self.create_publisher(
            PoseWithCovarianceStamped, "/absolute_reference/pose", 10
        )
        self.create_subscription(Imu, "/imu", self._on_imu, 20)
        self.create_subscription(Odometry, "/odom", self._on_odom, 20)
        self.create_subscription(Odometry, "/odometry/filtered", self._on_estimate, 20)
        self.create_timer(1.0 / max(self._publish_hz, 1e-3), self._tick)
        self.get_logger().info(
            f"Terrain-relative navigation active: {self._window_m:.0f} m window, fixing at "
            f"{self._publish_hz:.1f} Hz, +-{self._search_m:.0f} m search at {self._step_m:.2f} m, "
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

    def _advance_track(self, step_m: float) -> None:
        """Dead-reckon this node's OWN absolute position forward.

        The node keeps its own track and never reads the filter's corrected pose
        back into it. That independence is the whole point, and getting it wrong
        wrecked three live runs in a row: publishing `filter_pose + offset` is not
        an absolute measurement at all, it is a target defined relative to the
        estimate, so it sits a fixed distance ahead of wherever the filter has
        got to. The filter chases it, reads the pursuit as velocity, and
        accelerates - measured at 0.85 m/s of estimated motion while the wheels
        reported 0.045 m/s. A carrot on a stick.

        The offline replay never had this failure because its track is the raw
        OPEN-LOOP trajectory plus accumulated corrections - it never reads back a
        corrected estimate. This method is the live equivalent of that, and the
        node is now a small standalone terrain-aided dead-reckoner whose output
        happens to be fused by an EKF downstream.
        """
        if self._track is None or self._yaw is None:
            return
        self._track = (
            self._track[0] + step_m * math.cos(self._yaw),
            self._track[1] + step_m * math.sin(self._yaw),
        )

    def _on_odom(self, msg: Odometry) -> None:
        """Wheel odometry, used ONLY to measure distance travelled.

        Its pose is not used for anything else - see the buffer comment in
        __init__ for what happened when it was. Distance comes from here rather
        than from the EKF because a published fix moves the EKF by up to
        `search_m` in one step, and that jump is not travel: counted as travel it
        would bring the next fix forward and make the node fix more often the
        worse it is doing.
        """
        # Integrate the SIGNED body-forward velocity rather than differencing the
        # odometry pose. Two reasons, both load-bearing: the pose difference is a
        # magnitude, so it lays every reverse leg of an escape manoeuvre forwards,
        # and /odom's own frame is rotated relative to the world (measured at 47
        # degrees on seed 123), so its deltas cannot be used as world vectors
        # without a rotation that is itself drifting.
        # Gate sampling on the WHEELS' speed, not the filter's. The filter's
        # velocity is a fused quantity this node's own output feeds into, so
        # gating on it would be one more way to read our own answer back; the
        # offline validation used odom_vx and the node should be the same
        # computation, not nearly the same one.
        self._odom_vx = msg.twist.twist.linear.x
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_odom_stamp is not None:
            dt = stamp - self._last_odom_stamp
            if 0.0 < dt < 1.0:
                step_m = msg.twist.twist.linear.x * dt
                before = self._track
                self._advance_track(step_m)
                if before is not None and self._track is not None:
                    self._pending_dx += self._track[0] - before[0]
                    self._pending_dy += self._track[1] - before[1]
                self._travelled_m += abs(step_m)
        self._last_odom_stamp = stamp
        self._last_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _on_estimate(self, msg: Odometry) -> None:
        if self._attitude is None or self._last_xy is None:
            return
        _, _, yaw = rpy_from_quaternion(msg.pose.pose.orientation)
        self._yaw = yaw
        # The filter's pose seeds this node's track ONCE, before any fix has been
        # published and so before the filter can have been influenced by one.
        # After that the track is the node's own and the filter's pose is not
        # read again - see _advance_track.
        if self._track is None:
            self._track = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        # Only sample while genuinely driving. A wedged or reversing rover's
        # attitude is set by the boulder it is on, not by the terrain the DEM
        # knows about, and those samples are noise in the match, not signal.
        if abs(self._odom_vx) < self._min_speed:
            return
        if self._samples and self._travelled_m - self._samples[-1].travelled_m < self._stride_m:
            return
        roll, pitch = self._attitude
        self._samples.append(
            Sample(self._pending_dx, self._pending_dy, yaw, roll, pitch, self._travelled_m)
        )
        self._pending_dx = self._pending_dy = 0.0
        while self._samples and self._travelled_m - self._samples[0].travelled_m > self._window_m:
            self._samples.pop(0)
        self._last_orientation = msg.pose.pose.orientation
        self._last_stamp = msg.header.stamp

    def _tick(self) -> None:
        """Re-match and publish on a clock, not on distance travelled.

        Re-running the match rather than republishing a stored offset is what
        keeps this from running away: each publish is `current estimate + freshly
        matched offset`, so as the filter converges onto the matched position the
        offset shrinks to nothing on its own. Republishing a remembered
        correction against an estimate that has already absorbed it would add it
        twice, and keep adding it.

        The window barely changes between ticks, so this is mostly the same
        measurement over and over - which is the point. It is an anchor, and its
        correlation is paid for in `position_variance` above.
        """
        if self._last_stamp is None or self._track is None:
            return
        if len(self._samples) < self._min_samples:
            return
        span_m = self._samples[-1].travelled_m - self._samples[0].travelled_m
        if span_m < self._window_m * 0.5:
            return
        self._attempt_fix()

    def _attempt_fix(self) -> None:
        arr = np.array(self._samples, dtype=float)
        # Lay the window out behind wherever the filter currently believes it is,
        # so a correction the filter absorbs moves the window with it.
        xs, ys = reconstruct_window(arr[:, 0], arr[:, 1], self._track[0], self._track[1])
        yaws, rolls, pitches = arr[:, 2], arr[:, 3], arr[:, 4]
        dx, dy, margin = match_offset(
            self._gx,
            self._gy,
            self._world_m,
            self._res_m,
            xs,
            ys,
            yaws,
            rolls,
            pitches,
            search_m=self._search_m,
            step_m=self._step_m,
        )

        if margin < self._min_margin:
            self._fixes_rejected_margin += 1
            self.get_logger().info(
                f"Terrain fix rejected: margin {margin:.2f} < {self._min_margin:.2f} - this "
                f"stretch of terrain does not determine position ({len(self._samples)} samples "
                f"over {self._window_m:.0f} m)",
                throttle_duration_sec=30.0,
            )
            self._last_offset = (dx, dy)
            self._last_offset_at_m = self._travelled_m
            return

        # Two INDEPENDENT windows have to agree before the filter is told
        # anything. A single window's gross mismatch - a crater rim that looks
        # like the next crater rim - is the failure mode that would actively harm
        # the estimate, and on the recorded runs this gate cut the p90 error from
        # 3.91 m to 1.56 m for the cost of rejecting a third of the fixes.
        #
        # "Independent" is doing real work now that this runs on a clock. At 1 Hz
        # consecutive windows share every sample but the newest, so they agree
        # trivially and comparing them would be a gate that always passes - it
        # would have been vacuous exactly where the previous version's version of
        # it already failed, letting through a wrong answer that MOVED smoothly.
        # So the comparison is against the last offset from at least half a
        # window of travel ago, which is the last one built on substantially
        # different terrain.
        independent_m = self._window_m * 0.5
        if (self._last_offset is not None and self._last_offset_at_m is not None
                and self._travelled_m - self._last_offset_at_m < independent_m):
            pass  # too soon to be a second opinion; keep the older one to compare against
        elif self._last_offset is not None:
            disagreement = math.hypot(dx - self._last_offset[0], dy - self._last_offset[1])
            if disagreement > self._consistency_m:
                self._fixes_rejected_consistency += 1
                self.get_logger().info(
                    f"Terrain fix rejected: disagrees with the previous window by "
                    f"{disagreement:.2f} m (> {self._consistency_m:.2f}); offset now "
                    f"({dx:+.2f}, {dy:+.2f}), was ({self._last_offset[0]:+.2f}, "
                    f"{self._last_offset[1]:+.2f})",
                    throttle_duration_sec=30.0,
                )
                self._last_offset = (dx, dy)
                self._last_offset_at_m = self._travelled_m
                return
            self._last_offset = (dx, dy)
            self._last_offset_at_m = self._travelled_m
        else:
            # Nothing to agree with yet. Hold the first fix back rather than
            # publish an ungated one into a filter that has no defence against it.
            self._last_offset = (dx, dy)
            self._last_offset_at_m = self._travelled_m
            self.get_logger().info(
                f"Terrain fix held: first window, offset ({dx:+.2f}, {dy:+.2f}), margin "
                f"{margin:.2f} - waiting for an independent window to confirm"
            )
            return

        dx, dy = clamp_correction(dx, dy, self._max_step_m)
        # Correct this node's own track. Nothing here consults the filter.
        self._track = (self._track[0] + dx, self._track[1] + dy)
        out = PoseWithCovarianceStamped()
        out.header.stamp = self._last_stamp
        out.header.frame_id = "odom"  # the frame the EKF estimates in
        out.pose.pose.orientation = self._last_orientation
        out.pose.pose.position.x = self._track[0]
        out.pose.pose.position.y = self._track[1]
        out.pose.pose.position.z = 0.0
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
        # Summarise rather than log every fix: at 1 Hz the per-fix line would be
        # ~3600 entries an hour in a launch log that several analyses grep.
        if self._travelled_m - self._last_report_m >= self._report_every_m:
            self._last_report_m = self._travelled_m
            self.get_logger().info(
                f"Terrain fix #{self._fixes_published} at {self._travelled_m:.1f} m travelled: "
                f"correction ({dx:+.2f}, {dy:+.2f}) m, margin {margin:.2f}, "
                f"{len(self._samples)} samples (rejected so far: "
                f"{self._fixes_rejected_margin} ambiguous, "
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
