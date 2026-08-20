#!/usr/bin/env python3
# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Stops a wedged rover's spinning wheels from corrupting the EKF.

The failure this exists for, measured live over a full run (PROGRESS.md, res40
M4 acceptance):

    phase                        ground truth moved   EKF believed
    driving normally                     tracks          tracks       (0.17 m apart)
    wedged on a boulder, 270 s           0.94 m          4.67 m
    driving again                        8.64 m          8.65 m       (4.29 m apart, forever)

While the chassis is pinned the wheels keep turning, so wheel odometry
integrates distance that never happened. The EKF fuses wheel odometry and IMU
only - nothing in the stack ever observes absolute position - so the error is
permanent: once the rover breaks free the two traces move in lockstep again,
4.3 m apart, for the rest of the run. Three acceptance runs ended 17-36 m from
their goals while reporting "Goal reached (within 1.50 m)", because the arrival
check lives in the same corrupted frame.

The fix is a zero-velocity update (ZUPT), the standard treatment for exactly
this in wheeled/inertial odometry: while the vehicle is judged to be slipping
in place, feed the estimator a measured zero velocity instead of the wheels'
claim.

WHAT THIS MAY USE, AND WHAT IT MAY NOT. Detection here is onboard-only - wheel
odometry and the IMU, both of which a real rover carries. It deliberately does
NOT use /ground_truth/pose, even though the stuck detector in
flip_recovery_node.py does (that one is a simulation backstop and is labelled
as such). A localization fix that consulted ground truth would be measuring
itself against the answer key: the whole point is that the estimate has no
absolute reference, and an oracle-fed ZUPT would make M4's numbers meaningless.

THE DETECTOR. Over a sliding 15 s window it compares what the wheels claim
against what the IMU shows the body actually did, and declares slip on either
of two independent signatures.

  1. ROTATION THE GYRO NEVER SEES (the one that fires in practice). A rover
     wedged against a boulder is usually still being commanded to turn, so its
     wheels spin differentially and wheel odometry integrates yaw that never
     happens. The gyro measures the yaw that did happen. This is the measured
     separation over a recorded seed-42 wedge, gyro-observed rotation as a
     fraction of the wheels' claim:

         slipping windows (n=2075)     0.082 .. 0.124
         honest driving   (n=7014)     0.169 .. 1.73   (median 0.85)

     Two disjoint bands with a gap between them, so the threshold sits in the
     gap at 0.145. Note the honest-driving floor is 0.17, not 1.0: skid-steer
     wheel odometry always over-claims rotation because turning requires the
     wheels to scrub (PROGRESS.md M3 measured it ~3x off). The test is
     therefore about the SIZE of a disagreement that is always present, which
     is why the threshold had to be measured rather than reasoned about.

  2. A RIGIDLY STILL BODY while the wheels claim distance - attitude span
     <= 0.010 rad and gyro RMS <= 0.005 rad/s. This covers the original
     wheels-locked-static case, where nothing is commanded to turn and
     signature 1 has nothing to work with. Reported honestly: on the recorded
     run this path fired on 0 of 7,755 genuine driving windows (so it is not a
     false-positive risk) and also on 0 of 968 slipping ones (so it has never
     yet caught a real event either). It is kept because it covers a
     physically different failure, not because it has been shown to work.

The first design of this detector used signature 2 alone, on the reasoning
that a pinned rover cannot tilt. Recorded data refuted it: during the real
wedge the chassis bucks against the boulder while the wheels spin, spanning
0.119-0.195 rad of attitude - MORE than the median driving window. The
"pinned means still" intuition was wrong, and only measuring it showed that.

WHY THE WINDOW IS 15 SECONDS. An IMU cannot tell constant velocity from rest -
Galilean invariance, not a tuning problem - so any test of this kind needs
enough time for a real disagreement to accumulate. Attitude span over genuine
driving, by window length, from the same recording:

    window     min attitude span over genuinely-driving windows
      3 s      0.0000 rad   (p5 0.0007)   <- no threshold separates
      6 s      0.0009 rad   (p5 0.0087)
     10 s      0.0080 rad   (p5 0.0269)
     15 s      0.0276 rad   (p5 0.0307)   <- 2.8x above the threshold used

At 3 s, signature 2 fired on 29% of genuinely-driving windows. The cost of the
longer window is detection delay, and it is small against the failure it
prevents: the measured episode accumulated 4.3 m of permanent error.

Thresholds are calibrated against recorded runs rather than guessed - see
scripts/calibrate_slip_detector.py, test_wheel_slip_detector.py, and
PROGRESS.md. One recorded wedge on one seed is the evidence base; that is
stated as a limit, not dressed up as validation.
"""

from collections import deque
import math

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool


def _roll_pitch_yaw(q) -> tuple:
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


def _unwrap(previous: float, current: float) -> float:
    """Continuous yaw, so a wrap at +-pi doesn't look like a huge attitude change."""
    return previous + math.atan2(math.sin(current - previous), math.cos(current - previous))


class SlipDetector:
    """Pure logic, no ROS - so it can be tested against recorded runs.

    Feed it samples; ask it whether the wheels are currently slipping in place.
    """

    def __init__(
        self,
        window_s: float = 15.0,
        min_claimed_distance_m: float = 0.25,
        min_claimed_rotation_rad: float = 0.5,
        rotation_ratio: float = 0.145,
        release_rotation_ratio: float = 0.25,
        release_window_s: float = 5.0,
        max_attitude_span_rad: float = 0.010,
        max_gyro_rms_rps: float = 0.005,
    ):
        self.window_s = window_s
        self.min_claimed_distance_m = min_claimed_distance_m
        self.min_claimed_rotation_rad = min_claimed_rotation_rad
        self.rotation_ratio = rotation_ratio
        self.release_rotation_ratio = release_rotation_ratio
        self.release_window_s = release_window_s
        self.max_attitude_span_rad = max_attitude_span_rad
        self.max_gyro_rms_rps = max_gyro_rms_rps
        self._samples = deque()  # (t, vx, wz_wheel, wz_gyro, roll, pitch, yaw_unwrapped)
        self.dropped_out_of_order = 0
        self._yaw = None

    def add(self, t: float, vx: float, wz_wheel: float, wz_gyro: float, rpy: tuple) -> None:
        roll, pitch, yaw = rpy
        self._yaw = yaw if self._yaw is None else _unwrap(self._yaw, yaw)
        # Drop anything that does not move time forward. features() integrates
        # over consecutive pairs with dt = t1 - t0, so one out-of-order sample
        # makes dt negative and every accumulator wrong - including the sum of
        # SQUARES, which then goes negative and takes math.sqrt with it. That is
        # not hypothetical: it killed an M4 acceptance run 30 s in with
        # "ValueError: math domain error", and because this node is required,
        # the whole launch went down with it. Guarding here rather than at each
        # accumulator fixes all of them at once, and a stale duplicate carries
        # no information worth keeping anyway.
        if self._samples and t <= self._samples[-1][0]:
            self.dropped_out_of_order += 1
            return
        self._samples.append((t, vx, wz_wheel, wz_gyro, roll, pitch, self._yaw))
        while self._samples and t - self._samples[0][0] > self.window_s:
            self._samples.popleft()

    def features(self, window_s: float = None) -> dict:
        """None until the window is full enough to mean anything.

        `window_s` shortens the window to its most recent slice, used for
        RELEASING slip: the 15 s declaration window necessarily still contains
        the wedge's own history for 15 s after the rover breaks free, and
        holding a zero-velocity update over a rover that is really driving
        loses real distance - the same corruption the ZUPT exists to prevent,
        in the other direction.
        """
        window_s = self.window_s if window_s is None else window_s
        samples = self._samples
        if window_s < self.window_s:
            cutoff = samples[-1][0] - window_s if samples else 0.0
            samples = [s for s in samples if s[0] >= cutoff]
        if len(samples) < 5:
            return None
        span_s = samples[-1][0] - samples[0][0]
        if span_s < window_s * 0.5:
            return None

        claimed_distance = 0.0
        claimed_rotation = 0.0
        observed_rotation = 0.0
        gyro_sq = 0.0
        for (t0, vx0, wheel0, gyro0, *_), (t1, *_rest) in zip(samples, list(samples)[1:]):
            dt = t1 - t0
            claimed_distance += abs(vx0) * dt
            claimed_rotation += abs(wheel0) * dt
            observed_rotation += abs(gyro0) * dt
            gyro_sq += gyro0 * gyro0 * dt
        rolls = [s[4] for s in samples]
        pitches = [s[5] for s in samples]
        yaws = [s[6] for s in samples]
        return {
            "span_s": span_s,
            "claimed_distance_m": claimed_distance,
            "claimed_rotation_rad": claimed_rotation,
            "observed_rotation_rad": observed_rotation,
            # max(..., 0) is not redundant with the monotonicity guard in add():
            # it is the difference between this node degrading and this node
            # killing the whole launch, and it costs nothing.
            "gyro_rms_rps": math.sqrt(max(gyro_sq, 0.0) / span_s) if span_s > 0 else 0.0,
            "attitude_span_rad": max(
                max(rolls) - min(rolls), max(pitches) - min(pitches), max(yaws) - min(yaws)
            ),
        }

    def _rotation_disagrees(self, f: dict) -> bool:
        """The wheels claim a lot of turning; the gyro barely sees any of it."""
        if f["claimed_rotation_rad"] < self.min_claimed_rotation_rad:
            return False
        return f["observed_rotation_rad"] <= self.rotation_ratio * f["claimed_rotation_rad"]

    def _body_is_rigid(self, f: dict) -> bool:
        """Nothing the IMU can see moved at all, in any axis."""
        return (
            f["attitude_span_rad"] <= self.max_attitude_span_rad
            and f["gyro_rms_rps"] <= self.max_gyro_rms_rps
        )

    def slipping(self) -> bool:
        f = self.features()
        if f is None:
            return False
        if f["claimed_distance_m"] < self.min_claimed_distance_m:
            return False  # no phantom distance to suppress
        return self._rotation_disagrees(f) or self._body_is_rigid(f)

    def clearing(self) -> bool:
        """True when it is safe to say the slip episode has ended.

        Judged on the RECENT slice (release_window_s), not the full declaration
        window, which still contains the wedge for 15 s after the rover breaks
        free. Deliberately not just `not slipping()` either: the ratio has to
        recover past a looser threshold (release_rotation_ratio, 0.25) than the
        one that declared slip (rotation_ratio, 0.145), so a statistic wobbling
        around the boundary cannot flicker the ZUPT on and off.
        """
        f = self.features(self.release_window_s)
        if f is None:
            return False  # not enough recent evidence to justify releasing

        scale = self.release_window_s / self.window_s
        if f["claimed_distance_m"] < self.min_claimed_distance_m * scale:
            # The wheels have gone quiet, so there is nothing for the gate to
            # suppress either way - stay latched rather than release on the
            # absence of evidence. Releasing here was a real bug: the full
            # window still saw the disagreement, so the state flipped back on
            # the very next message and the gate flickered at the /odom rate
            # (24 declare/clear pairs in 2 s, observed live).
            return False

        still_disagreeing = (
            f["claimed_rotation_rad"] >= self.min_claimed_rotation_rad * scale
            and f["observed_rotation_rad"]
            <= self.release_rotation_ratio * f["claimed_rotation_rad"]
        )
        return not (still_disagreeing or self._body_is_rigid(f))


class WheelSlipNode(Node):
    def __init__(self):
        super().__init__("regolith_wheel_slip")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("gated_topic", "/odom/gated")
        self.declare_parameter("window_s", 15.0)
        self.declare_parameter("min_claimed_distance_m", 0.25)
        self.declare_parameter("min_claimed_rotation_rad", 0.5)
        self.declare_parameter("rotation_ratio", 0.145)
        self.declare_parameter("release_rotation_ratio", 0.25)
        self.declare_parameter("release_window_s", 5.0)
        # Floor on how fast the gate may toggle, in simulated seconds. Belt and
        # braces on top of clearing()'s hysteresis: a gate that chatters is
        # neither suppressing nor passing the velocity cleanly, and it buries
        # the log.
        self.declare_parameter("min_dwell_s", 2.0)
        self.declare_parameter("max_attitude_span_rad", 0.010)
        self.declare_parameter("max_gyro_rms_rps", 0.005)

        self._detector = SlipDetector(
            window_s=self.get_parameter("window_s").value,
            min_claimed_distance_m=self.get_parameter("min_claimed_distance_m").value,
            min_claimed_rotation_rad=self.get_parameter("min_claimed_rotation_rad").value,
            rotation_ratio=self.get_parameter("rotation_ratio").value,
            release_rotation_ratio=self.get_parameter("release_rotation_ratio").value,
            release_window_s=self.get_parameter("release_window_s").value,
            max_attitude_span_rad=self.get_parameter("max_attitude_span_rad").value,
            max_gyro_rms_rps=self.get_parameter("max_gyro_rms_rps").value,
        )
        self._imu = None  # (wz, rpy)
        self._slipping = False
        self._slip_declared_at = None
        self._slip_events = 0
        self._zupt_messages = 0
        self._suppressed_m = 0.0  # phantom distance this node kept out of the EKF
        self._last_stamp = None

        self.create_subscription(Imu, "/imu", self._on_imu, 20)
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self._on_odom, 20
        )
        self._pub = self.create_publisher(Odometry, self.get_parameter("gated_topic").value, 20)
        self._slip_pub = self.create_publisher(Bool, "/wheel_slip", 10)
        self.get_logger().info(
            f"Wheel-slip ZUPT armed: {self.get_parameter('odom_topic').value} -> "
            f"{self.get_parameter('gated_topic').value} (onboard signals only - "
            "wheel odometry and IMU, never ground truth)"
        )

    def _on_imu(self, msg: Imu) -> None:
        self._imu = (msg.angular_velocity.z, _roll_pitch_yaw(msg.orientation))

    def _on_odom(self, msg: Odometry) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z

        if self._imu is not None:
            self._detector.add(stamp, vx, wz, self._imu[0], self._imu[1])
            self._update_state(vx, stamp)

        out = msg
        if self._slipping:
            out.twist.twist.linear.x = 0.0
            out.twist.twist.linear.y = 0.0
            out.twist.twist.angular.z = 0.0
            self._zupt_messages += 1
        self._pub.publish(out)

    def _update_state(self, vx: float, stamp: float) -> None:
        was_slipping = self._slipping
        if self._slipping:
            dwell = self.get_parameter("min_dwell_s").value
            held_long_enough = (
                self._slip_declared_at is None or (stamp - self._slip_declared_at) >= dwell
            )
            if held_long_enough and self._detector.clearing():
                self._slipping = False
        elif self._detector.slipping():
            self._slipping = True
            self._slip_declared_at = stamp

        if self._slipping and self._last_stamp is not None:
            self._suppressed_m += abs(vx) * max(0.0, stamp - self._last_stamp)
        self._last_stamp = stamp

        if self._slipping != was_slipping:
            self._slip_pub.publish(Bool(data=self._slipping))
            f = self._detector.features() or {}
            claimed_rotation = f.get("claimed_rotation_rad", 0.0)
            observed_rotation = f.get("observed_rotation_rad", 0.0)
            ratio = (
                observed_rotation / claimed_rotation if claimed_rotation > 1e-6 else float("nan")
            )
            if self._slipping:
                self._slip_events += 1
                self.get_logger().warn(
                    f"WHEEL SLIP #{self._slip_events}: over the last {f.get('span_s', 0.0):.1f} s "
                    f"the wheels claim {f.get('claimed_distance_m', 0.0):.2f} m and "
                    f"{claimed_rotation:.2f} rad of turning; the gyro saw "
                    f"{observed_rotation:.2f} rad ({ratio:.0%} of it) and the attitude spanned "
                    f"{math.degrees(f.get('attitude_span_rad', 0.0)):.2f} deg. Feeding the EKF a "
                    "zero-velocity update instead of the wheels' claim."
                )
            else:
                # Report the RELEASE window's figures, not the declaration
                # window's: the decision to release is made on the recent slice,
                # and printing the 15 s numbers made the log read as though slip
                # had been released while the gyro still corroborated only 6%.
                r = self._detector.features(self._detector.release_window_s) or {}
                r_claimed = r.get("claimed_rotation_rad", 0.0)
                r_observed = r.get("observed_rotation_rad", 0.0)
                r_ratio = r_observed / r_claimed if r_claimed > 1e-6 else float("nan")
                self.get_logger().warn(
                    f"WHEEL SLIP #{self._slip_events} cleared - over the last "
                    f"{r.get('span_s', 0.0):.1f} s the wheels claimed "
                    f"{r_claimed:.2f} rad of turning and the gyro corroborated "
                    f"{r_ratio:.0%} of it. Phantom distance kept out of the EKF so far: "
                    f"{self._suppressed_m:.2f} m."
                )


def main() -> None:
    rclpy.init()
    node = WheelSlipNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.get_logger().info(
        f"Wheel-slip summary: {node._slip_events} slip episodes, {node._zupt_messages} "
        f"zero-velocity updates, {node._suppressed_m:.2f} m of phantom wheel distance "
        "kept out of the EKF"
    )
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
