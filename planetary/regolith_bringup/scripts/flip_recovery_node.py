#!/usr/bin/env python3
# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Simulated flip recovery for the Regolith demo.

The primary flip fix is preventive: the terrain collision geometry now uses
tilted "shingle" slabs instead of flat-topped boxes, which collapses the
vertical cliffs at cell boundaries that used to kick the rover into a roll
(see regolith_terrain_gen/heightmap.py and PROGRESS.md M4/M5). This node is
the honest backstop for the rare residual case: a wheeled rover cannot
physically self-right, so instead of leaving a flipped rover to permanently
kill an unattended demo (the previous "detect and halt, restart the demo"
behaviour), it performs an explicitly SIMULATED recovery - it teleports the
model back to its last known-upright pose via gz-sim's set_pose service.

This is deliberately labelled as a simulated reset in every log line: it is
NOT physical self-righting and would not exist on real hardware, where the
preventive terrain smoothing (and, on a real rover, a suitable chassis /
righting mechanism) is what matters. It exists so the PoC demo keeps running.

Detection and the reset pose both come from ground truth (/ground_truth/pose,
the model's world-frame pose that set_pose also operates in) rather than the
EKF, whose two_d_mode estimate can never show roll/pitch. The reset keeps the
rover's (x, y) essentially where it was - stepping back only ~backoff_s of
travel to get off the offending lip - so the localization estimate (which
does not observe the teleport) stays consistent with the physical position to
within its normal noise, and forces the orientation upright.
"""

import math
import subprocess
from collections import deque

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node


def _roll_pitch_yaw(q) -> tuple:
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


class FlipRecoveryNode(Node):
    def __init__(self):
        super().__init__("regolith_flip_recovery")
        self.declare_parameter("world_name", "regolith_moon")
        self.declare_parameter("model_name", "rover")
        self.declare_parameter("flip_threshold_deg", 60.0)
        self.declare_parameter("safe_threshold_deg", 25.0)
        self.declare_parameter("debounce_s", 1.0)   # sustained flip before acting
        self.declare_parameter("backoff_s", 2.0)    # base "how far back (in sim time)" the reset pose is
        self.declare_parameter("max_backoff_s", 40.0)  # cap on progressive backoff
        self.declare_parameter("cooldown_s", 5.0)   # re-arm delay after a reset
        self.declare_parameter("relapse_window_s", 20.0)  # re-flip within this of a reset => "stuck", back off further
        self.declare_parameter("clearance_m", 0.3)  # z lift above recorded ground height
        self.declare_parameter("check_period_s", 0.2)

        # ~180 s of upright trail at check_period_s, so progressive backoff can
        # step back to a genuinely different location rather than the same lip.
        self._history = deque(maxlen=900)  # (t, x, y, z, yaw) upright poses
        self._pose = None
        self._flip_since = None
        self._cooldown_until = None
        self._resets = 0
        self._consecutive = 0        # rapid re-flips near the same spot
        self._last_reset_t = None

        self.create_subscription(PoseStamped, "/ground_truth/pose", self._on_pose, 10)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        period = self.get_parameter("check_period_s").value
        self.create_timer(period, self._tick)
        self.get_logger().info("Flip recovery armed (simulated set_pose backstop)")

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_pose(self, msg: PoseStamped) -> None:
        self._pose = msg.pose

    def _tick(self) -> None:
        if self._pose is None:
            return
        roll, pitch, yaw = _roll_pitch_yaw(self._pose.orientation)
        t = self._now_s()
        safe = math.radians(self.get_parameter("safe_threshold_deg").value)
        flip = math.radians(self.get_parameter("flip_threshold_deg").value)
        p = self._pose.position

        if abs(roll) < safe and abs(pitch) < safe:
            # Upright: record as a candidate recovery pose.
            self._history.append((t, p.x, p.y, p.z, yaw))
            self._flip_since = None
            return

        if abs(roll) > flip or abs(pitch) > flip:
            if self._cooldown_until is not None and t < self._cooldown_until:
                return
            if self._flip_since is None:
                self._flip_since = t
                return
            if (t - self._flip_since) < self.get_parameter("debounce_s").value:
                return
            self._recover(roll, pitch)

    def _recover(self, roll: float, pitch: float) -> None:
        if not self._history:
            self.get_logger().warn(
                "Rover flipped but no upright pose was ever recorded - cannot reset"
            )
            self._cooldown_until = self._now_s() + self.get_parameter("cooldown_s").value
            return
        t_now = self._now_s()
        # If we re-flip shortly after a reset, the last reset dropped the rover
        # right back onto whatever flipped it - back off PROGRESSIVELY further
        # down the recorded trail instead of looping forever on the same lip
        # (the previous behaviour: fixed 2 s backoff + clearing history, which
        # produced 20+ teleports at one spot - see PROGRESS.md).
        relapse_window = self.get_parameter("relapse_window_s").value
        if self._last_reset_t is not None and (t_now - self._last_reset_t) < relapse_window:
            self._consecutive += 1
        else:
            self._consecutive = 0

        base_backoff = self.get_parameter("backoff_s").value
        max_backoff = self.get_parameter("max_backoff_s").value
        backoff = min(base_backoff * (2 ** self._consecutive), max_backoff)

        # Newest upright pose that is at least `backoff` seconds old (steps the
        # rover back along the path it actually drove); fall back to the oldest.
        # History is NOT cleared, so successive relapses reach ever-earlier,
        # genuinely-different poses rather than re-selecting near the flip.
        target = self._history[0]
        for entry in reversed(self._history):
            if t_now - entry[0] >= backoff:
                target = entry
                break
        _, x, y, z, yaw = target
        z += self.get_parameter("clearance_m").value

        self._stop()  # zero cmd_vel so it doesn't drive off mid-teleport
        ok = self._set_pose(x, y, z, yaw)
        self._resets += 1
        stuck = " (stuck: backing off further along the trail)" if self._consecutive else ""
        self.get_logger().warn(
            f"SIMULATED RECOVERY #{self._resets}{stuck}: rover flipped "
            f"(roll {math.degrees(roll):.0f} deg, pitch {math.degrees(pitch):.0f} deg). "
            f"Teleported upright to ({x:.2f}, {y:.2f}, {z:.2f}) yaw {math.degrees(yaw):.0f} deg "
            f"(backoff {backoff:.0f} s) via gz set_pose [{'ok' if ok else 'FAILED'}]. "
            "This is a simulated self-right, not physical - see flip_recovery_node.py."
        )
        self._flip_since = None
        self._last_reset_t = t_now
        self._cooldown_until = t_now + self.get_parameter("cooldown_s").value
        # Drop trail entries newer than the target (they lead into the flip);
        # keep the earlier trail so a further relapse can back off more.
        while self._history and self._history[-1][0] > target[0]:
            self._history.pop()

    def _stop(self) -> None:
        self._cmd_pub.publish(Twist())

    def _set_pose(self, x: float, y: float, z: float, yaw: float) -> bool:
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        world = self.get_parameter("world_name").value
        model = self.get_parameter("model_name").value
        req = (
            f'name: "{model}" '
            f"position {{ x: {x:.4f} y: {y:.4f} z: {z:.4f} }} "
            f"orientation {{ x: 0 y: 0 z: {qz:.6f} w: {qw:.6f} }}"
        )
        cmd = [
            "gz", "service", "-s", f"/world/{world}/set_pose",
            "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000", "--req", req,
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
        except Exception as exc:  # noqa: BLE001 - log and report failure, never crash the node
            self.get_logger().error(f"set_pose call raised: {exc}")
            return False
        if out.returncode != 0:
            self.get_logger().error(f"set_pose failed: {out.stderr.strip() or out.stdout.strip()}")
            return False
        return "true" in out.stdout.lower()


def main() -> None:
    rclpy.init()
    node = FlipRecoveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
