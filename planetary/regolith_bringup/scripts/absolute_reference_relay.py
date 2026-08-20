#!/usr/bin/env python3
# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""SIMULATED absolute position reference, for experiments only.

This republishes `/ground_truth/pose` as a `PoseWithCovarianceStamped` that the
EKF can fuse as an absolute x/y observation. It is an ORACLE: it hands the
estimator the answer. Nothing that runs by default may use it, and no
acceptance number obtained with it is an acceptance number - which is why it
sits behind `hello_moon.launch.py`'s `localization_oracle` argument, defaults
to off, and announces itself loudly at startup.

WHAT IT IS FOR. The res40 M4 verification (PROGRESS.md) ended 0/3 with a
measured diagnosis: recovery works (25 escape maneuvers, 25 successful), the
wheels over-claim distance by only ~1% once the slip gate is in, and the IMU's
heading is accurate - but the rover slides sideways ~10% of its total motion,
and a differential-drive odometry model cannot represent lateral velocity at
all while an IMU cannot observe it. That predicts something specific and
falsifiable:

    if localization is the ONLY thing between this stack and M4, then giving
    the estimator an absolute reference should make the acceptance pass, with
    no change to planning, control or recovery.

Running the acceptance with this node on tests exactly that. A pass supports
the diagnosis and bounds what visual odometry would have to buy. A failure
refutes it and says something else is binding - which is worth knowing before
anyone writes a line of visual odometry.

The covariance is deliberately loose (see POSITION_VARIANCE) rather than
near-zero: a real terrain-relative or visual-odometry fix arrives with metre-
scale uncertainty and at a low rate, so an oracle pretending to millimetre
precision would answer a question nobody is asking.
"""

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node

# 0.25 m^2 -> 0.5 m standard deviation on x and y. Chosen to be comparable to
# what a working visual-odometry/TRN fix would actually deliver on this rover,
# not to what the simulator knows.
POSITION_VARIANCE = 0.25
ORIENTATION_VARIANCE = 0.05


class AbsoluteReferenceRelay(Node):
    def __init__(self):
        super().__init__("regolith_absolute_reference")
        self.declare_parameter("publish_rate_divisor", 10)  # ~1 Hz from a ~10 Hz source
        self._divisor = max(1, int(self.get_parameter("publish_rate_divisor").value))
        self._count = 0

        self._pub = self.create_publisher(PoseWithCovarianceStamped, "/absolute_reference/pose", 10)
        self.create_subscription(PoseStamped, "/ground_truth/pose", self._on_pose, 10)
        self.get_logger().warn(
            "SIMULATED ABSOLUTE REFERENCE IS ACTIVE - the EKF is being fed ground-truth "
            "position at ~1 Hz with 0.5 m sigma. This is an oracle standing in for the "
            "visual odometry this PoC does not have. Any acceptance result produced in "
            "this mode is an EXPERIMENT, not a milestone result. See "
            "absolute_reference_relay.py and PROGRESS.md."
        )

    def _on_pose(self, msg: PoseStamped) -> None:
        self._count += 1
        if self._count % self._divisor:
            return
        out = PoseWithCovarianceStamped()
        out.header = msg.header
        out.header.frame_id = "odom"  # the frame the EKF estimates in
        out.pose.pose = msg.pose
        covariance = [0.0] * 36
        covariance[0] = POSITION_VARIANCE  # x
        covariance[7] = POSITION_VARIANCE  # y
        covariance[14] = POSITION_VARIANCE  # z
        covariance[21] = ORIENTATION_VARIANCE  # roll
        covariance[28] = ORIENTATION_VARIANCE  # pitch
        covariance[35] = ORIENTATION_VARIANCE  # yaw
        out.pose.covariance = covariance
        self._pub.publish(out)


def main() -> None:
    rclpy.init()
    node = AbsoluteReferenceRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
