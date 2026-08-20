#!/usr/bin/env python3
# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Republishes /imu and /odom with non-zero covariance filled in, and fixes the IMU's frame_id so robot_localization can actually use it.

gz-sim's IMU and DiffDrive-odometry outputs both publish all-zero covariance
matrices (no noise model configured), which by REP-145 convention means
"unknown" - robot_localization's EKF silently discards a measurement with
all-zero covariance rather than trusting it, rather than fusing the one
accurate absolute-heading source we have (the simulated IMU's orientation
exactly matches ground truth - see PROGRESS.md M3) or the wheel velocity.
Rather than tune a Gazebo noise model and hope it populates every covariance
field this needs, this relay just fills in small, fixed diagonal covariances
directly.
"""

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

ORIENTATION_VARIANCE = 0.01
ANGULAR_VELOCITY_VARIANCE = 0.001
LINEAR_ACCELERATION_VARIANCE = 0.01
ODOM_POSE_VARIANCE = 0.05
ODOM_TWIST_VARIANCE = 0.01


def _diagonal_covariance(variance: float, size: int = 3) -> list:
    cov = [0.0] * (size * size)
    for i in range(size):
        cov[i * size + i] = variance
    return cov


class SensorCovarianceRelay(Node):
    def __init__(self):
        super().__init__("sensor_covariance_relay")
        # The odometry source is a parameter because wheel_slip_node.py sits
        # between gz and this relay in the running system, publishing
        # /odom/gated - raw wheel odometry with a zero-velocity update
        # substituted while the wheels are slipping in place. This relay is
        # deliberately kept downstream of the gate so the covariance it fills
        # in is the one the EKF sees for whichever velocity it is given.
        self.declare_parameter("odom_topic", "/odom")
        self._imu_pub = self.create_publisher(Imu, "/imu/with_covariance", 10)
        self.create_subscription(Imu, "/imu", self._on_imu, 10)
        self._odom_pub = self.create_publisher(Odometry, "/odom/with_covariance", 10)
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self._on_odom, 10
        )

    def _on_imu(self, msg: Imu) -> None:
        # gz-sim reports the sensor's own internal frame name (e.g.
        # "rover/base_link/imu"), which has no corresponding TF frame - only
        # "imu_link" (from the URDF, published by robot_state_publisher) does.
        # robot_localization needs a resolvable TF transform from the sensor
        # frame to base_link and silently drops messages it can't transform.
        msg.header.frame_id = "imu_link"
        msg.orientation_covariance = _diagonal_covariance(ORIENTATION_VARIANCE)
        msg.angular_velocity_covariance = _diagonal_covariance(ANGULAR_VELOCITY_VARIANCE)
        msg.linear_acceleration_covariance = _diagonal_covariance(LINEAR_ACCELERATION_VARIANCE)
        self._imu_pub.publish(msg)

    def _on_odom(self, msg: Odometry) -> None:
        msg.pose.covariance = _diagonal_covariance(ODOM_POSE_VARIANCE, size=6)
        msg.twist.covariance = _diagonal_covariance(ODOM_TWIST_VARIANCE, size=6)
        self._odom_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = SensorCovarianceRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
