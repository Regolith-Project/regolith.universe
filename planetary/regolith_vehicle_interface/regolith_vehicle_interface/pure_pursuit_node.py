# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Minimal pure pursuit path follower, outputting skid-steer cmd_vel directly
(linear + angular velocity - no steering-angle conversion needed, unlike the
Ackermann-oriented autoware_pure_pursuit; see docs/architecture.md's reuse log
for why that package wasn't reused here).

Recovery is intentionally minimal per the plan ("do not build elaborate FDIR
now"): if the rover strays far from the path or stalls, it stops and
re-triggers planning from wherever it currently is, rather than anything more
sophisticated.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import Bool


def _yaw_from_quaternion(q) -> float:
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))


def _normalize_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__("regolith_pure_pursuit")
        self.declare_parameter("lookahead_distance_m", 1.5)
        self.declare_parameter("base_speed_mps", 0.2)
        self.declare_parameter("max_angular_velocity", 0.3)
        self.declare_parameter("goal_tolerance_m", 1.5)
        self.declare_parameter("path_deviation_limit_m", 4.0)
        self.declare_parameter("stall_timeout_s", 8.0)
        self.declare_parameter("control_period_s", 0.1)

        self._path = None
        self._current_pose = None
        self._current_goal = None
        self._costmap = None
        self._goal_reached = True
        self._last_progress_time = self.get_clock().now()
        self._last_progress_position = None

        costmap_qos = QoSProfile(depth=1)
        costmap_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(OccupancyGrid, "/costmap", self._on_costmap, costmap_qos)

        path_qos = QoSProfile(depth=1)
        path_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Path, "/planned_path", self._on_path, path_qos)

        self.create_subscription(Odometry, "/odometry/filtered", self._on_odometry, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._goal_reached_pub = self.create_publisher(Bool, "/goal_reached", 10)

        period = self.get_parameter("control_period_s").value
        self.create_timer(period, self._control_step)

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        self._costmap = msg

    def _on_path(self, msg: Path) -> None:
        self._path = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        self._goal_reached = False
        self._last_progress_time = self.get_clock().now()
        self._last_progress_position = None

    def _on_goal(self, msg: PoseStamped) -> None:
        self._current_goal = msg

    def _on_odometry(self, msg: Odometry) -> None:
        self._current_pose = msg.pose.pose

    def _cost_at(self, x_m: float, y_m: float) -> float:
        if self._costmap is None:
            return 0.0
        info = self._costmap.info
        col = int((x_m - info.origin.position.x) / info.resolution)
        row = int((y_m - info.origin.position.y) / info.resolution)
        if not (0 <= row < info.height and 0 <= col < info.width):
            return 0.0
        return float(self._costmap.data[row * info.width + col])

    def _stop(self) -> None:
        self._cmd_pub.publish(Twist())

    def _control_step(self) -> None:
        if self._path is None or self._goal_reached or self._current_pose is None or len(self._path) == 0:
            return

        position = np.array([self._current_pose.position.x, self._current_pose.position.y])
        yaw = _yaw_from_quaternion(self._current_pose.orientation)

        goal_xy = self._path[-1]
        distance_to_goal = float(np.linalg.norm(goal_xy - position))
        if distance_to_goal < self.get_parameter("goal_tolerance_m").value:
            self._stop()
            self._goal_reached = True
            self.get_logger().info(f"Goal reached (within {distance_to_goal:.2f} m)")
            self._goal_reached_pub.publish(Bool(data=True))
            return

        distances = np.linalg.norm(self._path - position, axis=1)
        nearest_idx = int(np.argmin(distances))
        deviation = float(distances[nearest_idx])

        if self._check_stalled_or_deviated(position, deviation):
            return

        lookahead_m = self.get_parameter("lookahead_distance_m").value
        target_idx = nearest_idx
        accumulated = 0.0
        while target_idx < len(self._path) - 1 and accumulated < lookahead_m:
            accumulated += float(np.linalg.norm(self._path[target_idx + 1] - self._path[target_idx]))
            target_idx += 1
        target_xy = self._path[target_idx]

        heading_to_target = float(np.arctan2(target_xy[1] - position[1], target_xy[0] - position[0]))
        alpha = _normalize_angle(heading_to_target - yaw)

        max_angular = self.get_parameter("max_angular_velocity").value
        angular_z = float(np.clip(1.5 * alpha, -max_angular, max_angular))

        # Modest speed profile: slow down for sharp turns and for high-cost terrain.
        # Large heading errors (e.g. right after a (re)plan, or a sharp corner) stop
        # forward motion entirely and rotate in place first - driving forward while
        # turning sharply on rough terrain is what flipped the rover in testing (see
        # PROGRESS.md M4); rotate-then-drive is a standard, more robust pattern.
        base_speed = self.get_parameter("base_speed_mps").value
        if abs(alpha) > np.pi / 6:
            linear_x = 0.0
        else:
            turn_factor = max(0.15, 1.0 - abs(alpha) / (np.pi / 3))
            cost_factor = max(0.4, 1.0 - self._cost_at(*position) / 100.0)
            linear_x = base_speed * turn_factor * cost_factor

        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self._cmd_pub.publish(cmd)

    def _check_stalled_or_deviated(self, position: np.ndarray, deviation: float) -> bool:
        """Minimal recovery: if far off the path or not making progress, stop and
        re-trigger planning from the current position. Returns True if recovery
        action was taken (caller should skip the normal control step this cycle)."""
        now = self.get_clock().now()
        deviation_limit = self.get_parameter("path_deviation_limit_m").value
        if deviation > deviation_limit:
            self.get_logger().warn(f"Deviated {deviation:.2f} m from path - stopping and replanning")
            self._stop()
            self._replan()
            return True

        stall_timeout = self.get_parameter("stall_timeout_s").value
        if self._last_progress_position is None:
            self._last_progress_position = position
            self._last_progress_time = now
        elif np.linalg.norm(position - self._last_progress_position) > 0.2:
            self._last_progress_position = position
            self._last_progress_time = now
        elif (now - self._last_progress_time).nanoseconds / 1e9 > stall_timeout:
            self.get_logger().warn(f"Stalled for {stall_timeout:.0f}s - stopping and replanning")
            self._stop()
            self._replan()
            return True
        return False

    def _replan(self) -> None:
        if self._current_goal is not None:
            self._goal_pub.publish(self._current_goal)
        self._path = None
        self._last_progress_position = None
        self._last_progress_time = self.get_clock().now()


def main() -> None:
    rclpy.init()
    node = PurePursuitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
