# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Minimal pure pursuit path follower, outputting skid-steer cmd_vel directly
(linear + angular velocity - no steering-angle conversion needed, unlike the
Ackermann-oriented autoware_pure_pursuit; see docs/architecture.md's reuse log
for why that package wasn't reused here).

Recovery is intentionally minimal per the plan ("do not build elaborate FDIR
now"): if the rover strays far from the path or stalls, it stops and
re-triggers planning from wherever it currently is, rather than anything more
sophisticated. The one hard failure it does detect explicitly is a flipped
rover (large roll/pitch on the raw IMU): it pauses following instead of
replanning forever, and the separate flip_recovery_node performs a simulated
set_pose righting, after which this node's attitude check clears and following
resumes automatically.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool


def _yaw_from_quaternion(q) -> float:
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))


def _roll_pitch_from_quaternion(q) -> tuple:
    roll = np.arctan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x**2 + q.y**2))
    pitch = np.arcsin(np.clip(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0))
    return float(roll), float(pitch)


def _normalize_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__("regolith_pure_pursuit")
        self.declare_parameter("lookahead_distance_m", 1.5)
        self.declare_parameter("base_speed_mps", 0.2)
        self.declare_parameter("max_angular_velocity", 0.3)
        # Deliberately TIGHTER than the 1.5 m the milestone asks for. When the
        # two were equal the rover stopped the instant it crossed the
        # requirement, so every pass measured exactly 1.50 m and any small
        # difference between this node's frame and the judge's would flip it to
        # a fail. Arriving with margin is the point; the bar itself is unchanged.
        self.declare_parameter("goal_tolerance_m", 1.0)
        self.declare_parameter("path_deviation_limit_m", 4.0)
        self.declare_parameter("stall_timeout_s", 8.0)
        self.declare_parameter("control_period_s", 0.1)
        self.declare_parameter("flipped_attitude_deg", 60.0)
        # Recovery is deliberately minimal (see module docstring), but minimal
        # still needs a floor: without a cap, a goal that's unreachable in
        # practice (terrain edge case, or the rover genuinely stuck nearby)
        # makes _replan() retrigger itself forever - observed running for
        # hours straight, tens of thousands of "stopping and replanning"
        # cycles, in a session where it turned out two overlapping demo
        # launches were also fighting over /goal_pose (see PROGRESS.md's
        # "overnight freeze" note). That root cause is separate, but this
        # node having no give-up condition at all made it far worse, and is
        # worth fixing on its own regardless of what triggers the retries.
        self.declare_parameter("max_consecutive_replans", 8)

        self._path = None
        self._current_pose = None
        self._current_goal = None
        self._costmap = None
        self._goal_reached = True
        self._last_progress_time = self.get_clock().now()
        self._last_progress_position = None
        self._imu_orientation = None
        self._flipped = False
        self._replan_count = 0
        self._given_up_goal_xy = None  # (x, y) of a goal we've stopped retrying
        self._recovery_active = False

        costmap_qos = QoSProfile(depth=1)
        costmap_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(OccupancyGrid, "/costmap", self._on_costmap, costmap_qos)

        path_qos = QoSProfile(depth=1)
        path_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Path, "/planned_path", self._on_path, path_qos)

        self.create_subscription(Odometry, "/odometry/filtered", self._on_odometry, 10)
        # Raw IMU, for flip detection only: the EKF runs in two_d_mode, so its
        # fused orientation never shows roll/pitch even when the rover is
        # physically upside-down (see PROGRESS.md M5 - stall recovery used to
        # silently replan forever after a flip).
        self.create_subscription(Imu, "/imu", self._on_imu, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        # While the recovery node is running an escape maneuver it owns
        # /cmd_vel outright. Publishing "faster than" this node is not the same
        # as controlling the rover - at 30 Hz against 10 Hz, a quarter of the
        # commands gz-sim executed during a maneuver were still this node's,
        # driving forward into the obstacle the maneuver was reversing out of.
        self.create_subscription(Bool, "/recovery_active", self._on_recovery_active, 10)
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._goal_reached_pub = self.create_publisher(Bool, "/goal_reached", 10)

        period = self.get_parameter("control_period_s").value
        self.create_timer(period, self._control_step)

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        # A costmap that has genuinely changed (the recovery node marking a
        # keep-out zone where the rover wedged) means the next attempt at this
        # goal is not the same attempt again, so the give-up budget below is
        # restored. Without this, a run with several wedges exhausts its 8
        # replans on approaches the planner can now route around, and the rover
        # parks with the goal declared unreachable.
        if self._costmap is not None and msg.data != self._costmap.data:
            if self._replan_count or self._given_up_goal_xy is not None:
                self.get_logger().info(
                    "Costmap changed (a new keep-out zone) - restoring the replan budget: the "
                    "route being retried is not the one that already failed"
                )
            self._replan_count = 0
            self._given_up_goal_xy = None
        self._costmap = msg

    def _on_path(self, msg: Path) -> None:
        self._path = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        self._goal_reached = False
        self._last_progress_time = self.get_clock().now()
        self._last_progress_position = None

    def _on_goal(self, msg: PoseStamped) -> None:
        new_xy = (msg.pose.position.x, msg.pose.position.y)
        old_xy = (
            (self._current_goal.pose.position.x, self._current_goal.pose.position.y)
            if self._current_goal is not None
            else None
        )
        if new_xy != old_xy:
            # A genuinely new goal (RViz click, or the next tour waypoint) -
            # separate from this node's own _replan() republishing the same
            # goal after a deviation/stall. Give the new goal a fresh budget.
            self._replan_count = 0
            self._given_up_goal_xy = None
        self._current_goal = msg

    def _on_odometry(self, msg: Odometry) -> None:
        self._current_pose = msg.pose.pose

    def _on_imu(self, msg: Imu) -> None:
        self._imu_orientation = msg.orientation

    def _on_recovery_active(self, msg: Bool) -> None:
        if msg.data != self._recovery_active:
            self.get_logger().info(
                "Recovery node has taken over /cmd_vel - pausing path following"
                if msg.data
                else "Recovery finished - resuming path following"
            )
        self._recovery_active = msg.data
        if msg.data:
            # Progress timing restarts after the maneuver: the rover being held
            # still by someone else is not a stall of this node's making.
            self._last_progress_position = None
            self._last_progress_time = self.get_clock().now()

    def _check_flipped(self) -> bool:
        """Fail loudly instead of silently cycling through stall recovery when the
        rover is physically flipped. Returns True while the attitude is beyond the
        limit (caller must stop and skip the control step)."""
        if self._imu_orientation is None:
            return False
        roll, pitch = _roll_pitch_from_quaternion(self._imu_orientation)
        limit = np.deg2rad(self.get_parameter("flipped_attitude_deg").value)
        if abs(roll) > limit or abs(pitch) > limit:
            if not self._flipped:
                self._flipped = True
                self.get_logger().error(
                    f"Rover attitude is roll {np.degrees(roll):.0f} deg, pitch "
                    f"{np.degrees(pitch):.0f} deg - it has likely flipped. Pausing path "
                    "following; the flip_recovery_node should teleport it upright "
                    "(simulated recovery), after which following resumes automatically."
                )
            else:
                self.get_logger().warn(
                    "Rover still flipped - path following remains halted",
                    throttle_duration_sec=30.0,
                )
            return True
        if self._flipped:
            self._flipped = False
            self.get_logger().info("Rover attitude back within limits - resuming path following")
        return False

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
        if self._recovery_active:
            return  # the recovery node owns /cmd_vel; do not fight it
        if self._check_flipped():
            self._stop()
            return
        if self._path is None or self._goal_reached or self._current_pose is None or len(self._path) == 0:
            return

        current_goal_xy = (
            (self._current_goal.pose.position.x, self._current_goal.pose.position.y)
            if self._current_goal is not None
            else None
        )
        if self._given_up_goal_xy is not None and self._given_up_goal_xy == current_goal_xy:
            return

        position = np.array([self._current_pose.position.x, self._current_pose.position.y])
        yaw = _yaw_from_quaternion(self._current_pose.orientation)

        # Arrival is measured against the goal that was COMMANDED, not against
        # path[-1]. The path's last waypoint is a costmap cell CENTRE - the
        # planner snaps both ends to the grid - so at this world's 0.78 m cells
        # it sits up to ~0.55 m from the goal actually asked for. Stopping
        # "within 1.50 m" of that point can leave the rover nearly 2 m from the
        # goal, and it did: with localization made perfect by the experimental
        # absolute reference, seeds 42 and 7 both stopped at exactly 1.70 m and
        # failed a 1.5 m bar for no other reason (path[-1] was 0.42 m off the
        # goal on seed 42). That is the same mistake as trusting /goal_reached -
        # measuring against the wrong reference - so it is fixed the same way.
        goal_xy = (
            np.array(current_goal_xy) if current_goal_xy is not None else self._path[-1]
        )
        distance_to_goal = float(np.linalg.norm(goal_xy - position))
        if distance_to_goal < self.get_parameter("goal_tolerance_m").value:
            self._stop()
            self._goal_reached = True
            self._replan_count = 0
            self._given_up_goal_xy = None
            self.get_logger().info(f"Goal reached (within {distance_to_goal:.2f} m of the commanded goal)")
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
        if target_idx == len(self._path) - 1:
            # Past the end of the path, steer at the real goal rather than the
            # grid-snapped waypoint, so the final approach closes the last
            # half-metre instead of parking next to it.
            target_xy = goal_xy

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
        self._path = None
        self._last_progress_position = None
        self._last_progress_time = self.get_clock().now()

        if self._current_goal is None:
            return

        self._replan_count += 1
        max_replans = self.get_parameter("max_consecutive_replans").value
        if self._replan_count > max_replans:
            goal_xy = (self._current_goal.pose.position.x, self._current_goal.pose.position.y)
            self._given_up_goal_xy = goal_xy
            self.get_logger().error(
                f"Giving up on goal ({goal_xy[0]:.1f}, {goal_xy[1]:.1f}) after "
                f"{self._replan_count} consecutive deviate/stall replans with no progress - "
                "not retrying it again until a genuinely new /goal_pose arrives. (This cap "
                "exists so an unreachable goal can't loop forever - see PROGRESS.md's "
                "\"overnight freeze\" note for why.)"
            )
            return

        self._goal_pub.publish(self._current_goal)


def main() -> None:
    rclpy.init()
    node = PurePursuitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
