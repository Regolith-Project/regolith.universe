#!/usr/bin/env python3
# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Drives a fixed sequence of waypoints automatically, for the `mission:=tour`
hello_moon.launch.py demo. Waypoints are short (~10-20 m apart) rather than a
single long traverse: M4 testing found the rover can flip at a terrain-
collision boundary crossing on longer runs (see PROGRESS.md M4), and a
demo that's meant to run unattended shouldn't gamble on that. Each leg still
crosses genuine costmap-flagged terrain, so the planner/follower are doing
real work, just not at the full 60-100 m single-goal distance from the M4
acceptance check.
"""

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool

# A short loop back to (roughly) the start, staying within the range verified
# stable in M4 testing.
WAYPOINTS = [
    (12.0, 8.0),
    (18.0, -4.0),
    (4.0, -14.0),
    (-10.0, -6.0),
    (0.0, 0.0),
]
WAYPOINT_TIMEOUT_S = 90.0
START_DELAY_S = 10.0


class TourMissionNode(Node):
    def __init__(self):
        super().__init__("regolith_tour_mission")
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.create_subscription(Bool, "/goal_reached", self._on_goal_reached, 10)
        self._waypoint_index = 0
        self._timeout_timer = None
        self._start_timer = self.create_timer(START_DELAY_S, self._start_once)

    def _start_once(self) -> None:
        # create_timer has no one-shot mode; cancel after first fire.
        self._start_timer.cancel()
        self._send_next_waypoint()

    def _send_next_waypoint(self) -> None:
        if self._waypoint_index >= len(WAYPOINTS):
            self.get_logger().info("Tour complete - all waypoints visited")
            return

        x_m, y_m = WAYPOINTS[self._waypoint_index]
        goal = PoseStamped()
        goal.header.frame_id = "odom"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x_m
        goal.pose.position.y = y_m
        goal.pose.orientation.w = 1.0
        self._goal_pub.publish(goal)
        self.get_logger().info(
            f"Tour: heading to waypoint {self._waypoint_index + 1}/{len(WAYPOINTS)} "
            f"({x_m:.1f}, {y_m:.1f})"
        )

        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
        self._timeout_timer = self.create_timer(WAYPOINT_TIMEOUT_S, self._on_timeout)

    def _on_goal_reached(self, msg: Bool) -> None:
        if not msg.data:
            return
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
        self._waypoint_index += 1
        self._send_next_waypoint()

    def _on_timeout(self) -> None:
        self.get_logger().warn(
            f"Waypoint {self._waypoint_index + 1} timed out after {WAYPOINT_TIMEOUT_S:.0f}s - "
            "moving on to the next one anyway"
        )
        self._timeout_timer.cancel()
        self._waypoint_index += 1
        self._send_next_waypoint()


def main() -> None:
    rclpy.init()
    node = TourMissionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
