#!/usr/bin/env python3
# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Drives a short automatic tour, for the `mission:=tour` hello_moon.launch.py demo.

Legs are short (~10-20 m) rather than one long traverse: M4 testing found the rover can
flip at a terrain-collision boundary crossing on longer runs (see PROGRESS.md M4), and a
demo meant to run unattended shouldn't gamble on that.

THE ROUTE IS DERIVED FROM THE COSTMAP, not hardcoded. It used to be five fixed (x, y)
pairs chosen by hand, and once the terrain went to res40 and rock collision started
working they no longer described drivable ground: seed 42's first waypoint, seed 7's
second, and seed 123's second and third were all sitting on LETHAL cells. The planner
refuses those, so the leg was skipped after a 90 s timeout and the demo sat still. See
`regolith_planner.tour` for how a leg is now chosen and checked.

It plans against `/costmap` - the very grid the planner node will use, subscribed rather
than rebuilt - so the tour cannot be validated against a costmap that differs from the
one in play. Rebuilding it here from the manifest would mean duplicating the resolution,
rover radius and slope threshold that `hello_moon.launch.py` passes to `costmap_node`,
and any drift between the two copies would put this straight back where it started.
"""

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from regolith_planner.tour import plan_tour
from std_msgs.msg import Bool

WAYPOINT_TIMEOUT_S = 90.0
START_DELAY_S = 10.0
# How long to wait for /costmap before saying so. It is latched and published by the
# same launch, so this only trips if costmap_node is genuinely absent or broken.
COSTMAP_WAIT_WARN_S = 20.0


class TourMissionNode(Node):
    def __init__(self):
        super().__init__("regolith_tour_mission")
        self.declare_parameter("seed", 42)

        self._waypoints = []
        self._seen_costmap = False
        self._waypoint_index = 0
        self._timeout_timer = None
        self._start_timer = None

        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.create_subscription(Bool, "/goal_reached", self._on_goal_reached, 10)

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        # Announced once, latched, so mission_markers_node can flag the whole route from
        # the moment it comes up rather than revealing it one leg at a time.
        self._route_pub = self.create_publisher(Path, "/mission_waypoints", latched)
        self.create_subscription(OccupancyGrid, "/costmap", self._on_costmap, latched)

        self._warn_timer = self.create_timer(COSTMAP_WAIT_WARN_S, self._warn_no_costmap)
        self.get_logger().info("Tour waiting for /costmap before choosing a route")

    def _warn_no_costmap(self) -> None:
        if self._waypoints:
            return
        # Distinguish the two ways of having no route: nothing to plan on, or nothing
        # plannable on it. They need different things looked at.
        if self._seen_costmap:
            self.get_logger().warn(
                "a costmap arrived but no drivable tour could be built from it - the "
                "rover will not move"
            )
        else:
            self.get_logger().warn(
                f"still no /costmap after {COSTMAP_WAIT_WARN_S:.0f}s - the tour cannot "
                f"choose a route until it arrives, and will not drive until it does"
            )

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        """The costmap is republished at 1 Hz; the route is chosen once, off the first."""
        if self._waypoints:
            return
        self._seen_costmap = True
        grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        result = plan_tour(
            grid,
            msg.info.resolution,
            msg.info.origin.position.x,
            msg.info.origin.position.y,
            seed=int(self.get_parameter("seed").value),
        )
        self._waypoints = result["waypoints"]
        for note in result["notes"]:
            self.get_logger().warn(f"Tour route: {note}")
        if not self._waypoints:
            self.get_logger().error("No drivable tour could be built from this costmap")
            return

        self._warn_timer.cancel()
        self.get_logger().info(
            "Tour route from the live costmap: "
            + " -> ".join(f"({x:.1f}, {y:.1f})" for x, y in self._waypoints)
        )
        self._publish_route()
        self._start_timer = self.create_timer(START_DELAY_S, self._start_once)

    def _publish_route(self) -> None:
        route = Path()
        route.header.frame_id = "odom"
        route.header.stamp = self.get_clock().now().to_msg()
        for x_m, y_m in self._waypoints:
            pose = PoseStamped()
            pose.header = route.header
            pose.pose.position.x = float(x_m)
            pose.pose.position.y = float(y_m)
            pose.pose.orientation.w = 1.0
            route.poses.append(pose)
        self._route_pub.publish(route)

    def _start_once(self) -> None:
        # create_timer has no one-shot mode; cancel after first fire.
        self._start_timer.cancel()
        self._send_next_waypoint()

    def _send_next_waypoint(self) -> None:
        if self._waypoint_index >= len(self._waypoints):
            self.get_logger().info("Tour complete - all waypoints visited")
            return

        x_m, y_m = self._waypoints[self._waypoint_index]
        goal = PoseStamped()
        goal.header.frame_id = "odom"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(x_m)
        goal.pose.position.y = float(y_m)
        goal.pose.orientation.w = 1.0
        self._goal_pub.publish(goal)
        self.get_logger().info(
            f"Tour: heading to waypoint {self._waypoint_index + 1}/{len(self._waypoints)} "
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
