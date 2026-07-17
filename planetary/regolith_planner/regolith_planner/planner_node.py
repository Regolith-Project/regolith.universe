# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Plans a cost-aware path from the current (EKF-estimated) pose to an RViz
"2D Goal Pose" click, over the /costmap published by regolith_costmap."""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from regolith_planner.astar import plan_path, smooth_path


class PlannerNode(Node):
    def __init__(self):
        super().__init__("regolith_planner")
        self._costmap = None
        self._current_pose = None

        costmap_qos = QoSProfile(depth=1)
        costmap_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(OccupancyGrid, "/costmap", self._on_costmap, costmap_qos)
        self.create_subscription(Odometry, "/odometry/filtered", self._on_odometry, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)

        path_qos = QoSProfile(depth=1)
        path_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._path_pub = self.create_publisher(Path, "/planned_path", path_qos)

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        self._costmap = msg

    def _on_odometry(self, msg: Odometry) -> None:
        self._current_pose = msg.pose.pose

    def _world_to_grid(self, x_m: float, y_m: float) -> tuple:
        info = self._costmap.info
        col = int((x_m - info.origin.position.x) / info.resolution)
        row = int((y_m - info.origin.position.y) / info.resolution)
        return row, col

    def _grid_to_world(self, row: int, col: int) -> tuple:
        info = self._costmap.info
        x_m = info.origin.position.x + (col + 0.5) * info.resolution
        y_m = info.origin.position.y + (row + 0.5) * info.resolution
        return x_m, y_m

    def _on_goal(self, goal: PoseStamped) -> None:
        if self._costmap is None or self._current_pose is None:
            self.get_logger().warn("No costmap or current pose yet - ignoring goal")
            return

        info = self._costmap.info
        cost_grid = np.array(self._costmap.data, dtype=np.int16).reshape(info.height, info.width)

        start_rc = self._world_to_grid(self._current_pose.position.x, self._current_pose.position.y)
        goal_rc = self._world_to_grid(goal.pose.position.x, goal.pose.position.y)

        if not (0 <= start_rc[0] < info.height and 0 <= start_rc[1] < info.width):
            self.get_logger().warn("Current pose is outside the costmap bounds")
            return
        if not (0 <= goal_rc[0] < info.height and 0 <= goal_rc[1] < info.width):
            self.get_logger().warn("Goal is outside the costmap bounds")
            return

        grid_path = plan_path(cost_grid, start_rc, goal_rc)
        if not grid_path:
            self.get_logger().warn(
                f"No path found from {start_rc} to {goal_rc} (goal may be unreachable "
                "or in a lethal cell)"
            )
            return

        world_path = [self._grid_to_world(r, c) for r, c in grid_path]
        world_path = smooth_path(world_path)

        path_msg = Path()
        path_msg.header.frame_id = self._costmap.header.frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for x_m, y_m in world_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x_m
            pose.pose.position.y = y_m
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self._path_pub.publish(path_msg)
        self.get_logger().info(
            f"Planned path: {len(world_path)} waypoints, "
            f"start={start_rc} goal={goal_rc}"
        )


def main() -> None:
    rclpy.init()
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
