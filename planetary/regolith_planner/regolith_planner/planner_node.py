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

from regolith_planner.astar import LETHAL_COST, plan_path, smooth_path


def nearest_free_cell(cost_grid: np.ndarray, start_rc: tuple, max_radius_cells: int):
    """Closest non-lethal cell to start_rc, searched outward in rings.

    Used when the rover's own cell is lethal, which happens for two real
    reasons: localization drift putting the estimate inside an inflated rock,
    and the rover having marked a keep-out zone around the obstacle it was just
    wedged against (see costmap_node's hazard marking). Refusing to plan in
    that state strands the rover permanently; planning from a cell a metre or
    two away gives it something to drive onto as soon as it is clear.
    """
    rows, cols = cost_grid.shape
    for radius in range(1, max_radius_cells + 1):
        best = None
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue  # only the new ring; inner ones were checked already
                r, c = start_rc[0] + dr, start_rc[1] + dc
                if not (0 <= r < rows and 0 <= c < cols) or cost_grid[r, c] >= LETHAL_COST:
                    continue
                distance = dr * dr + dc * dc
                if best is None or distance < best[0]:
                    best = (distance, (r, c))
        if best is not None:
            return best[1]
    return None


class PlannerNode(Node):
    def __init__(self):
        super().__init__("regolith_planner")
        self.declare_parameter("escape_radius_cells", 5)
        self._escape_radius_cells = int(self.get_parameter("escape_radius_cells").value)
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

    def _nearest_free_cell(self, cost_grid: np.ndarray, start_rc: tuple):
        return nearest_free_cell(cost_grid, start_rc, self._escape_radius_cells)

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

        if cost_grid[goal_rc] >= LETHAL_COST:
            self.get_logger().warn(
                f"Goal cell {goal_rc} is lethal (obstacle or too-steep slope) - pick another goal"
            )
            return
        if cost_grid[start_rc] >= LETHAL_COST:
            escaped = self._nearest_free_cell(cost_grid, start_rc)
            if escaped is None:
                self.get_logger().warn(
                    f"Current-position cell {start_rc} is lethal and no free cell was found "
                    f"within {self._escape_radius_cells} cells - cannot plan from here"
                )
                return
            # Refusing to plan from a lethal start used to strand the rover
            # permanently in two situations that both really happen: localization
            # drift putting the estimate inside an inflated rock, and a keep-out
            # zone the rover marked around the obstacle it was just wedged
            # against (see costmap_node's hazard marking). Planning from the
            # nearest free cell instead gives a path the rover can pick up as
            # soon as it is a metre clear, which is what it is trying to do.
            self.get_logger().warn(
                f"Current-position cell {start_rc} is lethal (drift, or a keep-out zone this "
                f"rover marked itself) - planning from the nearest free cell {escaped} instead"
            )
            start_rc = escaped

        grid_path = plan_path(cost_grid, start_rc, goal_rc)
        if not grid_path:
            self.get_logger().warn(
                f"No traversable path exists from {start_rc} to {goal_rc} - the goal is "
                "walled off by lethal cells"
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
