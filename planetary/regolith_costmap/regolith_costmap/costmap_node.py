# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Publishes a 2.5D traversability costmap from the known terrain heightmap.

Per the plan's honest simplifications: the costmap comes from the generated
terrain heightmap (the world is known a priori for this PoC), not from
onboard perception. Sensor-derived costmaps are the explicit next milestone.
"""

import json
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from PIL import Image
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from scipy import ndimage


def build_costmap(
    manifest: dict,
    heightmap: np.ndarray,
    resolution_m: float,
    rover_radius_m: float,
    slope_lethal_deg: float,
) -> tuple:
    """Returns (cost_grid uint8 [0-100], resolution_m, origin_x_m, origin_y_m)."""
    world_size_m = manifest["world_size_m"]
    grid_size = int(round(world_size_m / resolution_m))

    n = heightmap.shape[0]
    block = max(1, (n - 1) // grid_size)
    usable = ((n - 1) // block) * block
    trimmed = heightmap[:usable, :usable]
    blocks = usable // block
    elevation = trimmed.reshape(blocks, block, blocks, block).mean(axis=(1, 3))
    actual_resolution_m = world_size_m * (usable / (n - 1)) / blocks

    # Slope from the elevation gradient (degrees).
    gy, gx = np.gradient(elevation, actual_resolution_m)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

    # Roughness: local elevation standard deviation (3x3 window).
    local_mean = ndimage.uniform_filter(elevation, size=3)
    local_sq_mean = ndimage.uniform_filter(elevation**2, size=3)
    roughness = np.sqrt(np.clip(local_sq_mean - local_mean**2, 0.0, None))

    lethal = slope_deg > slope_lethal_deg

    half_world = world_size_m / 2.0
    rows, cols = elevation.shape
    for rock in manifest["rocks"]:
        row = int((rock["y_m"] + half_world) / actual_resolution_m)
        col = int((rock["x_m"] + half_world) / actual_resolution_m)
        radius_cells = max(1, int(round(rock["scale_m"] / actual_resolution_m)))
        r0, r1 = max(0, row - radius_cells), min(rows, row + radius_cells + 1)
        c0, c1 = max(0, col - radius_cells), min(cols, col + radius_cells + 1)
        lethal[r0:r1, c0:c1] = True

    # Inflate lethal cells by the rover's radius so the planner keeps the whole
    # footprint clear, not just its center point.
    inflation_cells = max(1, int(round(rover_radius_m / actual_resolution_m)))
    structure = ndimage.generate_binary_structure(2, 2)
    lethal_inflated = ndimage.binary_dilation(lethal, structure=structure, iterations=inflation_cells)

    # Non-lethal cost: normalized blend of slope and roughness, scaled to [0, 99].
    slope_cost = np.clip(slope_deg / slope_lethal_deg, 0.0, 1.0)
    roughness_cost = np.clip(roughness / (roughness.max() + 1e-9), 0.0, 1.0)
    cost = 0.6 * slope_cost + 0.4 * roughness_cost
    cost_grid = np.clip(cost * 99, 0, 99).astype(np.int8)
    cost_grid[lethal_inflated] = 100

    return cost_grid, actual_resolution_m, -half_world, -half_world


class CostmapNode(Node):
    def __init__(self):
        super().__init__("regolith_costmap")
        self.declare_parameter("manifest_path", "")
        self.declare_parameter("resolution_m", 1.0)
        self.declare_parameter("rover_radius_m", 0.3)
        self.declare_parameter("slope_lethal_deg", 20.0)

        resolution_m = self.get_parameter("resolution_m").value
        if resolution_m <= 0.0:
            self.get_logger().error(f"Parameter resolution_m must be > 0, got {resolution_m}")
            raise SystemExit(1)

        manifest_path = Path(self.get_parameter("manifest_path").value)
        try:
            manifest = json.loads(manifest_path.read_text())
            heightmap = np.array(Image.open(manifest["heightmap_png"])).astype(np.float64)
            heightmap = heightmap / heightmap.max() * manifest["height_range_m"]
        except (OSError, KeyError, ValueError) as error:
            self.get_logger().error(
                f"Failed to load terrain manifest '{manifest_path}': {error!r}. The "
                "regolith_bringup launch files generate it via regolith_terrain_gen; if it is "
                "stale or corrupted, delete ~/.cache/regolith/worlds/seed_<N> and relaunch."
            )
            raise SystemExit(1)

        cost_grid, resolution_m, origin_x, origin_y = build_costmap(
            manifest,
            heightmap,
            resolution_m,
            self.get_parameter("rover_radius_m").value,
            self.get_parameter("slope_lethal_deg").value,
        )
        self._msg = self._to_occupancy_grid(cost_grid, resolution_m, origin_x, origin_y)

        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._pub = self.create_publisher(OccupancyGrid, "/costmap", qos)
        self.create_timer(1.0, self._publish)
        self.get_logger().info(
            f"Published costmap: {cost_grid.shape[1]}x{cost_grid.shape[0]} cells "
            f"at {resolution_m:.3f} m/cell"
        )

    def _to_occupancy_grid(self, cost_grid: np.ndarray, resolution_m: float, origin_x: float, origin_y: float):
        msg = OccupancyGrid()
        msg.header.frame_id = "odom"
        msg.info.resolution = float(resolution_m)
        msg.info.width = int(cost_grid.shape[1])
        msg.info.height = int(cost_grid.shape[0])
        msg.info.origin.position.x = float(origin_x)
        msg.info.origin.position.y = float(origin_y)
        msg.info.origin.orientation.w = 1.0
        msg.data = cost_grid.flatten().astype(np.int8).tolist()
        return msg

    def _publish(self) -> None:
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._msg)


def main() -> None:
    rclpy.init()
    node = CostmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
