# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Publishes a 2.5D traversability costmap from the known terrain heightmap.

Per the plan's honest simplifications: the costmap comes from the generated
terrain heightmap (the world is known a priori for this PoC), not from
onboard perception. Sensor-derived costmaps are the explicit next milestone.
"""

import json
import math
from pathlib import Path

from PIL import Image
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSProfile
from scipy import ndimage


def load_heightmap(manifest: dict) -> np.ndarray:
    """Decode the terrain heightmap PNG into this module's [row = y, col = x] array, in absolute metres.

    The vertical decode uses the manifest's `heightmap_z_min_m` / `heightmap_z_span_m`,
    which is what the PNG was actually encoded against (save_heightmap_png writes the
    surface full-range, 0 -> z_min and 65535 -> z_min + span). It is deliberately NOT
    `height_range_m`: that is the range the generator was configured to be able to use,
    and a given seed's surface occupies less of it - 8.02 / 8.34 / 8.19 m of the
    configured 10.0 on seeds 42 / 7 / 123. Scaling to 10.0 stretched every elevation by
    that ratio and so overstated every slope by 1.20-1.25x, which ran the costmap at an
    effective ~16 deg lethal threshold against the 20 deg configured. The error was
    conservative (it over-flagged, never under-flagged), which is why it survived a full
    milestone without ever surfacing as a failure. See PROGRESS.md "costmap decodes the
    wrong height span".

    The `.T` is not cosmetic. The PNG is written in gz's axis order - its first image
    axis is world X - because that is what gz's <heightmap> requires (see
    regolith_terrain_gen's heightmap.save_heightmap_png, and the orientation regression
    test beside it). Everything on this side, including the rock stamping below and the
    planner's _world_to_grid, indexes [row = y, col = x]. Reading the file without the
    transpose mirrors the entire slope field about the x = y diagonal while the rocks -
    which come from the manifest's real x/y - stay put, so the planner routes around
    steep ground that isn't there and straight into ground that is. Measured on seed 42
    at the parameters hello_moon.launch.py uses (1.0 m, 0.3 m, 20 deg): 1.73% of cells
    get the wrong lethal verdict, and the total lethal fraction is unchanged (a
    transpose preserves the histogram), which is exactly why it is invisible in any
    summary statistic. See PROGRESS.md "Rendered terrain was transposed".
    """
    try:
        z_min = manifest["heightmap_z_min_m"]
        z_span = manifest["heightmap_z_span_m"]
    except KeyError as exc:
        # Fail loudly rather than falling back to height_range_m: a silent fallback would
        # reinstate the slope overstatement above with nothing on screen to show for it.
        # Every launch regenerates the world, so this only hits a stale cached manifest.
        raise KeyError(
            f"manifest is missing {exc} - it predates the heightmap z-span fix. "
            "Regenerate the world (any launch does this, or `ros2 run regolith_terrain_gen "
            "generate --seed N`) instead of decoding it against height_range_m."
        ) from exc

    pixels = np.array(Image.open(manifest["heightmap_png"])).astype(np.float64).T
    return z_min + pixels / 65535.0 * z_span


def build_costmap(
    manifest: dict,
    heightmap: np.ndarray,
    resolution_m: float,
    rover_radius_m: float,
    slope_lethal_deg: float,
) -> tuple:
    """Return (cost_grid uint8 [0-100], resolution_m, origin_x_m, origin_y_m)."""
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
        # Prefer the rock's real horizontal collision footprint over scale_m, which is
        # the mesh BOUNDING radius and overstates these anisotropic boulders' thin axes.
        # Older manifests (pre-ellipsoid-collision) have no radii, so fall back.
        radii = rock.get("collision_radii_m")
        footprint_m = max(radii[0], radii[1]) if radii else rock["scale_m"]
        radius_cells = max(1, int(round(footprint_m / actual_resolution_m)))
        r0, r1 = max(0, row - radius_cells), min(rows, row + radius_cells + 1)
        c0, c1 = max(0, col - radius_cells), min(cols, col + radius_cells + 1)
        lethal[r0:r1, c0:c1] = True

    # Inflate lethal cells by the rover's radius so the planner keeps the whole
    # footprint clear, not just its center point.
    #
    # CEIL, not round: this is a safety margin, and rounding a margin DOWN spends it.
    # `round` under-inflates whenever the radius is not close to a whole number of
    # cells - at a 0.25 m costmap it gives 1 cell (0.25 m) for a 0.30 m rover, which
    # is a planner that will happily route a corner of the rover through a boulder.
    # At the shipped 0.781 m/cell both give 1 cell, so this changes nothing about any
    # measurement recorded so far; it is the finer resolutions that were wrong.
    #
    # Note what the quantisation means for the parameter: at 0.781 m/cell, every
    # rover_radius_m from 0 to 0.78 m produces the same single cell of inflation.
    # Tuning it inside that range does nothing at all. The rover's real circumscribed
    # radius is 0.347 m (0.52 x 0.46 m footprint, from regolith_rover_description),
    # so one cell already covers it with room to spare.
    inflation_cells = max(1, math.ceil(rover_radius_m / actual_resolution_m))
    structure = ndimage.generate_binary_structure(2, 2)
    lethal_inflated = ndimage.binary_dilation(
        lethal, structure=structure, iterations=inflation_cells
    )

    # Non-lethal cost: normalized blend of slope and roughness, scaled to [0, 99].
    slope_cost = np.clip(slope_deg / slope_lethal_deg, 0.0, 1.0)
    roughness_cost = np.clip(roughness / (roughness.max() + 1e-9), 0.0, 1.0)
    cost = 0.6 * slope_cost + 0.4 * roughness_cost
    cost_grid = np.clip(cost * 99, 0, 99).astype(np.int8)
    cost_grid[lethal_inflated] = 100

    return cost_grid, actual_resolution_m, -half_world, -half_world


def stamp_hazard(cost_grid: np.ndarray, row: int, col: int, radius_cells: int) -> int:
    """Mark a lethal disc of `radius_cells` around (row, col). Returns cells marked.

    Mutates the grid in place - hazards accumulate over a run, and a later one
    must not undo an earlier one.
    """
    rows, cols = cost_grid.shape
    rr, cc = np.ogrid[:rows, :cols]
    disc = (rr - row) ** 2 + (cc - col) ** 2 <= radius_cells**2
    cost_grid[disc] = 100
    return int(disc.sum())


class CostmapNode(Node):
    def __init__(self):
        super().__init__("regolith_costmap")
        self.declare_parameter("manifest_path", "")
        self.declare_parameter("resolution_m", 1.0)
        self.declare_parameter("rover_radius_m", 0.3)
        self.declare_parameter("slope_lethal_deg", 20.0)
        # Learned keep-out zones: places the rover actually got wedged. The
        # a-priori map above knows every rock's footprint but not whether a gap
        # between two of them is really drivable, so a wedge is information the
        # map did not have - without recording it the planner keeps routing
        # through the same gap after every recovery (see PROGRESS.md's res40
        # M4 failure: 64 stuck events, all of them re-approaching the obstacle
        # that caused the previous one). Marking hazards is what a real rover's
        # FDIR does with a bumper/stall event, not a simulation shortcut.
        self.declare_parameter("hazard_radius_m", 1.2)

        resolution_m = self.get_parameter("resolution_m").value
        if resolution_m <= 0.0:
            self.get_logger().error(f"Parameter resolution_m must be > 0, got {resolution_m}")
            raise SystemExit(1)

        manifest_path = Path(self.get_parameter("manifest_path").value)
        try:
            manifest = json.loads(manifest_path.read_text())
            heightmap = load_heightmap(manifest)
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
        self._base_grid = cost_grid  # a-priori map, never overwritten
        self._grid = cost_grid.copy()  # base + learned hazards
        self._resolution_m = resolution_m
        self._origin = (origin_x, origin_y)
        self._hazards = []
        self._msg = self._to_occupancy_grid(self._grid, resolution_m, origin_x, origin_y)

        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._pub = self.create_publisher(OccupancyGrid, "/costmap", qos)
        self.create_subscription(PointStamped, "/hazard/stuck_point", self._on_hazard, 10)
        self.create_timer(1.0, self._publish)
        self.get_logger().info(
            f"Published costmap: {cost_grid.shape[1]}x{cost_grid.shape[0]} cells "
            f"at {resolution_m:.3f} m/cell"
        )

    def _on_hazard(self, msg: PointStamped) -> None:
        """Stamps a lethal disc where the rover reported getting wedged.

        The point arrives in the estimator's frame (that is the frame the
        planner and this costmap both work in), so a drifting estimate marks a
        drifting hazard - deliberately: the marking only has to be consistent
        with the frame the planner routes in for the rover to stop retrying the
        same approach.
        """
        radius_m = self.get_parameter("hazard_radius_m").value
        rover_radius_m = self.get_parameter("rover_radius_m").value
        rows, cols = self._grid.shape
        col = int((msg.point.x - self._origin[0]) / self._resolution_m)
        row = int((msg.point.y - self._origin[1]) / self._resolution_m)
        if not (0 <= row < rows and 0 <= col < cols):
            self.get_logger().warn(
                f"Ignoring hazard at ({msg.point.x:.1f}, {msg.point.y:.1f}) - outside the costmap"
            )
            return

        # Inflate by the rover radius here, the same way build_costmap inflates
        # the a-priori lethal cells, so the whole footprint clears the hazard.
        cells = max(1, int(round((radius_m + rover_radius_m) / self._resolution_m)))
        marked = stamp_hazard(self._grid, row, col, cells)
        self._hazards.append((msg.point.x, msg.point.y))
        self._msg = self._to_occupancy_grid(
            self._grid, self._resolution_m, self._origin[0], self._origin[1]
        )
        self._publish()
        self.get_logger().warn(
            f"Hazard #{len(self._hazards)} marked at ({msg.point.x:.2f}, {msg.point.y:.2f}): "
            f"{marked} cells lethal ({radius_m:.1f} m + {rover_radius_m:.1f} m rover "
            "radius). The rover got wedged here; the planner will route around it from now on."
        )

    def _to_occupancy_grid(
        self, cost_grid: np.ndarray, resolution_m: float, origin_x: float, origin_y: float
    ):
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
