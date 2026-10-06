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
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSProfile
from scipy import ndimage
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool

from regolith_costmap.dem_degradation import bearing_from_param
from regolith_costmap.dem_degradation import degrade_dem
from regolith_costmap.dem_degradation import shift_vector_m
from regolith_costmap.hazcam_cloud import xy_from_cloud


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
    rock_shift_xy_m: tuple = (0.0, 0.0),
    rock_min_detect_diameter_m: float = 0.0,
) -> tuple:
    """Return (cost_grid uint8 [0-100], resolution_m, origin_x_m, origin_y_m).

    The last two arguments model a rock layer taken from orbit rather than read
    from the world manifest (docs/TRL4_BASELINE.md, gap G9). Rocks whose footprint
    diameter is below `rock_min_detect_diameter_m` are not in the map at all, and
    the rest sit at their true position displaced by `rock_shift_xy_m` - the same
    registration offset the DEM carries, because a mis-registered map is
    mis-registered everywhere on it. Defaults reproduce the perfect-oracle costmap.
    """
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
        # Prefer the rock's real horizontal collision footprint over scale_m, which is
        # the mesh BOUNDING radius and overstates these anisotropic boulders' thin axes.
        # Older manifests (pre-ellipsoid-collision) have no radii, so fall back.
        radii = rock.get("collision_radii_m")
        footprint_m = max(radii[0], radii[1]) if radii else rock["scale_m"]
        if 2.0 * footprint_m < rock_min_detect_diameter_m:
            continue
        row = int((rock["y_m"] + rock_shift_xy_m[1] + half_world) / actual_resolution_m)
        col = int((rock["x_m"] + rock_shift_xy_m[0] + half_world) / actual_resolution_m)
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


def build_apriori_costmap(
    manifest: dict,
    heightmap: np.ndarray,
    resolution_m: float,
    rover_radius_m: float,
    slope_lethal_deg: float,
    dem_post_m: float = 0.0,
    dem_noise_m: float = 0.0,
    dem_shift_m: float = 0.0,
    dem_shift_bearing_deg=None,
    dem_noise_seed: int = 0,
    rock_min_detect_diameter_m: float = 0.0,
) -> tuple:
    """`build_costmap` on the rover's a-priori map instead of on the generator's truth.

    The height layer comes from `degrade_dem` (the same call, seed and shift the
    terrain matcher makes, so both see one map) and the rock layer is filtered and
    displaced to match. Returns what `build_costmap` returns. With every argument
    at its default nothing is degraded and the result is identical to
    `build_costmap`'s, which is what keeps the shipped launch unchanged.
    """
    if dem_post_m or dem_noise_m or dem_shift_m:
        # Native post spacing, as terrain_relative_node computes it.
        native_m = float(manifest["world_size_m"]) / (heightmap.shape[0] - 1)
        heightmap, _ = degrade_dem(
            heightmap, native_m, post_m=dem_post_m, noise_m=dem_noise_m,
            shift_m=dem_shift_m, noise_seed=dem_noise_seed,
            shift_bearing_deg=dem_shift_bearing_deg,
        )
    shift_xy = shift_vector_m(dem_shift_m, dem_shift_bearing_deg) if dem_shift_m else (0.0, 0.0)
    return build_costmap(
        manifest, heightmap, resolution_m, rover_radius_m, slope_lethal_deg,
        rock_shift_xy_m=shift_xy, rock_min_detect_diameter_m=rock_min_detect_diameter_m,
    )


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


def world_to_cell(x_m: float, y_m: float, origin: tuple, resolution_m: float) -> tuple:
    """(row, col) of a point, the same int() truncation planner_node._world_to_grid uses."""
    return int((y_m - origin[1]) / resolution_m), int((x_m - origin[0]) / resolution_m)


def detection_cells(points_xy, origin: tuple, resolution_m: float, shape: tuple) -> set:
    """Distinct in-bounds (row, col) cells hit by one hazcam message's (x, y) points."""
    rows, cols = shape
    cells = set()
    for x_m, y_m in points_xy:
        if not (math.isfinite(x_m) and math.isfinite(y_m)):
            continue
        rc = world_to_cell(x_m, y_m, origin, resolution_m)
        if 0 <= rc[0] < rows and 0 <= rc[1] < cols:
            cells.add(rc)
    return cells


def goal_protection_mask(goal_xy, clearance_m: float, origin: tuple, resolution_m: float,
                         shape: tuple) -> np.ndarray:
    """Cells the hazcam layer may never make lethal.

    Every cell whose centre lies within clearance_m of the goal, and always the goal's
    own cell.

    The same failure flip_recovery_node's hazard_goal_clearance_m guards against for
    wedge marks: the goal is a fixed point, detections are placed through a drifting
    estimate, and one lethal goal cell makes planner_node refuse the goal for the rest
    of the run (PROGRESS.md, seed 55). Protecting by cell centre, including against
    inflation, keeps the goal cell and the ring around it free, so the hazcam layer on
    its own cannot wall the goal off. The cost is stated, not hidden: a real rock this
    close to the goal is not avoided, and the rover may find it by contact.
    """
    protected = np.zeros(shape, dtype=bool)
    if goal_xy is None:
        return protected
    rows, cols = shape
    rr, cc = np.mgrid[:rows, :cols]
    cx = origin[0] + (cc + 0.5) * resolution_m
    cy = origin[1] + (rr + 0.5) * resolution_m
    protected |= np.hypot(cx - goal_xy[0], cy - goal_xy[1]) <= clearance_m
    goal_rc = world_to_cell(goal_xy[0], goal_xy[1], origin, resolution_m)
    if 0 <= goal_rc[0] < rows and 0 <= goal_rc[1] < cols:
        protected[goal_rc] = True
    return protected


def mark_hazcam_cells(cost_grid: np.ndarray, cells: set, hits: dict, confirm_hits: int,
                      inflation_cells: int, protected: np.ndarray) -> list:
    """Count one message's detection cells; mark the newly confirmed ones lethal, inflated.

    A cell is confirmed once it has been reported by `confirm_hits` separate messages
    (frames). Marks are permanent, so one bad frame must not be enough to block a cell
    for the rest of the run; a real rock is seen in every frame for tens of seconds.
    Inflation is the static layer's: a (2k + 1)-square of k = ceil(rover_radius /
    resolution) cells, the 8-connected dilation build_costmap applies. Protected cells
    are skipped. Mutates cost_grid; returns the cells that went from non-lethal to
    lethal, which is what decides whether the current path needs a replan.

    Protected cells (goal_protection_mask) are never marked, and a detection IN a
    protected cell is dropped whole, inflation included - so detections near the
    goal cannot wall it off. A closed ring of real obstacles beyond the clearance
    still can, as it would in the a-priori map.
    """
    rows, cols = cost_grid.shape
    newly = []
    for rc in cells:
        hits[rc] = hits.get(rc, 0) + 1
        if hits[rc] < confirm_hits or protected[rc]:
            # A protected detection marks nothing, not even its inflation: inflating
            # from the cells round the goal would ring the goal cell with lethal ones.
            continue
        # Re-applied on every later report too (idempotent: lethal cells are skipped),
        # so a cell that was goal-protected when confirmed is marked once the mission
        # moves on to a goal it no longer protects.
        r0, r1 = max(0, rc[0] - inflation_cells), min(rows, rc[0] + inflation_cells + 1)
        c0, c1 = max(0, rc[1] - inflation_cells), min(cols, rc[1] + inflation_cells + 1)
        for r in range(r0, r1):
            for c in range(c0, c1):
                if protected[r, c] or cost_grid[r, c] >= 100:
                    continue
                cost_grid[r, c] = 100
                newly.append((r, c))
    return newly


def path_cells_ahead(path_xy, rover_xy, origin: tuple, resolution_m: float, shape: tuple) -> set:
    """Cells the path still has to cross.

    From the path vertex nearest the rover to its end, sampled at a quarter cell so a
    segment cannot skip a cell it passes through.
    """
    if path_xy is None or len(path_xy) == 0:
        return set()
    path = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
    start = 0
    if rover_xy is not None:
        start = int(np.argmin(np.hypot(path[:, 0] - rover_xy[0], path[:, 1] - rover_xy[1])))
    path = path[start:]
    samples = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(math.ceil(np.hypot(*(b - a)) / (resolution_m / 4.0))))
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        samples.append(a + t * (b - a))
    return detection_cells(np.vstack(samples), origin, resolution_m, shape)


class ReplanLimiter:
    """At most one hazcam-triggered replan per min_interval_s.

    A request inside the interval is held, not dropped, and fires as soon as the
    interval has passed.

    A rock is confirmed cell by cell as the rover closes on it, and every cell on the
    path would otherwise trigger its own replan: a storm of /goal_pose messages, each
    one restarting pure pursuit on a fresh path. Holding (rather than dropping) keeps
    the guarantee that a newly blocked path is always replanned, just not instantly.
    """

    def __init__(self, min_interval_s: float):  # noqa: D107
        self.min_interval_s = float(min_interval_s)
        self.pending = False
        self._last_s = None

    def request(self) -> None:
        """Ask for a replan."""
        self.pending = True

    def cancel(self) -> None:
        """Withdraw a held request (the path it was for is finished)."""
        self.pending = False

    def poll(self, now_s: float) -> bool:
        """Return True exactly when a held request should be sent now; records the send."""
        if not self.pending:
            return False
        if self._last_s is not None and now_s - self._last_s < self.min_interval_s:
            return False
        self.pending = False
        self._last_s = now_s
        return True


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
        # The a-priori map the planner sees (docs/TRL4_BASELINE.md, gap G9). All
        # default to "perfect": the generator's heightmap plus every rock at its
        # exact position, which is a hazard oracle no mission has. The dem_* values
        # mean exactly what they do on terrain_relative_node, and the registration
        # shift moves the rocks with the DEM. dem_shift_bearing_deg NaN = not set
        # = the legacy diagonal. rock_min_detect_diameter_m drops rocks too small
        # to be seen from orbit (0 = every rock).
        self.declare_parameter("dem_post_m", 0.0)
        self.declare_parameter("dem_noise_m", 0.0)
        self.declare_parameter("dem_shift_m", 0.0)
        self.declare_parameter("dem_shift_bearing_deg", float("nan"))
        self.declare_parameter("dem_noise_seed", 0)
        self.declare_parameter("rock_min_detect_diameter_m", 0.0)
        # Onboard hazard layer (docs/TRL4_BASELINE.md gap G10): obstacle points from
        # hazcam_node, in odom. Off by default, and when off nothing below subscribes
        # to anything, so the reference configuration's costmap is untouched.
        self.declare_parameter("hazcam", False)
        self.declare_parameter("hazcam_topic", "/hazcam/obstacles")
        self.declare_parameter("hazcam_confirm_hits", 2)
        # Exceeds the 1 m cell by half a cell, so the goal cell and its 4-neighbours
        # always stay free. Same value as flip_recovery_node's hazard_goal_clearance_m.
        self.declare_parameter("hazcam_goal_clearance_m", 1.5)
        # 5 s of sim time is 1 m at the 0.2 m/s cruise speed, against detections that
        # arrive from up to 6 m ahead: the planner is never late by more than a metre.
        self.declare_parameter("hazcam_replan_min_interval_s", 5.0)

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

        dem_post_m = float(self.get_parameter("dem_post_m").value)
        dem_noise_m = float(self.get_parameter("dem_noise_m").value)
        dem_shift_m = float(self.get_parameter("dem_shift_m").value)
        min_rock_m = float(self.get_parameter("rock_min_detect_diameter_m").value)
        cost_grid, resolution_m, origin_x, origin_y = build_apriori_costmap(
            manifest,
            heightmap,
            resolution_m,
            self.get_parameter("rover_radius_m").value,
            self.get_parameter("slope_lethal_deg").value,
            dem_post_m=dem_post_m,
            dem_noise_m=dem_noise_m,
            dem_shift_m=dem_shift_m,
            dem_shift_bearing_deg=bearing_from_param(
                self.get_parameter("dem_shift_bearing_deg").value),
            dem_noise_seed=int(self.get_parameter("dem_noise_seed").value),
            rock_min_detect_diameter_m=min_rock_m,
        )
        if dem_post_m or dem_noise_m or dem_shift_m or min_rock_m:
            self.get_logger().warn(
                f"COSTMAP BUILT FROM A DEGRADED A-PRIORI MAP: posts {dem_post_m:.2f} m, "
                f"noise {dem_noise_m:.2f} m, registration shift {dem_shift_m:.2f} m, "
                f"rocks under {min_rock_m:.2f} m diameter omitted. Not a perfect-oracle costmap."
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
        if self.get_parameter("hazcam").value:
            self._init_hazcam()
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

    def _init_hazcam(self) -> None:
        """Subscriptions and state for the onboard hazard layer.

        Why the replan lives here. planner_node plans only when a /goal_pose arrives,
        and pure_pursuit_node replans only on deviation or stall, so a costmap change
        on its own moves nothing: the rover keeps following the old path into the
        rock it has just seen. Wedge hazards get their replan from flip_recovery_node
        re-sending the goal after an escape. Hazcam marks get theirs from here, the
        only node that knows which cells just became lethal: if any of them lies on
        the part of /planned_path still ahead of the rover, the active goal is
        re-sent, rate-limited by ReplanLimiter, and never while the recovery node
        owns the rover (it replans itself when its maneuver ends).
        """
        self._hazcam_hits = {}
        self._hazcam_marked = 0
        self._hazcam_msgs = 0
        self._hazcam_skipped_protected = 0
        self._hazcam_replans = 0
        self._goal = None  # last /goal_pose message
        self._goal_xy = None
        self._protected = goal_protection_mask(
            None, 0.0, self._origin, self._resolution_m, self._grid.shape)
        self._path_xy = None
        self._path_active = False
        self._rover_xy = None
        self._recovery_active = False
        self._replan = ReplanLimiter(self.get_parameter("hazcam_replan_min_interval_s").value)

        path_qos = QoSProfile(depth=1)
        path_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            PointCloud2, self.get_parameter("hazcam_topic").value, self._on_hazcam, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self.create_subscription(PathMsg, "/planned_path", self._on_path, path_qos)
        self.create_subscription(Odometry, "/odometry/filtered", self._on_odometry, 10)
        self.create_subscription(Bool, "/recovery_active", self._on_recovery_active, 10)
        self.create_subscription(Bool, "/goal_reached", self._on_goal_reached, 10)
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.create_timer(0.5, self._hazcam_replan_tick)
        self.create_timer(30.0, self._hazcam_report)
        self.get_logger().info(
            "Hazcam layer ON: obstacle points on "
            f"{self.get_parameter('hazcam_topic').value} are marked lethal after "
            f"{self.get_parameter('hazcam_confirm_hits').value} frames, never within "
            f"{self.get_parameter('hazcam_goal_clearance_m').value:.1f} m of the goal."
        )

    def _on_goal(self, msg: PoseStamped) -> None:
        xy = (msg.pose.position.x, msg.pose.position.y)
        self._goal = msg
        if xy != self._goal_xy:
            self._goal_xy = xy
            self._protected = goal_protection_mask(
                xy, self.get_parameter("hazcam_goal_clearance_m").value,
                self._origin, self._resolution_m, self._grid.shape)

    def _on_path(self, msg: PathMsg) -> None:
        self._path_xy = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self._path_active = bool(self._path_xy)

    def _on_odometry(self, msg: Odometry) -> None:
        self._rover_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _on_recovery_active(self, msg: Bool) -> None:
        self._recovery_active = bool(msg.data)

    def _on_goal_reached(self, msg: Bool) -> None:
        if msg.data:
            # The path is spent. Re-sending the goal now would make the planner hand
            # pure pursuit a fresh path and drive an arrived rover again.
            self._path_active = False
            self._replan.cancel()

    def _on_hazcam(self, msg: PointCloud2) -> None:
        self._hazcam_msgs += 1
        try:
            points = xy_from_cloud(msg)
        except ValueError as error:
            self.get_logger().error(f"Ignoring hazcam message: {error}")
            return
        if msg.header.frame_id and msg.header.frame_id != "odom":
            self.get_logger().error(
                f"Ignoring hazcam message in frame '{msg.header.frame_id}': the costmap "
                "is in odom and this node does no TF lookups")
            return
        cells = detection_cells(points, self._origin, self._resolution_m, self._grid.shape)
        protected_hits = sum(1 for rc in cells if self._protected[rc])
        self._hazcam_skipped_protected += protected_hits
        rover_radius_m = self.get_parameter("rover_radius_m").value
        inflation_cells = max(1, math.ceil(rover_radius_m / self._resolution_m))
        newly = mark_hazcam_cells(
            self._grid, cells, self._hazcam_hits,
            int(self.get_parameter("hazcam_confirm_hits").value), inflation_cells,
            self._protected)
        if not newly:
            return
        self._hazcam_marked += len(newly)
        self._msg = self._to_occupancy_grid(
            self._grid, self._resolution_m, self._origin[0], self._origin[1])
        self._publish()
        ahead = path_cells_ahead(
            self._path_xy if self._path_active else None, self._rover_xy,
            self._origin, self._resolution_m, self._grid.shape)
        blocking = ahead.intersection(newly)
        if blocking:
            self._replan.request()
        note = f", {len(blocking)} on the path ahead - replan requested" if blocking else ""
        # The detected cells (before inflation), in odom: what an offline check
        # compares against the world's rocks to count true and false detections.
        seen = ", ".join(
            f"({self._origin[0] + (c + 0.5) * self._resolution_m:.1f},"
            f"{self._origin[1] + (r + 0.5) * self._resolution_m:.1f})"
            for r, c in sorted(cells))
        # Sim time and the rover's own estimate at the time, so a detection can be
        # lined up against the recorded signals (attitude transients, turns) when
        # diagnosing false positives offline.
        sim_t = self.get_clock().now().nanoseconds / 1e9
        rover = (f"({self._rover_xy[0]:.1f},{self._rover_xy[1]:.1f})"
                 if self._rover_xy is not None else "unknown")
        self.get_logger().info(f"Hazcam: {len(newly)} cells newly lethal{note}; detected at [{seen}]; "
                               f"sim_t={sim_t:.2f} rover_est={rover}")

    def _hazcam_replan_tick(self) -> None:
        # Deferred to a timer rather than sent from the callback: the updated costmap
        # has already gone out, and the planner reads the latest /costmap when the goal
        # arrives, so the replan is always against the map that contains the rock.
        if self._recovery_active or self._goal is None or not self._path_active:
            return
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if not self._replan.poll(now_s):
            return
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self._goal.header.frame_id or "odom"
        goal.pose = self._goal.pose
        self._hazcam_replans += 1
        self._goal_pub.publish(goal)
        self.get_logger().warn(
            f"Hazcam replan #{self._hazcam_replans}: an obstacle seen onboard blocks the "
            "current path - re-sending the goal")

    def _hazcam_report(self) -> None:
        self.get_logger().info(
            f"Hazcam layer: {self._hazcam_msgs} messages, {len(self._hazcam_hits)} cells "
            f"reported, {self._hazcam_marked} cells marked lethal (incl. inflation), "
            f"{self._hazcam_skipped_protected} goal-protected cell reports skipped, "
            f"{self._hazcam_replans} replans."
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
