#!/usr/bin/env python3
# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Marks the start and the mission goals in both windows: flags in Gazebo, markers in
RViz.

Watching this demo, there was no way to tell where the rover set off from or where it
is trying to get to - the Gazebo view is 200 m of grey regolith, and RViz showed a
planned path with nothing at either end of it. This node puts the same three things in
both views:

  * START - where the rover began, from the terrain manifest's spawn zone.
  * The planned waypoints, numbered, if the running mission publishes a list of them
    on ``/mission_waypoints`` (``tour_mission.py`` does).
  * The ACTIVE goal - whatever was last published on ``/goal_pose``. This one moves,
    and it is the only marker that appears for an ad-hoc goal (RViz's "2D Goal Pose"
    tool, or the M4 acceptance harness), which publish a goal without any waypoint list.

WHICH FRAME THESE ARE IN, because it is not a detail here. Goals are published in the
``odom`` frame; the Gazebo flags are placed in the world frame. Those two share an
origin at the spawn point, so a flag stands at the goal's true world position - but the
rover's *estimate* of where it is drifts from ground truth as the run goes on (M4
measured 4.3 m after a single wedging event). So a rover parked visibly short of a flag
while reporting the goal reached is not a bug in the flag; it is the localisation error,
made visible. That is worth being able to see, which is part of why the flags are placed
this way round rather than being drawn wherever the EKF currently thinks the goal is.

Flags are VISUAL ONLY - no <collision>. The rover drives through them. A decoration the
rover can wedge itself on would be a new failure mode invented by a debugging aid.

Gazebo entities are created through ``gz service`` calls, the same route
``flip_recovery_node`` uses for set_pose. Every call is best-effort: this node is a
visualisation, and it must never be able to take the demo down with it.
"""

import json
import math
from pathlib import Path
import subprocess

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMsg
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import Bool
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# Bright and emissive on purpose: the scene is lit by a 12 deg sun and is very dark, so
# a flag with an ordinary diffuse material reads as a grey smudge.
START_RGB = (0.10, 0.90, 0.25)
PENDING_RGB = (0.95, 0.65, 0.10)
ACTIVE_RGB = (1.00, 0.25, 0.10)
REACHED_RGB = (0.35, 0.55, 0.95)

POLE_M = 2.2
ACTIVE_POLE_M = 3.0  # the one that matters right now stands taller than the rest
BANNER_W = 0.62
BANNER_H = 0.42

# Goals are matched to planned waypoints by proximity, so a re-published or slightly
# nudged goal does not spawn a duplicate flag next to an existing one.
SAME_WAYPOINT_M = 1.0


def waypoint_near(waypoints, x: float, y: float):
    """Index of the planned waypoint a published goal belongs to, or None.

    By proximity, not equality: pure pursuit and flip recovery both re-publish the
    current goal, and a goal that came back a few centimetres off would otherwise
    sprout a second flag beside the one already standing there.
    """
    for i, (wx, wy) in enumerate(waypoints):
        if math.hypot(wx - x, wy - y) <= SAME_WAYPOINT_M:
            return i
    return None


def _flag_sdf(name, x, y, z, rgb, height=POLE_M) -> str:
    """One flag, as a single line of SDF - newlines cannot be embedded in the protobuf
    text-format string field the create service takes."""
    r, g, b = rgb
    return (
        f'<sdf version="1.10"><model name="{name}"><static>true</static>'
        f'<pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose><link name="link">'
        f'<visual name="pole"><pose>0 0 {height / 2:.3f} 0 0 0</pose>'
        f"<geometry><cylinder><radius>0.035</radius><length>{height:.3f}</length>"
        f"</cylinder></geometry><material><ambient>0.7 0.7 0.7 1</ambient>"
        f"<diffuse>0.8 0.8 0.8 1</diffuse><emissive>0.30 0.30 0.32 1</emissive>"
        f'</material></visual><visual name="banner">'
        f"<pose>{BANNER_W / 2:.3f} 0 {height - BANNER_H / 2 - 0.07:.3f} 0 0 0</pose>"
        f"<geometry><box><size>{BANNER_W} 0.015 {BANNER_H}</size></box></geometry>"
        f"<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>"
        f"<emissive>{r * 0.85:.3f} {g * 0.85:.3f} {b * 0.85:.3f} 1</emissive>"
        f"</material></visual></link></model></sdf>"
    )


class MissionMarkersNode(Node):
    def __init__(self):
        super().__init__("regolith_mission_markers")
        self.declare_parameter("world_name", "regolith_moon")
        self.declare_parameter("manifest_path", "")
        self.declare_parameter("gazebo_flags", True)

        self._world = self.get_parameter("world_name").value
        self._use_gazebo = bool(self.get_parameter("gazebo_flags").value)

        self._elevation = None
        self._start_xy = (0.0, 0.0)
        self._load_terrain()

        self._waypoints = []  # planned, from /mission_waypoints
        self._reached = set()  # indices of waypoints already visited
        self._active = None  # (x, y) of the current /goal_pose
        self._active_index = None  # which planned waypoint it is, if any
        self._spawned = set()  # Gazebo model names created so far
        # Flags asked for but not yet accepted by gz. The world takes ~20 s to load
        # 190 rocks and the terrain mesh, and its create service refuses entities until
        # it is up - so a flag requested at startup is normally rejected once. First
        # cut only retried the start flag, and the whole waypoint set was silently lost.
        self._pending = {}
        self._attempts = {}

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._marker_pub = self.create_publisher(MarkerArray, "/mission_markers", latched)
        self.create_subscription(PathMsg, "/mission_waypoints", self._on_waypoints, latched)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self.create_subscription(Bool, "/goal_reached", self._on_reached, 10)

        self._request_flag("mission_flag_start", *self._start_xy, START_RGB)
        self.create_timer(2.0, self._flush_pending)
        self._publish_markers()
        self.get_logger().info(
            f"Mission markers active - start at ({self._start_xy[0]:.1f}, "
            f"{self._start_xy[1]:.1f}), Gazebo flags "
            f"{'on' if self._use_gazebo else 'off'}, RViz on /mission_markers"
        )

    # ---- terrain ---------------------------------------------------------------

    def _load_terrain(self) -> None:
        """Flags stand on the ground gz DRAWS, read back from the shipped terrain mesh.

        Falls back to the spawn elevation for the whole world if the mesh is missing,
        which puts flags at a plausible height instead of burying them at z = 0.
        """
        raw = self.get_parameter("manifest_path").value
        if not raw:
            self.get_logger().warn("no manifest_path - flags will sit at z = 0")
            return
        try:
            manifest = json.loads(Path(raw).read_text())
            spawn = manifest["spawn_zone"]
            self._start_xy = (float(spawn["x_m"]), float(spawn["y_m"]))
            # Installed first, deliberately: if reading the mesh below throws, this is
            # what the flags fall back to - one plausible height everywhere, rather
            # than every flag buried at z = 0.
            flat = float(spawn["elevation_m"])
            self._elevation = lambda x, y: flat

            from regolith_terrain_gen.terrain_mesh import load_drawn_surface

            self._elevation = load_drawn_surface(Path(manifest["terrain_mesh_obj"]))
        except Exception as exc:  # noqa: BLE001 - a marker node must not break a run
            self.get_logger().warn(f"could not read the terrain surface ({exc!r})")

    def _ground(self, x: float, y: float) -> float:
        if self._elevation is None:
            return 0.0
        try:
            return float(self._elevation(x, y))
        except Exception:  # noqa: BLE001
            return 0.0

    # ---- subscriptions ---------------------------------------------------------

    def _on_waypoints(self, msg: PathMsg) -> None:
        self._waypoints = [(float(p.pose.position.x), float(p.pose.position.y)) for p in msg.poses]
        self.get_logger().info(f"Mission has {len(self._waypoints)} planned waypoints")
        self._place_waypoint_flags()
        self._publish_markers()

    def _on_goal(self, msg: PoseStamped) -> None:
        self._active = (float(msg.pose.position.x), float(msg.pose.position.y))
        self._active_index = self._waypoint_near(*self._active)
        self._move_active_flag()
        self._publish_markers()

    def _on_reached(self, msg: Bool) -> None:
        if not msg.data or self._active is None:
            return
        if self._active_index is not None:
            self._reached.add(self._active_index)
        self._publish_markers()

    def _waypoint_near(self, x: float, y: float):
        return waypoint_near(self._waypoints, x, y)

    # ---- Gazebo ----------------------------------------------------------------

    def _gz(self, service: str, reqtype: str, req: str) -> bool:
        cmd = [
            "gz",
            "service",
            "-s",
            f"/world/{self._world}/{service}",
            "--reqtype",
            reqtype,
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            req,
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"gz {service} call raised: {exc}")
            return False
        return out.returncode == 0 and "true" in out.stdout.lower()

    def _request_flag(self, name, x, y, rgb, height=POLE_M) -> None:
        """Queue a flag. _flush_pending places it as soon as gz will take it."""
        if not self._use_gazebo or name in self._spawned:
            return
        self._pending[name] = (x, y, rgb, height)

    def _flush_pending(self) -> None:
        for name, (x, y, rgb, height) in list(self._pending.items()):
            sdf = _flag_sdf(name, x, y, self._ground(x, y), rgb, height)
            escaped = sdf.replace("\\", "\\\\").replace('"', '\\"')
            if self._gz(
                "create",
                "gz.msgs.EntityFactory",
                f'sdf: "{escaped}" name: "{name}" allow_renaming: false',
            ):
                self._spawned.add(name)
                del self._pending[name]
                continue
            # Retry quietly while the world is still loading, then say so once. Silence
            # is what hid the first version of this bug: the calls failed and nothing
            # anywhere reported it.
            self._attempts[name] = self._attempts.get(name, 0) + 1
            if self._attempts[name] == 15:
                self.get_logger().warn(
                    f"gz has refused to create '{name}' {self._attempts[name]} times - "
                    f"giving up on that flag (the run is unaffected)"
                )
            if self._attempts[name] >= 15:
                del self._pending[name]

    def _is_the_start(self, x: float, y: float) -> bool:
        """The tour's last waypoint is the way home, so it lands on the start.

        Two flags at one position z-fight, and which banner wins is arbitrary - the
        green START flag came out amber in Gazebo because the waypoint flag was drawn
        over it. The start already marks that spot; the duplicate is dropped.
        """
        return math.hypot(x - self._start_xy[0], y - self._start_xy[1]) <= SAME_WAYPOINT_M

    def _place_waypoint_flags(self) -> None:
        for i, (x, y) in enumerate(self._waypoints):
            if self._is_the_start(x, y):
                continue
            self._request_flag(f"mission_flag_wp_{i + 1}", x, y, PENDING_RGB)

    def _move_active_flag(self) -> None:
        """One flag, moved. Recolouring an existing Gazebo visual has no service in this
        install, so 'which goal is live' is carried by a separate taller flag that
        follows the active goal rather than by restyling the waypoint flags."""
        if not self._use_gazebo or self._active is None:
            return
        x, y = self._active
        if "mission_flag_active" not in self._spawned:
            self._request_flag("mission_flag_active", x, y, ACTIVE_RGB, ACTIVE_POLE_M)
            return
        self._gz(
            "set_pose",
            "gz.msgs.Pose",
            f'name: "mission_flag_active" '
            f"position {{ x: {x:.3f} y: {y:.3f} z: {self._ground(x, y):.3f} }} "
            f"orientation {{ x: 0 y: 0 z: 0 w: 1 }}",
        )

    # ---- RViz ------------------------------------------------------------------

    def _publish_markers(self) -> None:
        array = MarkerArray()
        array.markers.append(self._delete_all())
        finishes_here = any(self._is_the_start(x, y) for x, y in self._waypoints)
        array.markers += self._flag_markers(
            "start",
            0,
            *self._start_xy,
            START_RGB,
            "START / FINISH" if finishes_here else "START",
            POLE_M,
        )

        for i, (x, y) in enumerate(self._waypoints):
            if self._is_the_start(x, y):
                continue  # already drawn, as the start
            if i in self._reached:
                rgb, label = REACHED_RGB, f"G{i + 1} done"
            elif i == self._active_index:
                rgb, label = ACTIVE_RGB, f"G{i + 1} ACTIVE"
            else:
                rgb, label = PENDING_RGB, f"G{i + 1}"
            array.markers += self._flag_markers("waypoint", i + 1, x, y, rgb, label, POLE_M)

        # An ad-hoc goal belongs to no waypoint list, so it gets its own marker.
        if self._active is not None and self._active_index is None:
            array.markers += self._flag_markers(
                "goal", 0, *self._active, ACTIVE_RGB, "GOAL", ACTIVE_POLE_M
            )
        self._marker_pub.publish(array)

    def _delete_all(self) -> Marker:
        marker = Marker()
        marker.action = Marker.DELETEALL
        return marker

    def _flag_markers(self, namespace, index, x, y, rgb, label, height) -> list:
        """A flag in RViz is three markers: pole, banner, floating label.

        RViz's fixed frame is odom and goals arrive in odom, so these are published in
        odom unchanged - no elevation lookup, because RViz has no terrain to stand on
        and z = 0 is the odom plane the costmap and path are already drawn in.
        """
        colour = ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=1.0)

        pole = Marker()
        pole.header.frame_id = "odom"
        pole.ns = f"{namespace}_pole"
        pole.id = index
        pole.type = Marker.CYLINDER
        pole.action = Marker.ADD
        pole.pose.position.x = float(x)
        pole.pose.position.y = float(y)
        pole.pose.position.z = height / 2.0
        pole.pose.orientation.w = 1.0
        pole.scale.x = pole.scale.y = 0.07
        pole.scale.z = height
        pole.color = ColorRGBA(r=0.85, g=0.85, b=0.85, a=1.0)

        banner = Marker()
        banner.header.frame_id = "odom"
        banner.ns = f"{namespace}_banner"
        banner.id = index
        banner.type = Marker.CUBE
        banner.action = Marker.ADD
        banner.pose.position.x = float(x) + BANNER_W / 2.0
        banner.pose.position.y = float(y)
        banner.pose.position.z = height - BANNER_H / 2.0 - 0.07
        banner.pose.orientation.w = 1.0
        banner.scale.x = BANNER_W
        banner.scale.y = 0.02
        banner.scale.z = BANNER_H
        banner.color = colour

        text = Marker()
        text.header.frame_id = "odom"
        text.ns = f"{namespace}_label"
        text.id = index
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(x)
        text.pose.position.y = float(y)
        text.pose.position.z = height + 0.45
        text.pose.orientation.w = 1.0
        text.scale.z = 0.55
        text.color = colour
        text.text = label

        return [pole, banner, text]


def main() -> None:
    rclpy.init()
    node = MissionMarkersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
