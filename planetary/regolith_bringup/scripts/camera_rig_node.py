#!/usr/bin/env python3
# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Drives the free-flying cinematic camera rig (`cine_rig` launch arg), for documentary-
style footage of a run - chase, orbit, crane/reveal and tracking/lockoff shots.

Mechanism, verified directly rather than assumed (see docs/media/README.md's "camera
rig" section for the full probe writeup): the rig is an ordinary camera-only model in
world.sdf, decoupled from the rover by construction (no joint, no shared link - see
hello_moon.launch.py's `_bake_cine_rig_sdf`), and this node moves it by calling gz-sim's
`/world/<world>/set_pose` service once per tick - the exact mechanism
`flip_recovery_node.py` already uses in production to teleport the rover upright after a
flip, reused here rather than invented fresh. Confirmed on this install (gz-sim 8.14,
Harmonic) three ways: the teleported pose reads back correctly through a bridged
PosePublisher, the RENDERED IMAGE visibly changes (not just the ECM pose - checked by
placing distinct objects in a scene and eyeballing the frames), and it works whether the
rig model is `<static>true</static>` or non-static with gravity disabled - so this file
ships the simpler static rig.

Why a per-tick service call is fast enough: this world renders at roughly 0.05-0.10x
real time with a 720p camera attached, so a 30 Hz SIMULATED update rate (this node's
default - see below for why 15 Hz was not enough) is only 1.5-3.0 Hz of WALL-CLOCK
calls needed. The mechanism here shells out to the `gz service` CLI once per tick
(same pattern `flip_recovery_node.py` already uses in production, kept identical
rather than switched to the lower-level gz-transport Python bindings, which measured
~1000x faster in isolation but add a partition-matching failure mode this node
doesn't need to own - see the probe writeup). Measured cost of that CLI call on this
install: ~300 ms, dominated by process startup, not by the service itself. At 30 Hz
sim-time that leaves as little as ~50-350 ms of wall-clock headroom per tick across
the measured RTF range (0.057-0.093) - workable but noticeably tighter than 15 Hz's
margin was; measured the actual delivered tick rate on a real capture rather than
assumed it holds (see docs/media/README.md's "frame count, not container fps"
section) before shipping any hero take at this rate. If measurement ever shows the
CLI call falling behind, the fix is switching this one call to the gz-transport
Python binding (already proven ~1000x faster in the probe), not lowering the rate.

Why 30 Hz and not 15: a clip's SMOOTHNESS is set by how many DISTINCT rendered
frames exist per second of simulated time, which is this sensor's `update_rate` (see
`_cine_rig_sdf` in hello_moon.launch.py - it must match this node's rate, not just
be close, or every other output frame repeats the same rig pose). A first pass at
15 Hz produced a 1.04 s clip with only 26 distinct frames once measured - it read as
visibly stuttery on inspection. 30 Hz was chosen to match a normal cinematic frame
rate 1:1 so no output frame is a duplicate.

Smoothing: everything the camera looks at or is placed relative to (the rover's
position, its heading) is passed through a critically-damped 2nd-order filter before
use, and the camera's own pursuit of its computed target pose is filtered again -
double damping, a standard chase-cam technique, so chassis-level bumps and turns never
transmit straight to the frame. The closed-form (Juckett) update is used rather than a
naive `pos += (target - pos) * dt / tau` because it is exact for any dt, including the
large, irregular per-tick dt this node sees while sim time crawls forward slowly.

Never touches `camera_link` (the rover's own RGB/depth pair) or the rover's mass,
inertia or collisions - this node only calls set_pose on its OWN rig model.
"""

import math
import subprocess
import time

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node

# ---------------------------------------------------------------------------
# Pure math: no ROS, no gz - testable/reasoned-about on its own.
# ---------------------------------------------------------------------------


def spring_damp(pos, vel, target, omega, dt):
    """One step of a critically damped spring toward `target`, closed-form (stable for
    any dt - see Ryan Juckett, "Damped Springs", the standard reference for this exact
    update). `pos`/`vel`/`target` are equal-length sequences (used here for 3-vectors).
    omega <= 0 means "no smoothing" (snap directly to target, vel zeroed).
    """
    if omega <= 0.0 or dt <= 0.0:
        return list(target), [0.0] * len(target)
    exp_term = math.exp(-omega * dt)
    new_pos = [0.0] * len(pos)
    new_vel = [0.0] * len(pos)
    for i in range(len(pos)):
        change = pos[i] - target[i]
        temp = (vel[i] + omega * change) * dt
        new_pos[i] = target[i] + (change + temp) * exp_term
        new_vel[i] = (vel[i] - omega * temp) * exp_term
    return new_pos, new_vel


def tau_to_omega(tau_s):
    """`tau_s` is roughly "seconds to settle" - not an exact time constant, but a
    knob a human can tune without knowing spring math. 0 or negative disables
    smoothing (snaps)."""
    return 0.0 if tau_s <= 0.0 else 1.0 / tau_s


def look_at_rpy(eye, target):
    """roll/pitch/yaw for a gz camera at `eye` aimed at `target` (camera looks along
    +x, same convention as scripts/render_still.py's `_look_at_rpy` - kept identical
    on purpose so the two tools' notions of "aim" never quietly diverge)."""
    dx, dy, dz = (target[0] - eye[0], target[1] - eye[1], target[2] - eye[2])
    yaw = math.atan2(dy, dx)
    pitch = -math.atan2(dz, math.hypot(dx, dy))
    return 0.0, pitch, yaw


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp3(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


# ---------------------------------------------------------------------------
# Shot library: each shot function takes the current smoothed rover state and
# elapsed time, and returns a (eye_xyz, aim_xyz) target pair. The node then runs
# that target through its own camera-pursuit spring - see CameraRigNode._tick.
# ---------------------------------------------------------------------------


def shot_chase(rover_pos, heading, t, spawn, params):
    fx, fy = heading
    back, up, lookahead, aim_h = (
        params["chase_standoff_back"],
        params["chase_standoff_up"],
        params["chase_lookahead"],
        params["chase_aim_height"],
    )
    eye = [rover_pos[0] - fx * back, rover_pos[1] - fy * back, rover_pos[2] + up]
    aim = [rover_pos[0] + fx * lookahead, rover_pos[1] + fy * lookahead, rover_pos[2] + aim_h]
    return eye, aim


def shot_orbit(rover_pos, heading, t, spawn, params):
    radius, height, period, aim_h = (
        params["orbit_radius"],
        params["orbit_height"],
        params["orbit_period_s"],
        params["orbit_aim_height"],
    )
    theta = 2.0 * math.pi * (t / max(period, 1e-3))
    eye = [
        rover_pos[0] + radius * math.cos(theta),
        rover_pos[1] + radius * math.sin(theta),
        rover_pos[2] + height,
    ]
    aim = [rover_pos[0], rover_pos[1], rover_pos[2] + aim_h]
    return eye, aim


def shot_crane(rover_pos, heading, t, spawn, params):
    duration = params["crane_duration_s"]
    ease = smoothstep(t / max(duration, 1e-3))
    start_eye = [spawn[0] + params["crane_start_back"], spawn[1] + params["crane_start_side"],
                 spawn[2] + params["crane_start_up"]]
    end_eye = [spawn[0] + params["crane_end_back"], spawn[1] + params["crane_end_side"], spawn[2] + params["crane_end_up"]]
    start_aim = [spawn[0], spawn[1], spawn[2] + params["crane_start_aim_height"]]
    # Reveal target: wide, blends from "close on spawn" to "out along the rover's
    # current smoothed position plus a chunk of horizon beyond it" - keeps the rover
    # loosely in frame while opening the shot onto the landscape past it.
    end_aim = [
        rover_pos[0] + heading[0] * params["crane_end_aim_reach"],
        rover_pos[1] + heading[1] * params["crane_end_aim_reach"],
        spawn[2] + params["crane_end_aim_height"],
    ]
    eye = lerp3(start_eye, end_eye, ease)
    aim = lerp3(start_aim, end_aim, ease)
    return eye, aim


def shot_track(rover_pos, heading, t, spawn, params):
    eye = [
        spawn[0] + params["track_offset_forward"],
        spawn[1] + params["track_offset_side"],
        spawn[2] + params["track_offset_up"],
    ]
    aim = [rover_pos[0], rover_pos[1], rover_pos[2] + params["track_aim_height"]]
    return eye, aim


SHOTS = {
    "chase": shot_chase,
    "orbit": shot_orbit,
    "crane": shot_crane,
    "track": shot_track,
}


class CameraRigNode(Node):
    def __init__(self):
        super().__init__("regolith_camera_rig")
        self.declare_parameter("world_name", "regolith_moon")
        self.declare_parameter("rig_model_name", "cine_rig")
        self.declare_parameter("shot", "chase")
        self.declare_parameter("update_rate_hz", 30.0)

        # Rover-tracking smoothing (dampens chassis bump/roll BEFORE it reaches any
        # shot math) and the camera's own pursuit smoothing (its lag behind the
        # computed target) - two independent taus, the standard chase-cam double-damp.
        self.declare_parameter("track_pos_tau_s", 0.35)
        self.declare_parameter("track_heading_tau_s", 0.6)
        self.declare_parameter("cam_eye_tau_s", 0.5)
        self.declare_parameter("cam_aim_tau_s", 0.4)

        self.declare_parameter("chase_standoff_back", 2.8)
        self.declare_parameter("chase_standoff_up", 1.3)
        self.declare_parameter("chase_lookahead", 0.8)
        self.declare_parameter("chase_aim_height", 0.25)

        # Tightened from an original 9.0/4.0 (paired with the rig's then-shared 1.3 rad
        # hfov) after a first orbit take measured the rover at ~3% of frame width -
        # technically clean footage that failed as a beauty shot. This radius/height,
        # combined with hello_moon.launch.py's narrower 0.7 rad hfov for the orbit
        # shot specifically, was checked with a still render before the reshoot and
        # puts the rover at roughly 16-30% of frame width through the sweep.
        self.declare_parameter("orbit_radius", 4.0)
        self.declare_parameter("orbit_height", 2.0)
        self.declare_parameter("orbit_period_s", 60.0)
        self.declare_parameter("orbit_aim_height", 0.3)

        self.declare_parameter("crane_duration_s", 20.0)
        # Positioned along the sun's bearing (~55 deg, from world.sdf's own <direction> -
        # see docs/media/README.md's "lit, not silhouetted" note) so the crane's reveal
        # looks AT the rover's lit face rather than its shadow side - checked with a
        # still render before shooting, not assumed. crane_end_up capped at 10 m,
        # comfortably under the ~15 m far-field-seam threshold (see the same README
        # section) with margin for local terrain height variation away from spawn.
        self.declare_parameter("crane_start_back", 0.975)
        self.declare_parameter("crane_start_side", 1.393)
        self.declare_parameter("crane_start_up", 0.35)
        self.declare_parameter("crane_start_aim_height", 0.15)
        self.declare_parameter("crane_end_back", 16.06)
        self.declare_parameter("crane_end_side", 22.94)
        self.declare_parameter("crane_end_up", 10.0)
        self.declare_parameter("crane_end_aim_reach", 30.0)
        self.declare_parameter("crane_end_aim_height", 2.0)

        self.declare_parameter("track_offset_forward", 6.0)
        self.declare_parameter("track_offset_side", 5.0)
        self.declare_parameter("track_offset_up", 1.0)
        self.declare_parameter("track_aim_height", 0.25)

        self.declare_parameter("gz_service_timeout_ms", 3000)

        self._shot_name = self.get_parameter("shot").value
        if self._shot_name not in SHOTS:
            raise RuntimeError(
                f"unknown shot '{self._shot_name}' - choose from {sorted(SHOTS)}"
            )
        self._shot_fn = SHOTS[self._shot_name]
        self._params = {
            name: self.get_parameter(name).value
            for name in (
                "chase_standoff_back", "chase_standoff_up", "chase_lookahead", "chase_aim_height",
                "orbit_radius", "orbit_height", "orbit_period_s", "orbit_aim_height",
                "crane_duration_s", "crane_start_back", "crane_start_side", "crane_start_up", "crane_start_aim_height",
                "crane_end_back", "crane_end_side", "crane_end_up", "crane_end_aim_reach",
                "crane_end_aim_height",
                "track_offset_forward", "track_offset_side", "track_offset_up", "track_aim_height",
            )
        }

        self._world = self.get_parameter("world_name").value
        self._model = self.get_parameter("rig_model_name").value
        self._svc_timeout_ms = int(self.get_parameter("gz_service_timeout_ms").value)

        self._rover_pos = None  # last raw /ground_truth/pose position (x, y, z)
        self._rover_yaw = None
        self._spawn = None  # first-seen rover position, frozen - crane/track anchor

        self._track_pos = None
        self._track_vel = [0.0, 0.0, 0.0]
        self._heading_vec = None
        self._heading_vel = [0.0, 0.0]

        self._cam_eye = None
        self._cam_eye_vel = [0.0, 0.0, 0.0]
        self._cam_aim = None
        self._cam_aim_vel = [0.0, 0.0, 0.0]

        self._last_tick_time = None
        self._start_time = None
        self._call_count = 0
        self._fail_count = 0
        self._call_ms_sum = 0.0
        self._wall_start = None

        self.create_subscription(PoseStamped, "/ground_truth/pose", self._on_pose, 20)
        period = 1.0 / max(self.get_parameter("update_rate_hz").value, 1.0)
        self.create_timer(period, self._tick)
        self.get_logger().info(
            f"camera_rig_node: shot='{self._shot_name}' model='{self._model}' "
            f"world='{self._world}' rate={1.0/period:.1f} Hz (sim time)"
        )

    def _on_pose(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self._rover_pos = (p.x, p.y, p.z)
        self._rover_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        if self._spawn is None:
            self._spawn = (p.x, p.y, p.z)

    def _tick(self):
        if self._rover_pos is None:
            return  # no ground truth yet - hold at the rig's spawn pose in world.sdf
        now = self.get_clock().now()
        if self._start_time is None:
            self._start_time = now
        t = (now - self._start_time).nanoseconds * 1e-9
        if self._last_tick_time is None:
            dt = 1e-3
        else:
            dt = (now - self._last_tick_time).nanoseconds * 1e-9
        self._last_tick_time = now
        if dt <= 0.0:
            return

        if self._track_pos is None:
            self._track_pos = list(self._rover_pos)
            fx, fy = math.cos(self._rover_yaw), math.sin(self._rover_yaw)
            self._heading_vec = [fx, fy]

        omega_pos = tau_to_omega(self.get_parameter("track_pos_tau_s").value)
        self._track_pos, self._track_vel = spring_damp(
            self._track_pos, self._track_vel, list(self._rover_pos), omega_pos, dt
        )
        raw_heading = [math.cos(self._rover_yaw), math.sin(self._rover_yaw)]
        omega_hdg = tau_to_omega(self.get_parameter("track_heading_tau_s").value)
        self._heading_vec, self._heading_vel = spring_damp(
            self._heading_vec, self._heading_vel, raw_heading, omega_hdg, dt
        )
        hn = math.hypot(*self._heading_vec) or 1.0
        heading = (self._heading_vec[0] / hn, self._heading_vec[1] / hn)

        target_eye, target_aim = self._shot_fn(
            self._track_pos, heading, t, self._spawn, self._params
        )

        if self._cam_eye is None:
            self._cam_eye, self._cam_aim = list(target_eye), list(target_aim)

        omega_eye = tau_to_omega(self.get_parameter("cam_eye_tau_s").value)
        omega_aim = tau_to_omega(self.get_parameter("cam_aim_tau_s").value)
        self._cam_eye, self._cam_eye_vel = spring_damp(
            self._cam_eye, self._cam_eye_vel, target_eye, omega_eye, dt
        )
        self._cam_aim, self._cam_aim_vel = spring_damp(
            self._cam_aim, self._cam_aim_vel, target_aim, omega_aim, dt
        )

        roll, pitch, yaw = look_at_rpy(self._cam_eye, self._cam_aim)
        self._set_pose(self._cam_eye, roll, pitch, yaw)

    def _set_pose(self, eye, roll, pitch, yaw):
        qx, qy, qz, qw = quat_from_rpy(roll, pitch, yaw)
        req = (
            f'name: "{self._model}" '
            f"position {{ x: {eye[0]:.4f} y: {eye[1]:.4f} z: {eye[2]:.4f} }} "
            f"orientation {{ x: {qx:.6f} y: {qy:.6f} z: {qz:.6f} w: {qw:.6f} }}"
        )
        cmd = [
            "gz", "service", "-s", f"/world/{self._world}/set_pose",
            "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
            "--timeout", str(self._svc_timeout_ms), "--req", req,
        ]
        if self._wall_start is None:
            self._wall_start = time.monotonic()
        self._call_count += 1
        t0 = time.monotonic()
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
            self._call_ms_sum += (time.monotonic() - t0) * 1000.0
            if out.returncode != 0 or "true" not in out.stdout.lower():
                self._fail_count += 1
                if self._fail_count <= 5 or self._fail_count % 50 == 0:
                    self.get_logger().warning(
                        f"set_pose failed ({self._fail_count}/{self._call_count} so far): "
                        f"{out.stderr.strip() or out.stdout.strip()}"
                    )
        except Exception as exc:  # noqa: BLE001 - never crash the rig over one missed frame
            self._fail_count += 1
            self.get_logger().warning(f"set_pose call raised: {exc}")
        if self._call_count % 100 == 0:
            wall_elapsed = time.monotonic() - self._wall_start
            self.get_logger().info(
                f"camera_rig perf: {self._call_count} calls, "
                f"avg {self._call_ms_sum / self._call_count:.1f} ms/call, "
                f"effective {self._call_count / max(wall_elapsed, 1e-3):.2f} Hz wall-clock, "
                f"{self._fail_count} failed"
            )


def main():
    rclpy.init()
    node = CameraRigNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"camera_rig_node exiting: {node._call_count} set_pose calls, "
            f"{node._fail_count} failed"
        )
        node.destroy_node()
        # rclpy's own SIGINT handler (installed by rclpy.init()) can already have
        # called context.shutdown() by the time we get here - ros2 launch sending
        # SIGINT on teardown races this exact path. A second shutdown() raises
        # RCLError ("rcl_shutdown already called"); harmless, but was turning a
        # clean teardown into a logged Traceback and a nonzero exit code on every
        # normal `ros2 launch` stop, which looked like a crash and was not one.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
