#!/usr/bin/env python3
# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Flip and stuck recovery for the Regolith demo.

The primary flip fix is preventive: the terrain collision geometry now uses
tilted "shingle" slabs instead of flat-topped boxes, which collapses the
vertical cliffs at cell boundaries that used to kick the rover into a roll
(see regolith_terrain_gen/heightmap.py and PROGRESS.md M4/M5). This node is
the honest backstop for the rare residual case: a wheeled rover cannot
physically self-right, so instead of leaving a flipped rover to permanently
kill an unattended demo (the previous "detect and halt, restart the demo"
behaviour), it performs an explicitly SIMULATED recovery - it teleports the
model back to its last known-upright pose via gz-sim's set_pose service.

This is deliberately labelled as a simulated reset in every log line: it is
NOT physical self-righting and would not exist on real hardware, where the
preventive terrain smoothing (and, on a real rover, a suitable chassis /
righting mechanism) is what matters. It exists so the PoC demo keeps running.

Detection and the reset pose both come from ground truth (/ground_truth/pose,
the model's world-frame pose that set_pose also operates in) rather than the
EKF, whose two_d_mode estimate can never show roll/pitch. The reset keeps the
rover's (x, y) essentially where it was - stepping back only ~backoff_s of
travel to get off the offending lip - so the localization estimate (which
does not observe the teleport) stays consistent with the physical position to
within its normal noise, and forces the orientation upright.

This node also carries a second, unrelated backstop: a "stuck" detector. It was
built for one failure mode and has since been measured against a different,
more common one - both are covered here, and they want different recoveries:

  * WHEELS LOCKED, UPRIGHT. In a tight skid-steer turn the fixed-axle wheels
    must scrub laterally, and under lunar gravity the low wheel normal force
    gives a narrow friction cone - the dartsim contact solver occasionally
    collapses to an all-static solution where the commanded joint velocity is
    infeasible, and the wheels lock up entirely while the rover stays upright
    (so the flip detector never sees it). Confirmed directly: position and yaw
    both froze while /cmd_vel kept commanding motion, and switching the command
    to a plain straight line broke the lock immediately.
  * WEDGED ON A BOULDER. Once rock <collision> started working at all (it was
    a silent no-op before - the rover drove through all 190 boulders), this
    became the dominant case by far: 64 events across three acceptance runs,
    attributed by experiment to rock collisions rather than terrain roughness.
    A straight-line nudge is exactly the wrong response here - it pushes
    harder into the obstacle - and measurement agrees: those 64 nudges freed
    the rover zero times. See PROGRESS.md.

Detection now has two triggers, and the run itself reports which one fired:

  * GROUND TRUTH - near-zero true motion while a non-trivial /cmd_vel is
    commanded, sustained past a debounce. A simulation oracle, like the flip
    detector, and it needs the rover to be almost completely stationary.
  * WHEEL SLIP (onboard) - wheel_slip_node.py's /wheel_slip, i.e. the wheels
    claiming distance the IMU cannot corroborate. This one uses no privileged
    information and catches the case the oracle misses: a rover creeping and
    scrubbing along a boulder at 1-2 cm/s is above the ground-truth trigger's
    "not moving" threshold while its odometry runs away just the same.

Recovery
is now an escalating escape maneuver - reverse, turn away, mark the obstacle
as a keep-out zone, re-trigger planning - see _recover_stuck. It is not
labelled "simulated" the way the flip reset is: every part of it is something
a real rover's FDIR could do. The node also reports, per event, whether the
maneuver actually moved the rover, so "recovery fired" is never again mistaken
for "recovery worked".
"""

from collections import deque
import math
import subprocess
import time

from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


def _roll_pitch_yaw(q) -> tuple:
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


def hazard_point_xy(estimated_xy: tuple, yaw: float, lead_m: float) -> tuple:
    """Compute where a keep-out marker would land.

    `lead_m` ahead of the estimated pose along its heading - pure geometry,
    no ROS, so the goal-clearance check below can be tested without a node.
    """
    x, y = estimated_xy
    return (x + lead_m * math.cos(yaw), y + lead_m * math.sin(yaw))


def hazard_too_close_to_goal(hazard_xy: tuple, goal_xy: tuple, clearance_m: float) -> bool:
    """Check whether a keep-out zone at hazard_xy would risk covering the goal.

    See _mark_hazard's docstring for why this can happen at all: the hazard
    is marked in the estimator's frame, the goal is not, so large divergence
    can bring a hazard down right on top of it.
    """
    return math.hypot(hazard_xy[0] - goal_xy[0], hazard_xy[1] - goal_xy[1]) < clearance_m


class FlipRecoveryNode(Node):
    def __init__(self):
        super().__init__("regolith_flip_recovery")
        self.declare_parameter("world_name", "regolith_moon")
        self.declare_parameter("model_name", "rover")
        self.declare_parameter("flip_threshold_deg", 60.0)
        self.declare_parameter("safe_threshold_deg", 25.0)
        self.declare_parameter("debounce_s", 1.0)  # sustained flip before acting
        self.declare_parameter(
            "backoff_s", 2.0
        )  # base "how far back (in sim time)" the reset pose is
        self.declare_parameter("max_backoff_s", 40.0)  # cap on progressive backoff
        self.declare_parameter("cooldown_s", 5.0)  # re-arm delay after a reset
        self.declare_parameter(
            "relapse_window_s", 20.0
        )  # re-flip within this of a reset => "stuck", back off further
        self.declare_parameter("clearance_m", 0.3)  # z lift above recorded ground height
        self.declare_parameter("check_period_s", 0.2)

        # Stuck (wheels-locked-but-upright) detection: see module docstring.
        self.declare_parameter(
            "stuck_min_speed_mps", 0.02
        )  # GT speed below this counts as "not moving"
        self.declare_parameter(
            "stuck_min_commanded_mps", 0.03
        )  # /cmd_vel magnitude above this counts as "trying to move"
        self.declare_parameter("stuck_debounce_s", 3.0)  # sustained mismatch before acting
        self.declare_parameter(
            "stuck_nudge_rate_hz", 30.0
        )  # publish rate during the override (must beat pure_pursuit's 10 Hz)
        self.declare_parameter("stuck_cooldown_s", 5.0)  # re-arm delay after a stuck recovery
        # Escape maneuver, escalating over consecutive events - see _recover_stuck.
        self.declare_parameter("escape_reverse_speed_mps", 0.2)
        self.declare_parameter("escape_reverse_s", 3.0)  # base; doubles per consecutive event
        self.declare_parameter("escape_turn_rate_rps", 0.5)
        self.declare_parameter("escape_turn_s", 2.0)  # base; +1.5 s per consecutive event
        self.declare_parameter("escape_max_reverse_s", 10.0)
        self.declare_parameter("escape_max_turn_s", 8.0)
        self.declare_parameter("escape_freed_threshold_m", 0.3)  # GT motion that counts as "freed"
        self.declare_parameter(
            "escape_check_delay_s", 1.0
        )  # sim time to wait before judging the result
        self.declare_parameter("stuck_relapse_window_s", 120.0)  # re-stick within this => escalate
        # Keep-out marking: where the obstacle is relative to the rover when it
        # wedges (it is in front - that is the direction it was pushing).
        self.declare_parameter("hazard_lead_m", 0.8)
        self.declare_parameter("publish_hazards", True)
        # Keep-out zones are marked in the ESTIMATOR's frame (see _mark_hazard),
        # while the mission goal is a fixed world-frame point that never moves
        # with the estimate. If divergence grows large enough that the two
        # coincide, a hazard can land on the goal's own cell and wall it off
        # permanently - a real, observed failure (PROGRESS.md, seed 55: 17.94 m
        # divergence, `planner_node` declaring the actual goal cell lethal
        # forever). costmap_node's own `hazard_radius_m` defaults to 1.2 m; this
        # clearance needs to exceed that by more than a rounding margin, so it
        # is set independently rather than duplicated from the other package.
        self.declare_parameter("hazard_goal_clearance_m", 1.5)
        # Second trigger, from wheel_slip_node's onboard detector. The
        # ground-truth trigger above needs the rover to be nearly stationary
        # (< stuck_min_speed_mps); a rover scrubbing slowly against a boulder
        # slips past it while its odometry runs away regardless. The onboard
        # detector keys on the disagreement itself - wheels claiming distance
        # the IMU cannot corroborate - so it catches the creeping case too.
        self.declare_parameter("slip_trigger_s", 5.0)
        self.declare_parameter("trigger_on_slip", True)
        # Diagnostic only, OFF by default (spams a line up to check_period_s's
        # rate while a stuck-candidate streak is live). Added to look INSIDE
        # _stuck_since/_stuck_cooldown_until - see PROGRESS.md, "what's left is
        # ... a piece of the node's own internal state this harness cannot
        # observe at all without adding debug logging". External 10 Hz logging
        # of GT speed and /cmd_vel already ruled out both being sustained-false
        # at the fixed arm's second chokepoint; this looks at whether _stuck_since
        # resets mid-streak at this node's own 5 Hz tick phase, which a 10 Hz
        # external log phase-aligned to nothing this node controls could miss.
        self.declare_parameter("stuck_debug", False)

        # ~180 s of upright trail at check_period_s, so progressive backoff can
        # step back to a genuinely different location rather than the same lip.
        self._history = deque(maxlen=900)  # (t, x, y, z, yaw) upright poses
        self._pose = None
        self._flip_since = None
        self._cooldown_until = None
        self._resets = 0
        self._consecutive = 0  # rapid re-flips near the same spot
        self._last_reset_t = None

        self._last_cmd = Twist()
        self._prev_tick_pose = None  # (t, x, y) from the previous _tick, for a GT speed estimate
        self._stuck_since = None
        self._stuck_cooldown_until = None
        self._stuck_resets = 0
        self._stuck_consecutive = 0  # escalation level: events inside relapse_window of each other
        self._last_stuck_t = None
        self._escapes_freed = 0  # escape maneuvers that produced real motion
        self._pending_escape_check = None  # (x, y, level) sampled before the maneuver
        self._slip_since = None  # onboard wheel-slip signal asserted since
        # Measured sim-seconds per wall-second, tracked in _tick. Historically
        # this converted escape-maneuver durations (specified in SIM time -
        # 0.2 m/s for 3 s means 0.6 m of ground covered) into wall-clock
        # sleeps, because the escape used to run as a blocking loop and the
        # ROS clock cannot advance while this node blocks its own executor.
        # That conversion is GONE (see _escape_tick / PROGRESS.md, "the
        # natural fix"): the escape now runs as a non-blocking state machine
        # driven by the real sim clock, so maneuver durations are exact
        # regardless of RTF. self._rtf is kept only as a diagnostic logged
        # alongside each recovery, for comparison against the pre-fix history
        # in PROGRESS.md - nothing reads it to compute a duration any more.
        self._rtf = 1.0
        self._rtf_sample = None  # (wall_t, sim_t) from the previous tick
        self._escape = None  # in-progress escape maneuver state - see _start_escape
        self._trigger_counts = {"ground truth": 0, "wheel slip (onboard)": 0}
        self._trigger = None  # which detector fired the current recovery
        self._estimated_pose = (
            None  # /odometry/filtered, for marking hazards in the planner's frame
        )
        self._last_goal = None

        self.create_subscription(PoseStamped, "/ground_truth/pose", self._on_pose, 10)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self.create_subscription(Odometry, "/odometry/filtered", self._on_odometry, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self.create_subscription(Bool, "/wheel_slip", self._on_slip, 10)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._hazard_pub = self.create_publisher(PointStamped, "/hazard/stuck_point", 10)
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        # Publishing an override faster than pure_pursuit is not the same as
        # having control: at 30 Hz against its 10 Hz, roughly a quarter of the
        # commands gz-sim acts on during a maneuver are still its forward
        # commands, which is precisely what the maneuver is trying to undo. So
        # the follower is muted outright for the duration - see pure_pursuit_node.
        self._recovery_pub = self.create_publisher(Bool, "/recovery_active", 10)
        period = self.get_parameter("check_period_s").value
        self.create_timer(period, self._tick)
        # Drives the in-progress escape maneuver, when there is one - see
        # _escape_tick. Runs unconditionally at stuck_nudge_rate_hz for the
        # node's whole life (cheap no-op while self._escape is None) rather
        # than being created/destroyed per maneuver, so there is no dynamic
        # timer lifecycle to get wrong. Must be faster than pure_pursuit_node's
        # 10 Hz control loop - see _escape_tick's docstring.
        nudge_rate_hz = self.get_parameter("stuck_nudge_rate_hz").value
        self.create_timer(1.0 / nudge_rate_hz, self._escape_tick)
        self.get_logger().info(
            "Flip/stuck recovery armed (simulated set_pose backstop for flips, "
            "straight-line cmd_vel override for stuck-but-upright)"
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_pose(self, msg: PoseStamped) -> None:
        self._pose = msg.pose

    def _on_cmd(self, msg: Twist) -> None:
        self._last_cmd = msg

    def _on_odometry(self, msg: Odometry) -> None:
        self._estimated_pose = msg.pose.pose

    def _on_goal(self, msg: PoseStamped) -> None:
        self._last_goal = msg

    def _on_slip(self, msg: Bool) -> None:
        self._slip_since = self._now_s() if msg.data else None

    def _tick(self) -> None:
        self._update_rtf()
        if self._escape is not None:
            # An escape maneuver is in progress, driven by _escape_tick on its
            # own timer. Skip flip/stuck detection and history recording for
            # the duration, same as the old blocking _hold() did by freezing
            # the whole executor - deliberately preserved here rather than
            # letting the flip/stuck detectors run concurrently with a
            # maneuver they didn't expect to overlap with.
            return
        if self._pose is None:
            return
        roll, pitch, yaw = _roll_pitch_yaw(self._pose.orientation)
        t = self._now_s()
        safe = math.radians(self.get_parameter("safe_threshold_deg").value)
        flip = math.radians(self.get_parameter("flip_threshold_deg").value)
        p = self._pose.position

        if abs(roll) < safe and abs(pitch) < safe:
            # Upright: record as a candidate recovery pose.
            self._history.append((t, p.x, p.y, p.z, yaw))
            self._flip_since = None
            self._check_stuck(t, p)
            return

        self._prev_tick_pose = (t, p.x, p.y)
        self._stuck_since = None  # a flip supersedes any in-progress stuck timer

        if abs(roll) > flip or abs(pitch) > flip:
            if self._cooldown_until is not None and t < self._cooldown_until:
                return
            if self._flip_since is None:
                self._flip_since = t
                return
            if (t - self._flip_since) < self.get_parameter("debounce_s").value:
                return
            self._recover(roll, pitch)

    def _check_stuck(self, t: float, p) -> None:
        if self._pending_escape_check is not None:
            self._check_escape_result(p, t)
        prev = self._prev_tick_pose
        self._prev_tick_pose = (t, p.x, p.y)
        if prev is None:
            return
        dt = t - prev[0]
        if dt <= 0.0:
            return
        gt_speed = math.hypot(p.x - prev[1], p.y - prev[2]) / dt

        cmd = self._last_cmd
        half_track = 0.23  # regolith_rover's wheel_separation / 2 - see regolith_rover.urdf.xacro
        commanded_speed = abs(cmd.linear.x) + abs(cmd.angular.z) * half_track
        min_commanded = self.get_parameter("stuck_min_commanded_mps").value
        min_speed = self.get_parameter("stuck_min_speed_mps").value

        if self._slip_triggered(t):
            self._fire_recovery(t, "wheel slip (onboard)")
            return

        debug = self.get_parameter("stuck_debug").value

        if commanded_speed < min_commanded or gt_speed >= min_speed:
            if debug and self._stuck_since is not None:
                # A streak was live and just got cut - the case PROGRESS.md
                # flagged as invisible to a 10 Hz external log: this node
                # samples gt_speed/commanded_speed at its own 5 Hz tick phase,
                # so a reset can land between two external samples that both
                # look continuous.
                reason = "commanded<min" if commanded_speed < min_commanded else "gt_speed>=min"
                self.get_logger().info(
                    f"[stuck_debug] streak RESET after {t - self._stuck_since:.2f}s "
                    f"({reason}: gt_speed={gt_speed:.4f} commanded={commanded_speed:.4f} "
                    f"dt={dt:.3f} t={t:.2f})"
                )
            self._stuck_since = None
            return

        if self._stuck_cooldown_until is not None and t < self._stuck_cooldown_until:
            return
        if self._stuck_since is None:
            if debug:
                self.get_logger().info(
                    f"[stuck_debug] streak START (gt_speed={gt_speed:.4f} "
                    f"commanded={commanded_speed:.4f} dt={dt:.3f} t={t:.2f})"
                )
            self._stuck_since = t
            return
        if (t - self._stuck_since) < self.get_parameter("stuck_debounce_s").value:
            return

        if debug:
            self.get_logger().info(
                f"[stuck_debug] streak FIRED after {t - self._stuck_since:.2f}s (t={t:.2f})"
            )

        self._fire_recovery(t, "ground truth")

    def _slip_triggered(self, t: float) -> bool:
        if not self.get_parameter("trigger_on_slip").value or self._slip_since is None:
            return False
        if self._stuck_cooldown_until is not None and t < self._stuck_cooldown_until:
            return False
        return (t - self._slip_since) >= self.get_parameter("slip_trigger_s").value

    def _fire_recovery(self, t: float, trigger: str) -> None:
        # Escalate only for events that are part of the same fight with the same
        # obstacle; an event long after the last one starts from level 0 again.
        window = self.get_parameter("stuck_relapse_window_s").value
        if self._last_stuck_t is not None and (t - self._last_stuck_t) < window:
            self._stuck_consecutive += 1
        else:
            self._stuck_consecutive = 0
        self._last_stuck_t = t
        self._trigger = trigger
        self._trigger_counts[trigger] += 1
        self._slip_since = None
        self._recover_stuck()

    def _update_rtf(self) -> None:
        """Tracks how fast simulated time runs against the wall clock.

        Sampled in the timer callback, where both clocks are advancing normally;
        an exponential average smooths the jitter. No longer used to convert
        any duration (see _recover_stuck / _escape_tick for why) - kept purely
        as a diagnostic, logged alongside each recovery for comparison against
        the pre-fix history in PROGRESS.md.
        """
        wall = time.monotonic()
        sim = self._now_s()
        previous = self._rtf_sample
        self._rtf_sample = (wall, sim)
        if previous is None:
            return
        wall_dt = wall - previous[0]
        sim_dt = sim - previous[1]
        if wall_dt <= 0.0 or sim_dt <= 0.0:
            return
        ratio = sim_dt / wall_dt
        if not (0.02 <= ratio <= 5.0):
            return  # a paused or stepping sim; keep the last sane estimate
        self._rtf = 0.9 * self._rtf + 0.1 * ratio

    def _recover_stuck(self) -> None:
        """Kick off an escalating escape maneuver: reverse, turn away, mark the spot, replan.

        The previous recovery was a 1.0 s straight-line forward nudge. Measured
        over the res40 acceptance runs it fired 64 times and freed the rover
        zero times (PROGRESS.md): pushing forward harder into the boulder the
        rover is already wedged against does nothing, and after the nudge
        pure_pursuit steered straight back onto the same path into the same
        rock, so the same event repeated on a ~31 s metronome for the rest of
        the run.

        What replaces it does three different things, in the order a real
        rover's FDIR would:

          1. REVERSE - back out along the way it came in, which is by
             construction obstacle-free; forward is where the obstacle is.
          2. TURN in place, alternating direction per attempt, so the rover
             leaves on a different heading instead of re-approaching.
          3. MARK the obstacle as a keep-out zone and re-trigger planning, so
             the new path routes around it rather than back into it. Without
             this the first two only buy one more approach.

        Each consecutive event (inside stuck_relapse_window_s) escalates the
        reverse and turn durations, because a wedge that survives one attempt
        needs a bigger disengagement, not the same one again.

        This method only sets up self._escape and returns - it does not block.
        The maneuver itself is driven by _escape_tick, on its own timer, one
        state (reverse, then turn) at a time, checked against the real sim
        clock. See PROGRESS.md ("the natural fix") for why: the previous
        version blocked this node's executor for the maneuver's duration and
        converted the sim-time duration into a wall-clock sleep using an
        exponentially-smoothed RTF estimate sampled before the block started,
        because the ROS clock cannot advance while an executor is blocked.
        That estimate is a poor proxy for the live RTF during the blocked
        window itself, and measurement (PROGRESS.md, seeds 7/55/123, 35+ runs)
        traced a real, reproducible run-to-run bifurcation in escape outcomes
        to timing sensitivity introduced by exactly this mechanism - among
        other candidates not ruled out. Driving the maneuver off the real sim
        clock instead removes RTF-estimation error as a contributor entirely;
        it does not by itself prove RTF-estimation error was the dominant
        contributor (the rigorous decision-point check in PROGRESS.md found a
        single RTF sample does not cleanly predict which attractor a run falls
        into), so the paired-campaign validation this change should get before
        being trusted is the next step, not optional.
        """
        level = self._stuck_consecutive
        reverse_s = min(
            self.get_parameter("escape_reverse_s").value * (2**level),
            self.get_parameter("escape_max_reverse_s").value,
        )
        turn_s = min(
            self.get_parameter("escape_turn_s").value + 1.5 * level,
            self.get_parameter("escape_max_turn_s").value,
        )
        reverse_speed = self.get_parameter("escape_reverse_speed_mps").value
        turn_rate = self.get_parameter("escape_turn_rate_rps").value
        turn_sign = 1.0 if level % 2 == 0 else -1.0

        start_xy = (self._pose.position.x, self._pose.position.y) if self._pose else None
        self._mark_hazard()
        self._set_recovery_active(True)

        t0 = self._now_s()
        self._escape = {
            "state": "reverse",
            "deadline": t0 + reverse_s,
            "level": level,
            "start_xy": start_xy,
            "reverse_speed": reverse_speed,
            "reverse_s": reverse_s,
            "turn_rate": turn_rate,
            "turn_s": turn_s,
            "turn_sign": turn_sign,
        }

    def _escape_tick(self) -> None:
        """Advance the in-progress escape maneuver by one publish, if there is one.

        Runs at stuck_nudge_rate_hz (default 30 Hz) unconditionally - see
        __init__. Has to publish faster than pure_pursuit_node's 10 Hz control
        loop for the whole maneuver, or gz-sim just sees whatever pure_pursuit
        published last - muting pure_pursuit via /recovery_active is not the
        same as having control of /cmd_vel (see the module docstring, "the
        follower is muted outright for the duration"), which is why this still
        out-publishes it rather than relying on the mute alone. Each state's
        deadline is a real sim timestamp (self._now_s()), not a wall-clock
        one, so the maneuver's actual sim-time duration is exact regardless of
        how fast or slow the sim is currently running relative to wall clock.
        """
        e = self._escape
        if e is None:
            return
        t = self._now_s()
        cmd = Twist()
        if e["state"] == "reverse":
            cmd.linear.x = -e["reverse_speed"]
            self._cmd_pub.publish(cmd)
            if t >= e["deadline"]:
                e["state"] = "turn"
                e["deadline"] = t + e["turn_s"]
            return
        if e["state"] == "turn":
            cmd.angular.z = e["turn_sign"] * e["turn_rate"]
            self._cmd_pub.publish(cmd)
            if t >= e["deadline"]:
                self._finish_escape(e, t)
            return

    def _finish_escape(self, e: dict, t_now: float) -> None:
        """Stop, log, replan, and re-arm detection once an escape maneuver ends."""
        self._stop()
        self._set_recovery_active(False)
        self._stuck_resets += 1
        self.get_logger().warn(
            f"STUCK RECOVERY #{self._stuck_resets} (escalation level {e['level']}, triggered by "
            f"{self._trigger}; {self._trigger_counts['ground truth']} ground-truth / "
            f"{self._trigger_counts['wheel slip (onboard)']} onboard triggers so far). "
            f"Escape maneuver: reversed "
            f"{e['reverse_speed']:.2f} m/s for {e['reverse_s']:.1f}s, then turned "
            f"{'left' if e['turn_sign'] > 0 else 'right'} at {e['turn_rate']:.2f} rad/s for "
            f"{e['turn_s']:.1f}s (sim time, exact - driven by the sim clock directly, not an "
            f"RTF-estimated wall-clock sleep; RTF over this window measured {self._rtf:.2f}x "
            "real time, logged for comparison only). "
            "A real rover's FDIR could do the same - this is not a simulated teleport."
        )

        self._replan_after_escape()
        self._stuck_since = None
        self._prev_tick_pose = None  # the maneuver moved the rover; don't measure speed across it
        self._stuck_cooldown_until = t_now + self.get_parameter("stuck_cooldown_s").value
        start_xy = e["start_xy"]
        if start_xy is not None:
            # Did the maneuver actually move the rover? This is the number the
            # pre-escape-maneuver recovery never reported about itself.
            #
            # Deferred by escape_check_delay_s rather than checked on the next
            # tick, as a settling margin - now more of a belt-and-braces margin
            # than a hard requirement: the old blocking version could queue up
            # /ground_truth/pose messages for the whole maneuver and read a
            # stale pre-maneuver pose on an immediate check (that happened: the
            # oracle run logged "STILL WEDGED, moved 0.00 m" for maneuvers the
            # acceptance harness's independent trace shows moving the rover
            # ~1.9 m). The non-blocking maneuver never stops the executor, so
            # pose callbacks are processed throughout - but the delay is kept
            # rather than removed, since it costs little and this is exactly
            # the kind of number this project must not get wrong either way.
            self._pending_escape_check = (start_xy[0], start_xy[1], e["level"], None)
        self._escape = None

    def _mark_hazard(self) -> None:
        """Publish the wedge point so the costmap can make it a keep-out zone.

        Marked in the ESTIMATOR's frame (/odometry/filtered), not ground truth:
        the planner routes in that frame, so a hazard marked there stays put
        relative to the path being planned even if the estimate has drifted.
        Marking the true world position instead would put the keep-out zone
        somewhere the planner's own path never goes.

        That choice has a failure mode of its own: the goal is a fixed
        world-frame point that does not drift with the estimate, so if
        divergence grows large enough, the estimator's frame can bring a
        hazard down right on top of the goal's own cell - walling it off
        forever, not because anything is actually there, but because the
        rover's belief of "here" and the goal's real location have converged
        by coincidence of drift. Observed directly (PROGRESS.md, seed 55):
        17.94 m of divergence, and `planner_node` refusing the real goal as
        lethal for the rest of the run. Skipping the mark when it would land
        this close to the active goal is a strictly local fix for that one
        collision - it does nothing about the divergence itself, and a hazard
        skipped this way is a real obstacle left unmarked, traded deliberately
        against permanently blocking the one cell the whole mission is
        driving toward.
        """
        if not self.get_parameter("publish_hazards").value or self._estimated_pose is None:
            return
        lead = self.get_parameter("hazard_lead_m").value
        _, _, yaw = _roll_pitch_yaw(self._estimated_pose.orientation)
        estimated_xy = (self._estimated_pose.position.x, self._estimated_pose.position.y)
        hazard_xy = hazard_point_xy(estimated_xy, yaw, lead)

        if self._last_goal is not None:
            goal_xy = (self._last_goal.pose.position.x, self._last_goal.pose.position.y)
            clearance = self.get_parameter("hazard_goal_clearance_m").value
            if hazard_too_close_to_goal(hazard_xy, goal_xy, clearance):
                self.get_logger().warn(
                    f"Skipping hazard mark at ({hazard_xy[0]:.2f}, {hazard_xy[1]:.2f}) - "
                    f"within {clearance:.1f} m of the active goal "
                    f"({goal_xy[0]:.2f}, {goal_xy[1]:.2f}). Marking it would risk walling "
                    "off the goal itself, most likely from EKF divergence rather than a "
                    "real obstacle there - see _mark_hazard's docstring."
                )
                return

        point = PointStamped()
        point.header.stamp = self.get_clock().now().to_msg()
        point.header.frame_id = "odom"
        point.point.x = hazard_xy[0]
        point.point.y = hazard_xy[1]
        self._hazard_pub.publish(point)

    def _replan_after_escape(self) -> None:
        """Re-sends the active goal so the planner re-plans against the costmap that now contains the keep-out zone. Without this the rover keeps following the old path - which still runs through the obstacle."""
        if self._last_goal is None:
            return
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self._last_goal.header.frame_id or "odom"
        goal.pose = self._last_goal.pose
        self._goal_pub.publish(goal)

    def _check_escape_result(self, p, t: float) -> None:
        """Report whether the escape maneuver actually moved the rover.

        Only once enough sim time has passed for post-maneuver poses to have
        been received - see the note where _pending_escape_check is set.
        """
        x0, y0, level, finished_t = self._pending_escape_check
        if finished_t is None:
            # First tick after the maneuver: the clock has caught up, so start
            # the delay here and judge on a later tick, by which time the queued
            # /ground_truth/pose messages will have been processed.
            self._pending_escape_check = (x0, y0, level, t)
            return
        if (t - finished_t) < self.get_parameter("escape_check_delay_s").value:
            return
        self._pending_escape_check = None
        moved = math.hypot(p.x - x0, p.y - y0)
        freed = moved >= self.get_parameter("escape_freed_threshold_m").value
        if freed:
            self._escapes_freed += 1
        self.get_logger().warn(
            f"STUCK RECOVERY #{self._stuck_resets} result: ground truth moved {moved:.2f} m "
            f"during the maneuver - {'FREED' if freed else 'STILL WEDGED'} "
            f"({self._escapes_freed}/{self._stuck_resets} escapes have freed the rover so far)"
        )

    def _recover(self, roll: float, pitch: float) -> None:
        if not self._history:
            self.get_logger().warn(
                "Rover flipped but no upright pose was ever recorded - cannot reset"
            )
            self._cooldown_until = self._now_s() + self.get_parameter("cooldown_s").value
            return
        t_now = self._now_s()
        # If we re-flip shortly after a reset, the last reset dropped the rover
        # right back onto whatever flipped it - back off PROGRESSIVELY further
        # down the recorded trail instead of looping forever on the same lip
        # (the previous behaviour: fixed 2 s backoff + clearing history, which
        # produced 20+ teleports at one spot - see PROGRESS.md).
        relapse_window = self.get_parameter("relapse_window_s").value
        if self._last_reset_t is not None and (t_now - self._last_reset_t) < relapse_window:
            self._consecutive += 1
        else:
            self._consecutive = 0

        base_backoff = self.get_parameter("backoff_s").value
        max_backoff = self.get_parameter("max_backoff_s").value
        backoff = min(base_backoff * (2**self._consecutive), max_backoff)

        # Newest upright pose that is at least `backoff` seconds old (steps the
        # rover back along the path it actually drove); fall back to the oldest.
        # History is NOT cleared, so successive relapses reach ever-earlier,
        # genuinely-different poses rather than re-selecting near the flip.
        target = self._history[0]
        for entry in reversed(self._history):
            if t_now - entry[0] >= backoff:
                target = entry
                break
        _, x, y, z, yaw = target
        z += self.get_parameter("clearance_m").value

        self._set_recovery_active(True)
        self._stop()  # zero cmd_vel so it doesn't drive off mid-teleport
        try:
            ok = self._set_pose(x, y, z, yaw)
        finally:
            self._set_recovery_active(False)
        self._resets += 1
        stuck = " (stuck: backing off further along the trail)" if self._consecutive else ""
        self.get_logger().warn(
            f"SIMULATED RECOVERY #{self._resets}{stuck}: rover flipped "
            f"(roll {math.degrees(roll):.0f} deg, pitch {math.degrees(pitch):.0f} deg). "
            f"Teleported upright to ({x:.2f}, {y:.2f}, {z:.2f}) yaw {math.degrees(yaw):.0f} deg "
            f"(backoff {backoff:.0f} s) via gz set_pose [{'ok' if ok else 'FAILED'}]. "
            "This is a simulated self-right, not physical - see flip_recovery_node.py."
        )
        self._flip_since = None
        self._last_reset_t = t_now
        self._cooldown_until = t_now + self.get_parameter("cooldown_s").value
        # Drop trail entries newer than the target (they lead into the flip);
        # keep the earlier trail so a further relapse can back off more.
        while self._history and self._history[-1][0] > target[0]:
            self._history.pop()

    def _set_recovery_active(self, active: bool) -> None:
        """Mutes/unmutes pure_pursuit_node for the duration of a maneuver."""
        for _ in range(3):  # a dropped mute would hand control straight back
            self._recovery_pub.publish(Bool(data=active))
            time.sleep(0.02)

    def _stop(self) -> None:
        self._cmd_pub.publish(Twist())

    def _set_pose(self, x: float, y: float, z: float, yaw: float) -> bool:
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        world = self.get_parameter("world_name").value
        model = self.get_parameter("model_name").value
        req = (
            f'name: "{model}" '
            f"position {{ x: {x:.4f} y: {y:.4f} z: {z:.4f} }} "
            f"orientation {{ x: 0 y: 0 z: {qz:.6f} w: {qw:.6f} }}"
        )
        cmd = [
            "gz",
            "service",
            "-s",
            f"/world/{world}/set_pose",
            "--reqtype",
            "gz.msgs.Pose",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            req,
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
        except Exception as exc:  # noqa: BLE001 - log and report failure, never crash the node
            self.get_logger().error(f"set_pose call raised: {exc}")
            return False
        if out.returncode != 0:
            self.get_logger().error(f"set_pose failed: {out.stderr.strip() or out.stdout.strip()}")
            return False
        return "true" in out.stdout.lower()


def main() -> None:
    rclpy.init()
    node = FlipRecoveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
