# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for flip_recovery_node.py's stuck_debug shadow streak.

The shadow streak exists because the live `_stuck_since` timer cannot answer
the question PROGRESS.md is stuck on. Three things stop the live timer short
of a verdict: the wheel-slip trigger preempts `_check_stuck` and returns
before any `_stuck_since` bookkeeping runs, the post-event cooldown skips it,
and an escape maneuver stops `_check_stuck` being called at all. So a rep can
end with "the oracle never fired here" meaning either "the condition was never
close" or "the condition was satisfied and something else got there first" -
which is exactly the distinction the fixed arm's short/long split turns on
(seed7_fixed_rep15).

The shadow tracks the raw ground-truth CONDITION through all three, and is
diagnostic only. These tests pin both halves of that: it counts what it should
count, and it writes nothing any decision reads.
"""

import importlib.util
from pathlib import Path
import sys
import types

FLIP_RECOVERY_NODE = Path(__file__).resolve().parent.parent / "scripts/flip_recovery_node.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("flip_recovery_node", FLIP_RECOVERY_NODE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["flip_recovery_node"] = module
    spec.loader.exec_module(module)
    return module


flip_recovery_node = _load_module()
_track = flip_recovery_node.FlipRecoveryNode._track_shadow_streak

MIN_SPEED = 0.02  # stuck_min_speed_mps
MIN_COMMANDED = 0.03  # stuck_min_commanded_mps


class _Tracker:
    """A stand-in self: the shadow tracker touches only attributes, params, logs.

    Deliberately not an rclpy Node - these tests must run without a ROS graph,
    like every other unit test in this package.
    """

    def __init__(self, check_period_s: float = 0.2, debounce_s: float = 3.0):
        self._shadow_stuck_since = None
        self._shadow_last_t = None
        self._shadow_resets = 0
        self._shadow_longest_s = 0.0
        self._shadow_announced = False
        # The live timer, untouched by anything under test.
        self._stuck_since = None
        self.lines = []
        self._params = {"check_period_s": check_period_s, "stuck_debounce_s": debounce_s}

    def get_parameter(self, name):
        return types.SimpleNamespace(value=self._params[name])

    def get_logger(self):
        return types.SimpleNamespace(info=self.lines.append)

    def feed(self, t: float, gt_speed: float, commanded_speed: float = 0.15):
        _track(self, t, gt_speed, commanded_speed, MIN_SPEED, MIN_COMMANDED)


def _run(samples, tracker=None, dt=0.2, t0=100.0):
    """Feed (gt_speed) samples at a fixed cadence and return the tracker."""
    tracker = tracker or _Tracker()
    for i, gt in enumerate(samples):
        tracker.feed(t0 + i * dt, gt)
    return tracker


def test_sustained_stall_starts_one_streak_and_passes_the_bar():
    # 4 s of true stall at 5 Hz, with cmd_vel asking for motion throughout.
    tracker = _run([0.005] * 20)
    assert tracker._shadow_stuck_since == 100.0
    assert tracker._shadow_resets == 0
    assert tracker._shadow_longest_s >= 3.0
    assert sum("shadow streak START" in line for line in tracker.lines) == 1
    # Announced once when it clears the bar, not once per tick past it.
    assert sum("PASSED" in line for line in tracker.lines) == 1


def test_noise_on_the_threshold_cuts_the_streak_repeatedly():
    """The noise-floor reading from rep15, in miniature.

    A true speed oscillating across stuck_min_speed_mps never accumulates the
    debounce, because any single sample at or above the threshold resets it -
    the behaviour that makes the detector a coin flip at the chokepoint.
    """
    # Two stalled samples, one over-threshold blip, repeated.
    tracker = _run([0.005, 0.010, 0.035] * 8)
    assert tracker._shadow_resets == 8
    assert tracker._shadow_longest_s < 3.0  # never reaches the bar
    assert not any("PASSED" in line for line in tracker.lines)
    assert all("gt_speed>=min" in line for line in tracker.lines if "CUT" in line)


def test_a_commanded_speed_drop_cuts_the_streak_and_says_so():
    tracker = _Tracker()
    tracker.feed(100.0, 0.005)
    tracker.feed(100.2, 0.005)
    tracker.feed(100.4, 0.005, commanded_speed=0.0)  # pure_pursuit idling
    assert tracker._shadow_resets == 1
    assert any("CUT" in line and "commanded<min" in line for line in tracker.lines)


def test_a_tick_gap_drops_the_streak_rather_than_spanning_it():
    """An escape maneuver (or a flip) stops the ticks; continuity is not implied."""
    tracker = _Tracker()
    tracker.feed(100.0, 0.005)
    tracker.feed(100.2, 0.005)
    tracker.feed(112.0, 0.005)  # ~12 s later: an escape ran in between
    assert any("DROPPED" in line for line in tracker.lines)
    # Dropped, not counted as a cut: nothing about the condition changed.
    assert tracker._shadow_resets == 0
    # A fresh streak starts at the resuming tick, not at the pre-gap one.
    assert tracker._shadow_stuck_since == 112.0


def test_gap_tolerance_allows_ordinary_tick_jitter():
    tracker = _Tracker(check_period_s=0.2)
    tracker.feed(100.0, 0.005)
    tracker.feed(100.5, 0.005)  # 2.5 ticks late, inside the 3x gap bar
    assert not any("DROPPED" in line for line in tracker.lines)
    assert tracker._shadow_stuck_since == 100.0


def test_the_shadow_never_touches_the_live_stuck_timer():
    """The whole point: this is measurement, not a second detector."""
    tracker = _run([0.005] * 40)
    assert tracker._shadow_longest_s >= 3.0  # the shadow would have fired
    assert tracker._stuck_since is None  # the live path is untouched
