# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for flip_recovery_node.escape_verdict - the onboard judgement of
whether an escape maneuver freed the rover.

The previous onboard verdict read EKF displacement, which wheel_slip_node's
ZUPT had zeroed for the whole maneuver: 14 of 23 freed escapes were judged
STILL WEDGED (PROGRESS.md, "The escape verdict, fixed"). The replacement reads
the wheels' raw reverse distance, corroborated by the gyro over the turn,
because wheels spinning against a wedge also produce odometry.
"""

import importlib.util
from pathlib import Path
import sys

FLIP_RECOVERY_NODE = Path(__file__).resolve().parent.parent / "scripts/flip_recovery_node.py"


def _load():
    spec = importlib.util.spec_from_file_location("flip_recovery_node_ev", FLIP_RECOVERY_NODE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["flip_recovery_node_ev"] = module
    spec.loader.exec_module(module)
    return module


ev = _load().escape_verdict
ARGS = dict(freed_threshold_m=0.3, min_turn_ratio=0.145, min_commanded_turn_rad=0.3)


def test_recorded_free_escape_is_freed():
    # A typical recorded free escape: 0.56 m of wheel odometry on the reverse,
    # gyro saw 0.49 of 0.98 rad commanded.
    freed, _ = ev(0.56, 0.49, 0.98, **ARGS)
    assert freed


def test_left_and_right_turns_both_count():
    assert ev(0.56, -0.40, -0.98, **ARGS)[0]


def test_short_reverse_is_wedged():
    freed, reason = ev(0.05, 0.49, 0.98, **ARGS)
    assert not freed and "reversed only" in reason


def test_spinning_wheels_with_a_pinned_chassis_is_wedged():
    # The wheels claim the full reverse, but the chassis cannot turn: the gyro
    # sees 5% of the commanded rotation. Odometry alone would call this freed.
    freed, reason = ev(0.6, 0.05, 1.0, **ARGS)
    assert not freed and "pinned" in reason


def test_a_turn_the_wrong_way_is_not_a_turn():
    assert not ev(0.6, -0.5, 1.0, **ARGS)[0]


def test_threshold_boundary_matches_slip_detector():
    assert ev(0.6, 0.145, 1.0, **ARGS)[0]
    assert not ev(0.6, 0.14, 1.0, **ARGS)[0]


def test_no_turn_to_judge_falls_back_to_odometry():
    assert ev(0.6, 0.0, 0.1, **ARGS)[0]
