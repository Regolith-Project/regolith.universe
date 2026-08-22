# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for flip_recovery_node.py's hazard-vs-goal clearance check.

Keep-out zones are marked in the ESTIMATOR's frame (see _mark_hazard's
docstring), deliberately, so a hazard stays put relative to the path the
planner is actually following even while the estimate drifts. The mission
goal is a fixed world-frame point that does NOT drift with the estimate, so
the two frames can coincide: with enough divergence, a hazard marked "in
front of where I think I am" can land on the goal's own cell and wall it off
forever - not because anything is really there, but because the rover's
drifted belief of "here" happened to converge on the goal.

Observed directly in PROGRESS.md (seed 55, legacy rep 1): 17.94 m of
divergence, and `planner_node` refusing the real, reachable goal as "lethal"
for the entire rest of the run. `hazard_too_close_to_goal` is the guard added
to stop that specific collision; these tests pin it against both the normal
case (hazard nowhere near the goal - marked as usual) and the failure case
(hazard within clearance of the goal - skipped).
"""

import importlib.util
import math
from pathlib import Path
import sys

FLIP_RECOVERY_NODE = Path(__file__).resolve().parent.parent / "scripts/flip_recovery_node.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("flip_recovery_node", FLIP_RECOVERY_NODE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["flip_recovery_node"] = module
    spec.loader.exec_module(module)
    return module


flip_recovery_node = _load_module()
hazard_point_xy = flip_recovery_node.hazard_point_xy
hazard_too_close_to_goal = flip_recovery_node.hazard_too_close_to_goal


def test_hazard_point_is_lead_distance_ahead_along_heading():
    # Facing +x: the hazard lands lead_m further along +x, same y.
    x, y = hazard_point_xy((10.0, -5.0), yaw=0.0, lead_m=0.8)
    assert x == 10.8
    assert abs(y - (-5.0)) < 1e-9

    # Facing +y (90 deg): lands lead_m further along +y, same x.
    x, y = hazard_point_xy((0.0, 0.0), yaw=math.pi / 2, lead_m=1.0)
    assert abs(x) < 1e-9
    assert abs(y - 1.0) < 1e-9


def test_hazard_far_from_goal_is_not_too_close():
    """The ordinary case: the rover wedges somewhere unrelated to the goal."""
    hazard_xy = (10.0, 10.0)
    goal_xy = (-60.58, 19.42)  # seed 55's actual goal, for concreteness
    assert not hazard_too_close_to_goal(hazard_xy, goal_xy, clearance_m=1.5)


def test_hazard_on_top_of_goal_is_too_close():
    """The failure mode this guard exists for: severe divergence puts the
    estimated pose (and so the hazard marked ahead of it) within a metre of
    the real goal - reproducing the seed 55 numbers directly (EKF estimate
    (-59.599, 19.588) against goal (-60.58, 19.42), about 1.0 m apart)."""
    hazard_xy = (-59.599, 19.588)
    goal_xy = (-60.58, 19.42)
    assert hazard_too_close_to_goal(hazard_xy, goal_xy, clearance_m=1.5)


def test_clearance_boundary_is_a_strict_less_than():
    goal_xy = (0.0, 0.0)
    clearance = 1.5
    just_inside = (clearance - 0.01, 0.0)
    just_outside = (clearance + 0.01, 0.0)
    exactly_on = (clearance, 0.0)
    assert hazard_too_close_to_goal(just_inside, goal_xy, clearance)
    assert not hazard_too_close_to_goal(just_outside, goal_xy, clearance)
    assert not hazard_too_close_to_goal(exactly_on, goal_xy, clearance)
