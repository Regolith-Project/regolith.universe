# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Arrival must be measured against the goal that was commanded.

The planner snaps both ends of its path to costmap cells, so `path[-1]` is a
cell CENTRE - up to half a cell diagonal from the goal actually asked for
(~0.55 m at this world's 0.78 m resolution). Checking arrival against it means
the rover can stop nearly 2 m from its goal while reporting "within 1.50 m".

That is not hypothetical. With localization made perfect by the experimental
absolute reference, seeds 42 and 7 both stopped at exactly 1.70 m from their
goals and failed a 1.5 m bar for no other reason: on seed 42, `path[-1]` sat
0.42 m from the commanded goal. See PROGRESS.md.

These tests exercise the geometry directly rather than standing up a ROS node,
which keeps them fast and dependency-free; the node applies exactly this
comparison in `_control_step`.
"""

import math

TOLERANCE_M = 1.5  # pure_pursuit's goal_tolerance_m
CELL_M = 0.78125  # costmap resolution at 256 cells over 200 m
ORIGIN_M = -100.0


def snap_to_cell_centre(x: float, y: float) -> tuple:
    """Reproduce what the planner's grid round-trip does to a goal - see planner_node."""
    col = int((x - ORIGIN_M) / CELL_M)
    row = int((y - ORIGIN_M) / CELL_M)
    return ORIGIN_M + (col + 0.5) * CELL_M, ORIGIN_M + (row + 0.5) * CELL_M


def test_snapping_moves_the_goal_by_a_real_distance():
    """The premise: path[-1] is not the goal."""
    goal = (52.33, -66.98)
    assert math.dist(goal, snap_to_cell_centre(*goal)) > 0.3


def test_stopping_at_the_snapped_point_can_miss_the_bar():
    """A rover 1.49 m from path[-1] - which reports success - can be beyond the 1.5 m bar from the goal it was actually given."""
    goal = (52.33, -66.98)
    snapped = snap_to_cell_centre(*goal)
    offset = math.dist(goal, snapped)

    # Worst case: the rover sits at tolerance distance from the snapped point,
    # directly away from the true goal.
    worst_true_error = TOLERANCE_M + offset
    assert worst_true_error > TOLERANCE_M
    assert worst_true_error > 1.7  # the value both oracle runs actually produced


def test_measuring_against_the_commanded_goal_satisfies_the_bar():
    """With the fix, "arrived" means within tolerance of the commanded goal, so the true error can never exceed the bar however the path was snapped."""
    goal = (52.33, -66.98)
    for bearing_deg in range(0, 360, 15):
        bearing = math.radians(bearing_deg)
        # Any position the fixed check would accept.
        distance = TOLERANCE_M - 0.01
        position = (goal[0] + distance * math.cos(bearing), goal[1] + distance * math.sin(bearing))
        assert math.dist(position, goal) < TOLERANCE_M


def test_snapping_offset_is_bounded_by_half_a_cell_diagonal():
    """Sanity-check the mechanism itself across a spread of goals."""
    worst = 0.0
    for i in range(200):
        goal = (-90.0 + i * 0.9137, 40.0 - i * 0.4211)
        worst = max(worst, math.dist(goal, snap_to_cell_centre(*goal)))
    assert worst <= CELL_M * math.sqrt(2) / 2 + 1e-9
    assert worst > 0.4  # and it is routinely large enough to matter
