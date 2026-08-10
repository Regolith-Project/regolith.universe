# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for mission_markers_node.py's flag geometry and goal bookkeeping.

The node's ROS plumbing needs a running stack to mean anything, so what is covered here
is the part that is pure logic and the part that failed first in practice:

  * the SDF a flag is spawned from. It is assembled by string formatting and then
    escaped into a protobuf text field, so a stray quote or newline turns into a service
    call gz rejects - silently, from the node's point of view.
  * NO COLLISION. A decoration the rover can wedge itself on would be a new failure mode
    invented by a debugging aid, on a rover whose headline known issue is getting wedged.
  * matching a published goal to a planned waypoint, which is what decides whether a
    goal shows as "G3 ACTIVE" or as a separate ad-hoc goal marker.
"""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

MARKERS_NODE = Path(__file__).resolve().parent.parent / "scripts/mission_markers_node.py"


def _load_module():
    """Import the node module without importing rclpy - these tests are about its
    geometry and bookkeeping, and CI has no ROS graph."""
    for name in ("rclpy", "rclpy.node", "rclpy.qos", "geometry_msgs", "geometry_msgs.msg",
                 "nav_msgs", "nav_msgs.msg", "std_msgs", "std_msgs.msg",
                 "visualization_msgs", "visualization_msgs.msg"):
        sys.modules.setdefault(name, mock.MagicMock())
    spec = importlib.util.spec_from_file_location("mission_markers_node", MARKERS_NODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


markers = _load_module()


def test_flag_sdf_is_a_single_line():
    """The create service takes the SDF inside a protobuf text-format string, which
    cannot contain a raw newline - one would make every spawn fail."""
    sdf = markers._flag_sdf("mission_flag_start", 1.0, -2.0, 5.5, (0.1, 0.9, 0.25))
    assert "\n" not in sdf and "\r" not in sdf


def test_flag_has_no_collision():
    """Visual only. The rover drives through the flags."""
    sdf = markers._flag_sdf("mission_flag_wp_1", 12.0, 8.0, 5.0, (0.95, 0.65, 0.10))
    assert "<collision" not in sdf
    assert sdf.count("<visual") == 2  # pole and banner


def test_flag_is_static_and_placed_where_asked():
    sdf = markers._flag_sdf("mission_flag_wp_2", 18.0, -4.0, 6.25, (0.95, 0.65, 0.10))
    assert "<static>true</static>" in sdf
    assert "<pose>18.000 -4.000 6.250 0 0 0</pose>" in sdf


def test_flag_sdf_survives_the_protobuf_escaping():
    """The exact round trip the node performs before handing the string to gz: every
    quote in the SDF has to come back out as a quote, or the model never appears."""
    sdf = markers._flag_sdf("mission_flag_active", 0.0, 0.0, 5.0, (1.0, 0.25, 0.10))
    escaped = sdf.replace("\\", "\\\\").replace('"', '\\"')
    assert '\\"' in escaped
    # Undo it the way a text-format parser does, and the original must be back.
    assert escaped.replace('\\"', '"').replace("\\\\", "\\") == sdf


def test_the_banner_hangs_off_the_top_of_the_pole():
    """A banner drawn below its pole, or floating above it, reads as a glitch rather
    than a flag - and it is only visible at a glance in a very dark scene."""
    height = markers.POLE_M
    sdf = markers._flag_sdf("f", 0.0, 0.0, 0.0, (1.0, 1.0, 1.0), height)
    banner_z = float(sdf.split('<visual name="banner"><pose>')[1].split()[2])
    assert banner_z + markers.BANNER_H / 2 <= height
    assert banner_z > height * 0.6


@pytest.mark.parametrize(
    "goal, expected",
    [
        ((12.0, 8.0), 0),        # exactly a planned waypoint
        ((12.4, 8.3), 0),        # a re-publish nudged within tolerance
        ((4.0, -14.0), 2),
        ((-30.0, 25.0), None),   # an ad-hoc goal clicked in RViz
    ],
)
def test_a_goal_is_matched_to_the_waypoint_it_belongs_to(goal, expected):
    """Goals are re-published by more than one node (pure pursuit and flip recovery both
    do), so identity by proximity rather than by equality is what keeps a single
    waypoint from sprouting a second flag beside it."""
    waypoints = [(12.0, 8.0), (18.0, -4.0), (4.0, -14.0), (-10.0, -6.0), (0.0, 0.0)]
    assert markers.waypoint_near(waypoints, *goal) == expected


def test_tolerance_is_smaller_than_the_gap_between_tour_waypoints():
    """Otherwise one flag would claim its neighbour's goals.

    Checked against the route builder's own guarantees rather than against a list of
    coordinates - the tour is drawn from the costmap now, so there is no fixed list to
    compare with, and MIN_SEPARATION_M is what bounds how close two of them can get.
    """
    from regolith_planner.tour import MIN_SEPARATION_M

    assert markers.SAME_WAYPOINT_M < MIN_SEPARATION_M / 2
