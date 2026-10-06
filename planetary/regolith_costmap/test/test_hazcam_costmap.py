# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""The hazcam layer inside the costmap: marking, goal protection, replan triggering.

Pure functions only, no ROS graph. The last test runs the whole chain on a
rendered depth image: rock -> detector -> odom -> PointCloud2 -> costmap cells ->
a path through the rock is reported as blocked, and A* routes around it.
"""

import math

import numpy as np
import pytest

from regolith_costmap.costmap_node import detection_cells
from regolith_costmap.costmap_node import goal_protection_mask
from regolith_costmap.costmap_node import mark_hazcam_cells
from regolith_costmap.costmap_node import path_cells_ahead
from regolith_costmap.costmap_node import ReplanLimiter
from regolith_costmap.costmap_node import world_to_cell
from regolith_costmap.hazcam import level_to_odom
from regolith_costmap.hazcam_cloud import make_cloud
from regolith_costmap.hazcam_cloud import xy_from_cloud

ORIGIN = (-20.0, -20.0)
RES = 1.0
SHAPE = (40, 40)
INFLATION = 1  # ceil(rover_radius_m 0.3 / 1.0 m), what the launch configures


def _grid():
    return np.zeros(SHAPE, dtype=np.int8)


def _no_protection():
    return goal_protection_mask(None, 0.0, ORIGIN, RES, SHAPE)


def _report(grid, hits, points, protected, confirm=2):
    return mark_hazcam_cells(grid, detection_cells(points, ORIGIN, RES, SHAPE), hits,
                             confirm, INFLATION, protected)


def test_confirmed_detection_marks_lethal_with_rover_inflation():
    grid, hits = _grid(), {}
    point = [(2.3, -4.6)]
    assert _report(grid, hits, point, _no_protection()) == []  # one frame is not enough
    assert (grid == 100).sum() == 0
    newly = _report(grid, hits, point, _no_protection())
    row, col = world_to_cell(2.3, -4.6, ORIGIN, RES)
    assert grid[row, col] == 100
    # Same inflation as build_costmap: the 8-connected ring of one cell.
    assert (grid[row - 1: row + 2, col - 1: col + 2] == 100).all()
    assert (grid == 100).sum() == 9 and len(newly) == 9
    # A third report changes nothing and reports nothing new.
    assert _report(grid, hits, point, _no_protection()) == []


def test_two_points_in_one_cell_in_one_frame_count_once():
    grid, hits = _grid(), {}
    assert _report(grid, hits, [(2.1, -4.1), (2.8, -4.9)], _no_protection()) == []
    assert (grid == 100).sum() == 0


def test_marks_never_lower_costs_and_report_only_changes():
    grid, hits = np.full(SHAPE, 40, dtype=np.int8), {}
    row, col = world_to_cell(0.5, 0.5, ORIGIN, RES)
    grid[row, col + 1] = 100  # a-priori lethal neighbour
    for _ in range(2):
        newly = _report(grid, hits, [(0.5, 0.5)], _no_protection())
    assert (row, col + 1) not in newly and len(newly) == 8
    assert grid[0, 0] == 40


def test_detection_on_the_goal_cell_is_not_marked():
    goal = (5.4, 7.2)
    protected = goal_protection_mask(goal, 1.5, ORIGIN, RES, SHAPE)
    grid, hits = _grid(), {}
    for _ in range(3):
        _report(grid, hits, [goal], protected)
    goal_rc = world_to_cell(*goal, ORIGIN, RES)
    assert grid[goal_rc] < 100
    assert (grid[protected] < 100).all()


def test_a_ring_of_detections_round_the_goal_cannot_wall_it_off():
    astar = pytest.importorskip("regolith_planner.astar")
    goal = (5.5, 7.5)
    protected = goal_protection_mask(goal, 1.5, ORIGIN, RES, SHAPE)
    grid, hits = _grid(), {}
    ring = [(goal[0] + dx, goal[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    for _ in range(2):
        _report(grid, hits, ring, protected)
    goal_rc = world_to_cell(*goal, ORIGIN, RES)
    assert grid[goal_rc] < 100
    start_rc = world_to_cell(-15.0, -15.0, ORIGIN, RES)
    assert astar.plan_path(grid, start_rc, goal_rc)


def test_protected_cell_is_marked_once_the_goal_moves_on():
    goal = (5.4, 7.2)
    grid, hits = _grid(), {}
    for _ in range(2):
        _report(grid, hits, [goal], goal_protection_mask(goal, 1.5, ORIGIN, RES, SHAPE))
    goal_rc = world_to_cell(*goal, ORIGIN, RES)
    assert grid[goal_rc] < 100
    _report(grid, hits, [goal], goal_protection_mask((-10.0, -10.0), 1.5, ORIGIN, RES, SHAPE))
    assert grid[goal_rc] == 100


def test_out_of_bounds_and_non_finite_points_are_ignored():
    cells = detection_cells([(1e6, 0.0), (float("nan"), 1.0), (0.0, float("inf")), (0.5, 0.5)],
                            ORIGIN, RES, SHAPE)
    assert cells == {world_to_cell(0.5, 0.5, ORIGIN, RES)}


def test_path_ahead_only():
    path = [(x, 0.5) for x in np.linspace(-10.0, 10.0, 11)]
    behind = world_to_cell(-8.5, 0.5, ORIGIN, RES)
    ahead = world_to_cell(6.5, 0.5, ORIGIN, RES)
    cells = path_cells_ahead(path, (-2.0, 0.6), ORIGIN, RES, SHAPE)
    assert ahead in cells and behind not in cells
    # Sampled finely enough that a long segment cannot jump a cell.
    assert all(world_to_cell(x + 0.5, 0.5, ORIGIN, RES) in cells for x in range(-2, 10))
    assert path_cells_ahead(None, (0.0, 0.0), ORIGIN, RES, SHAPE) == set()


def test_replan_limiter_holds_rather_than_drops():
    limiter = ReplanLimiter(5.0)
    assert not limiter.poll(0.0)  # nothing asked for
    limiter.request()
    assert limiter.poll(10.0)
    assert not limiter.poll(10.5)  # one request, one replan
    limiter.request()
    assert not limiter.poll(12.0)  # inside the interval: held
    assert not limiter.poll(14.9)
    assert limiter.poll(15.0)  # released as soon as it has passed
    limiter.request()
    limiter.cancel()
    assert not limiter.poll(100.0)


def test_cloud_round_trip():
    xy = np.array([[1.25, -3.5], [100.0, 0.125]])
    msg = make_cloud(xy, np.array([0.3, 0.2]), "odom", _stamp())
    assert msg.header.frame_id == "odom" and msg.width == 2
    assert xy_from_cloud(msg) == pytest.approx(xy)
    empty = make_cloud(np.zeros((0, 2)), np.zeros(0), "odom", _stamp())
    assert xy_from_cloud(empty).shape == (0, 2)
    msg.fields = [f for f in msg.fields if f.name != "y"]
    with pytest.raises(ValueError):
        xy_from_cloud(msg)


def _stamp():
    from builtin_interfaces.msg import Time
    return Time(sec=12, nanosec=5)


def test_rendered_rock_on_the_path_blocks_it_and_astar_goes_round():
    """End to end without ROS: a 0.3 m rock 5 m ahead of a rover driving +x."""
    astar = pytest.importorskip("regolith_planner.astar")
    import test_hazcam_geometry as scene

    base_xy, yaw = (-6.2, 0.4), 0.0
    box, rock_xy = scene._box_at(base_xy, yaw, 5.0, 0.0, 0.4, 0.3)
    base, roll, pitch, _ = scene.flat_pose(*base_xy, yaw)
    det = scene.detect(scene.render(base, roll, pitch, yaw, boxes=[box]), roll, pitch)
    odom = level_to_odom(det.xy_level, base_xy[0], base_xy[1], yaw)
    msg = make_cloud(odom, det.step_m, "odom", _stamp())

    goal = (12.5, 0.5)
    protected = goal_protection_mask(goal, 1.5, ORIGIN, RES, SHAPE)
    grid, hits = _grid(), {}
    newly = []
    for _ in range(2):  # two frames
        newly += _report(grid, hits, xy_from_cloud(msg), protected)
    assert grid[world_to_cell(*rock_xy, ORIGIN, RES)] == 100

    straight = [(x, 0.5) for x in np.linspace(base_xy[0], goal[0], 20)]
    assert path_cells_ahead(straight, base_xy, ORIGIN, RES, SHAPE) & set(newly)

    start_rc = world_to_cell(*base_xy, ORIGIN, RES)
    goal_rc = world_to_cell(*goal, ORIGIN, RES)
    route = astar.plan_path(grid, start_rc, goal_rc)
    assert route and all(grid[rc] < 100 for rc in route)
    rock_rc = world_to_cell(*rock_xy, ORIGIN, RES)
    assert min(math.hypot(r - rock_rc[0], c - rock_rc[1]) for r, c in route) >= 2
