# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for terrain_relative_node.py's matcher.

The matcher's job is to recover a position offset from the attitude a rover
measured while driving over known terrain. Two failure modes matter more than
raw accuracy, and most of this file is about them:

  1. CONFIDENTLY WRONG on featureless ground. Flat terrain determines nothing,
     and a matcher that returns its best guess anyway feeds the EKF a fabricated
     absolute position - strictly worse than the dead reckoning it replaces. The
     `margin` this returns has to collapse toward 1.0 there, because that is what
     the node's gate acts on.
  2. A SIGN OR AXIS ERROR in the attitude prediction. Roll and pitch are
     projections of the terrain gradient onto the heading, and a transposed or
     negated field still produces plausible-looking numbers and a plausible-
     looking match - it just localises to the wrong place. `load_heightmap`
     carries the same hazard for the costmap and it was a live defect once, so
     the geometry here is pinned against hand-computed slopes rather than
     against the matcher's own output.

Accuracy itself is not asserted here beyond "recovers a known offset on a
synthetic slope": the real accuracy claim is a measurement over 25 recorded runs,
not something a unit test can establish. See PROGRESS.md.
"""

import importlib.util
import math
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import pytest

# The node imports rclpy, the message packages and regolith_costmap at module
# scope; none of them is needed by the pure functions under test, and none is
# guaranteed importable in a bare test environment. Stub ONLY what is genuinely
# missing, and remove those stubs again once the module is loaded: leaving a
# MagicMock behind under a real package's name is invisible here and breaks
# whichever test file imports it for real later in the same pytest session,
# which is exactly what happened the first time this file was added.
_STUBBED = []
for _name in ("rclpy", "rclpy.node", "geometry_msgs", "geometry_msgs.msg", "nav_msgs",
              "nav_msgs.msg", "sensor_msgs", "sensor_msgs.msg", "regolith_costmap",
              "regolith_costmap.costmap_node"):
    if _name in sys.modules:
        continue
    try:
        importlib.import_module(_name)
    except ImportError:
        sys.modules[_name] = mock.MagicMock()
        _STUBBED.append(_name)

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "terrain_relative_node.py"
_spec = importlib.util.spec_from_file_location("terrain_relative_node", _PATH)
trn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trn)

for _name in _STUBBED:
    del sys.modules[_name]


WORLD_M = 100.0
RES_M = 0.5


def _plane_gradients(slope_x: float, slope_y: float):
    """Uniform-gradient fields covering a WORLD_M world at RES_M."""
    n = int(WORLD_M / RES_M) + 1
    return np.full((n, n), slope_x), np.full((n, n), slope_y)


def _bumpy_gradients(seed: int = 0):
    """A random but smooth gradient field - terrain with enough relief to localise on."""
    n = int(WORLD_M / RES_M) + 1
    rng = np.random.default_rng(seed)
    dem = rng.normal(size=(n, n))
    # Smooth by repeated box averaging so the field has structure at metre scale
    # rather than per-pixel noise, which is what real terrain looks like to a
    # 0.5 m rover.
    for _ in range(6):
        dem = (dem + np.roll(dem, 1, 0) + np.roll(dem, -1, 0)
               + np.roll(dem, 1, 1) + np.roll(dem, -1, 1)) / 5.0
    dem *= 3.0 / (dem.std() + 1e-9)
    gy, gx = np.gradient(dem, RES_M)
    return gx, gy


class TestPredictedAttitude:
    """The geometry, pinned to hand-computed values rather than to itself."""

    def test_driving_up_a_pure_x_slope_is_pitch_not_roll(self):
        gx, gy = _plane_gradients(0.1, 0.0)
        roll, pitch = trn.predicted_attitude(gx, gy, WORLD_M, RES_M, 0.0, 0.0, 0.0)
        assert pitch == pytest.approx(-math.atan(0.1), abs=1e-6)
        assert roll == pytest.approx(0.0, abs=1e-9)

    def test_the_same_slope_crossed_sideways_is_roll_not_pitch(self):
        """Heading 90 deg on an x-slope: the slope is now entirely to the rover's side."""
        gx, gy = _plane_gradients(0.1, 0.0)
        roll, pitch = trn.predicted_attitude(gx, gy, WORLD_M, RES_M, 0.0, 0.0, math.pi / 2)
        assert pitch == pytest.approx(0.0, abs=1e-9)
        assert roll == pytest.approx(-math.atan(0.1), abs=1e-6)

    def test_climbing_gives_negative_pitch_and_descending_positive(self):
        """Sign check, in the convention m4_acceptance and the IMU both report.

        Driving in +x up a surface rising in +x is nose-up. This test exists
        because a flipped sign here does not crash anything - it just localises
        the rover to a place where the terrain falls the way it should rise.
        """
        gx, gy = _plane_gradients(0.2, 0.0)
        _, uphill = trn.predicted_attitude(gx, gy, WORLD_M, RES_M, 0.0, 0.0, 0.0)
        _, downhill = trn.predicted_attitude(gx, gy, WORLD_M, RES_M, 0.0, 0.0, math.pi)
        assert uphill < 0.0 < downhill
        assert uphill == pytest.approx(-downhill, abs=1e-9)

    def test_y_gradient_is_read_from_the_y_field(self):
        """Guards the axis swap: gx and gy are not interchangeable."""
        gx, gy = _plane_gradients(0.0, 0.15)
        roll, pitch = trn.predicted_attitude(gx, gy, WORLD_M, RES_M, 0.0, 0.0, 0.0)
        assert pitch == pytest.approx(0.0, abs=1e-9)
        assert roll == pytest.approx(math.atan(0.15), abs=1e-6)

    def test_samples_outside_the_world_are_clamped_not_nan(self):
        """A wild candidate offset must score badly, not poison the cost surface."""
        gx, gy = _bumpy_gradients()
        roll, pitch = trn.predicted_attitude(
            gx, gy, WORLD_M, RES_M, np.array([1e6, -1e6]), np.array([1e6, -1e6]),
            np.array([0.0, 0.0])
        )
        assert np.all(np.isfinite(roll)) and np.all(np.isfinite(pitch))


class TestMatchOffset:
    def _trajectory(self, n=60, length_m=25.0):
        xs = np.linspace(-length_m / 2, length_m / 2, n)
        ys = np.linspace(-5.0, 5.0, n)
        yaws = np.full(n, math.atan2(10.0, length_m))
        return xs, ys, yaws

    def test_recovers_a_known_offset_from_synthetic_terrain(self):
        """The core claim: attitude measured at a displaced truth locates that displacement."""
        gx, gy = _bumpy_gradients(seed=3)
        xs, ys, yaws = self._trajectory()
        true_dx, true_dy = 1.5, -2.0
        rolls, pitches = trn.predicted_attitude(
            gx, gy, WORLD_M, RES_M, xs + true_dx, ys + true_dy, yaws
        )
        dx, dy, margin = trn.match_offset(
            gx, gy, WORLD_M, RES_M, xs, ys, yaws, rolls, pitches, search_m=4.0, step_m=0.25
        )
        assert dx == pytest.approx(true_dx, abs=0.3)
        assert dy == pytest.approx(true_dy, abs=0.3)
        assert margin > 5.0, "a noise-free match on textured terrain should be unambiguous"

    def test_sub_step_precision_beats_the_search_grid(self):
        """The parabolic refinement has to actually refine: 0.6 m is not on a 0.5 m grid."""
        gx, gy = _bumpy_gradients(seed=11)
        xs, ys, yaws = self._trajectory()
        rolls, pitches = trn.predicted_attitude(gx, gy, WORLD_M, RES_M, xs + 0.6, ys, yaws)
        dx, _, _ = trn.match_offset(
            gx, gy, WORLD_M, RES_M, xs, ys, yaws, rolls, pitches, search_m=3.0, step_m=0.5
        )
        assert dx != pytest.approx(0.5, abs=1e-9)
        assert dx == pytest.approx(0.6, abs=0.2)

    def test_flat_terrain_is_reported_as_ambiguous_not_guessed(self):
        """The failure mode that would actively harm the filter.

        On a uniform plane every candidate offset explains the measurements
        equally well, so there is no information about position at all. What
        matters is not which offset comes back - it is that `margin` collapses to
        ~1.0, which is the node's signal to publish nothing.
        """
        gx, gy = _plane_gradients(0.05, 0.02)
        xs, ys, yaws = self._trajectory()
        rolls, pitches = trn.predicted_attitude(gx, gy, WORLD_M, RES_M, xs, ys, yaws)
        _, _, margin = trn.match_offset(
            gx, gy, WORLD_M, RES_M, xs, ys, yaws, rolls, pitches, search_m=4.0, step_m=0.5
        )
        assert margin < 1.5, f"flat ground reported as a confident fix (margin {margin})"

    def test_textured_terrain_is_reported_as_confident(self):
        """The other half of the gate: it must not reject everything either."""
        gx, gy = _bumpy_gradients(seed=5)
        xs, ys, yaws = self._trajectory()
        rolls, pitches = trn.predicted_attitude(gx, gy, WORLD_M, RES_M, xs, ys, yaws)
        _, _, margin = trn.match_offset(
            gx, gy, WORLD_M, RES_M, xs, ys, yaws, rolls, pitches, search_m=4.0, step_m=0.5
        )
        assert margin > 2.0

    def test_attitude_noise_degrades_the_fix_without_breaking_it(self):
        """Real measurements carry ~1.2 deg of residual the DEM cannot explain."""
        gx, gy = _bumpy_gradients(seed=7)
        xs, ys, yaws = self._trajectory(n=120)
        rolls, pitches = trn.predicted_attitude(gx, gy, WORLD_M, RES_M, xs + 1.0, ys, yaws)
        rng = np.random.default_rng(42)
        noise = math.radians(1.2)
        dx, dy, _ = trn.match_offset(
            gx, gy, WORLD_M, RES_M, xs, ys, yaws,
            rolls + rng.normal(0, noise, len(xs)),
            pitches + rng.normal(0, noise, len(xs)),
            search_m=4.0, step_m=0.25,
        )
        assert math.hypot(dx - 1.0, dy) < 1.5

    def test_the_search_range_bounds_the_correction(self):
        """A fix can never move the estimate further than search_m, by construction.

        This is what makes a gross mismatch survivable rather than catastrophic,
        and the node's consistency gate assumes it.
        """
        gx, gy = _bumpy_gradients(seed=9)
        xs, ys, yaws = self._trajectory()
        rolls, pitches = trn.predicted_attitude(gx, gy, WORLD_M, RES_M, xs + 30.0, ys, yaws)
        dx, dy, _ = trn.match_offset(
            gx, gy, WORLD_M, RES_M, xs, ys, yaws, rolls, pitches, search_m=3.0, step_m=0.25
        )
        assert abs(dx) <= 3.0 + 0.25 and abs(dy) <= 3.0 + 0.25


class TestQuaternionConversion:
    def test_identity_is_level(self):
        q = mock.Mock(w=1.0, x=0.0, y=0.0, z=0.0)
        roll, pitch, yaw = trn.rpy_from_quaternion(q)
        assert (roll, pitch, yaw) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)

    def test_quarter_turn_about_z_is_yaw_only(self):
        h = math.sqrt(0.5)
        q = mock.Mock(w=h, x=0.0, y=0.0, z=h)
        roll, pitch, yaw = trn.rpy_from_quaternion(q)
        assert yaw == pytest.approx(math.pi / 2, abs=1e-9)
        assert roll == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-9)

    def test_gimbal_input_is_clamped_rather_than_raising(self):
        """asin's argument can exceed 1 by float error on a near-vertical quaternion."""
        h = math.sqrt(0.5)
        q = mock.Mock(w=h, x=0.0, y=h, z=0.0)
        _, pitch, _ = trn.rpy_from_quaternion(q)
        assert pitch == pytest.approx(math.pi / 2, abs=1e-6)


class TestReconstructWindow:
    """The anchored window. Three live runs failed on the code this replaced."""

    def test_a_straight_path_lays_out_behind_the_anchor(self):
        steps = [0.0, 1.0, 1.0, 1.0]
        yaws = [0.0, 0.0, 0.0, 0.0]           # heading +x
        xs, ys = trn.reconstruct_window(steps, yaws, 10.0, 5.0)
        assert xs == pytest.approx([7.0, 8.0, 9.0, 10.0])
        assert ys == pytest.approx([5.0, 5.0, 5.0, 5.0])

    def test_the_newest_sample_always_sits_exactly_on_the_anchor(self):
        """This is the property that makes the loop converge instead of running away."""
        rng = np.random.default_rng(0)
        steps = np.concatenate([[0.0], rng.uniform(0.1, 0.4, 40)])
        yaws = np.cumsum(rng.normal(0, 0.2, 41))
        xs, ys = trn.reconstruct_window(steps, yaws, -3.25, 7.5)
        assert xs[-1] == pytest.approx(-3.25, abs=1e-12)
        assert ys[-1] == pytest.approx(7.5, abs=1e-12)

    def test_moving_the_anchor_translates_the_whole_window_rigidly(self):
        """When the filter absorbs a correction the window must follow it exactly."""
        rng = np.random.default_rng(1)
        steps = np.concatenate([[0.0], rng.uniform(0.1, 0.4, 30)])
        yaws = np.cumsum(rng.normal(0, 0.3, 31))
        x0, y0 = trn.reconstruct_window(steps, yaws, 0.0, 0.0)
        x1, y1 = trn.reconstruct_window(steps, yaws, 2.5, -1.25)
        assert (x1 - x0) == pytest.approx(np.full(len(x0), 2.5))
        assert (y1 - y0) == pytest.approx(np.full(len(y0), -1.25))

    def test_a_turn_is_integrated_segment_by_segment(self):
        """A right-angle turn: total distance times one heading gets this wrong.

        That exact error scored 3.19 m against 0.42 m in replay, so it is pinned
        to a hand-computed answer rather than to the function's own output.
        """
        steps = [0.0, 2.0, 3.0]
        yaws = [0.0, 0.0, math.pi / 2]        # 2 m east, then 3 m north
        xs, ys = trn.reconstruct_window(steps, yaws, 0.0, 0.0)
        assert xs == pytest.approx([-2.0, 0.0, 0.0], abs=1e-9)
        assert ys == pytest.approx([-3.0, -3.0, 0.0], abs=1e-9)

    def test_the_first_step_is_ignored_not_applied(self):
        steps_a = [0.0, 1.0]
        steps_b = [99.0, 1.0]
        assert trn.reconstruct_window(steps_a, [0.0, 0.0], 0.0, 0.0)[0] == pytest.approx(
            trn.reconstruct_window(steps_b, [0.0, 0.0], 0.0, 0.0)[0]
        )


class TestClampCorrection:
    def test_a_small_correction_passes_through_untouched(self):
        assert trn.clamp_correction(0.03, -0.04, 0.10) == pytest.approx((0.03, -0.04))

    def test_a_large_correction_is_shortened_but_keeps_its_direction(self):
        dx, dy = trn.clamp_correction(3.0, 4.0, 0.10)
        assert math.hypot(dx, dy) == pytest.approx(0.10)
        assert dy / dx == pytest.approx(4.0 / 3.0)

    def test_zero_disables_the_clamp(self):
        assert trn.clamp_correction(9.0, 9.0, 0.0) == (9.0, 9.0)

    def test_a_zero_correction_does_not_divide_by_zero(self):
        assert trn.clamp_correction(0.0, 0.0, 0.10) == (0.0, 0.0)


class TestNoBufferShift:
    """The buffer must stay in the estimator's frame - a regression test in prose.

    `shift_samples` used to move the whole window by each published correction.
    Under sparse fixes that was right; under 1 Hz publishing it was the bug that
    made a live run oscillate between 0.5 and 13 m of divergence, because the
    buffer moved by the REQUESTED correction while the filter absorbed only part
    of it. The match then found the shifted buffer already aligned, reported
    (-0.00, +0.00), and the node certified a several-metre error as correct once
    a second.

    There is no function left to unit-test, so this asserts the property that
    replaced it: nothing in the module shifts a sample buffer.
    """

    def test_the_module_has_no_buffer_shifting_helper(self):
        assert not hasattr(trn, "shift_samples"), (
            "shift_samples is back - under dense publishing it pins the "
            "estimator's error in place instead of correcting it"
        )
