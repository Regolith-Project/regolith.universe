# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for wheel_slip_node.py's SlipDetector.

The detector's job is to tell "the wheels are turning and the rover is going
nowhere" from "the wheels are turning and the rover is driving", using wheel
odometry and an IMU and nothing else. The case that matters most here is the
FALSE POSITIVE: declaring slip on a moving rover feeds the EKF a zero-velocity
update that is just as wrong as the phantom odometry it exists to suppress, so
the driving cases below are the point of this file, not filler.

The numbers come from a recorded run (see calibrate_slip_detector.py and
PROGRESS.md), not from taste: over 4,700 windows of genuine driving the
smallest attitude span seen in a 15 s window was 0.0276 rad, against the
0.010 rad threshold `_body_is_rigid` used to act on.

`_body_is_rigid` ("signature 2") is retired as of 2026-08-20 - it no longer
decides `slipping()` or `clearing()` - after a live reproduction proved it
false-positives on ordinary straight-line driving over smooth ground (see
PROGRESS.md, "Root-caused: the benign-ground traction stall was never a
stall"). Tests that exercised it now assert the RETIRED behaviour and say so
in their docstrings; the attitude-span numbers above stay as the historical
basis for the 15 s window, which signature 1 still needs.
"""

import importlib.util
import math
from pathlib import Path
import sys

import pytest

SLIP_NODE = Path(__file__).resolve().parent.parent / "scripts/wheel_slip_node.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("wheel_slip_node", SLIP_NODE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["wheel_slip_node"] = module
    spec.loader.exec_module(module)
    return module


wheel_slip_node = _load_module()
SlipDetector = wheel_slip_node.SlipDetector


def feed(detector, duration_s, vx, wz_wheel, wz_gyro, attitude_rate=0.0, rate_hz=10.0):
    """Feed `duration_s` of samples; attitude_rate tilts the body over time."""
    steps = int(duration_s * rate_hz)
    start = detector._samples[-1][0] if detector._samples else 0.0
    for i in range(1, steps + 1):
        t = start + i / rate_hz
        roll = attitude_rate * t
        detector.add(t, vx, wz_wheel, wz_gyro, (roll, 0.0, 0.0))
    return detector


def test_symmetric_wedge_without_commanded_turn_is_a_known_undetected_gap():
    """Wheels claiming 0.2 m/s with zero commanded turn is a known undetected gap.

    Zero attitude change there is byte-for-byte ambiguous: it is both a rover
    wedged symmetrically enough to produce no net torque, and a rover driving
    dead straight at 0.2 m/s across flat, uniform ground. Onboard wheel
    odometry + IMU cannot tell these apart: neither of signature 2's inputs
    (attitude span, gyro RMS) observes translation at all, so no threshold on
    them separates the cases.

    This used to be flagged (signature 2, "a rigidly still body"), on the
    theory a symmetric wedge was more likely than 15 s of dead-straight
    driving over smooth ground. That theory was live-tested on seed 42
    (PROGRESS.md, "Root-caused: the benign-ground traction stall was never a
    stall") and produced a confirmed false positive - a genuinely moving
    rover flagged as wedged, its real wheel odometry fed to the EKF as a
    zero-velocity update - against zero true positives for signature 2
    across two calibration campaigns (7,755 + 968 windows). Retired. This
    case is now silently undetected, which is the honest state of the
    tradeoff, not an oversight: real coverage would need a signal that
    observes translation, which this detector does not have.
    """
    detector = SlipDetector()
    feed(detector, 20.0, vx=0.2, wz_wheel=0.0, wz_gyro=0.0)
    assert not detector.slipping()


def test_legacy_rigid_body_signature_lever_restores_the_retired_behaviour():
    """The A/B lever restores the retired signature-2 behaviour when enabled.

    `legacy_rigid_body_signature` exists so a before/after campaign can
    compare the two behaviours from the same build/commit, same discipline
    as goal_tolerance_m's campaign. This pins that the lever actually does
    what the campaign needs it to do: off (default) matches the current,
    fixed behaviour
    (test_symmetric_wedge_without_commanded_turn_is_a_known_undetected_gap);
    on restores the exact pre-fix false positive.
    """
    fixed = SlipDetector(legacy_rigid_body_signature=False)
    feed(fixed, 20.0, vx=0.2, wz_wheel=0.0, wz_gyro=0.0)
    assert not fixed.slipping()

    legacy = SlipDetector(legacy_rigid_body_signature=True)
    feed(legacy, 20.0, vx=0.2, wz_wheel=0.0, wz_gyro=0.0)
    assert legacy.slipping()


def test_driving_rover_is_not_flagged():
    """A driving rover with no commanded turn is never flagged.

    This holds regardless of how much the terrain tilts the body as it goes
    - signature 1 needs a commanded rotation to disagree with (there is none
    here, wz_wheel=0.0) and signature 2 is retired (see
    test_symmetric_wedge_without_commanded_turn_is_a_known_undetected_gap).
    Kept at the historically hardest attitude-span case in the recording
    (0.002 rad/s over 15 s = 0.030 rad, the smallest genuine-driving span
    measured across 4,700 windows) as a regression marker, even though that
    margin no longer decides the outcome.
    """
    detector = SlipDetector()
    feed(detector, 20.0, vx=0.2, wz_wheel=0.0, wz_gyro=0.01, attitude_rate=0.002)
    assert not detector.slipping()


def test_stationary_rover_with_still_wheels_is_not_flagged():
    """Parked is not slipping: the wheels are not claiming anything."""
    detector = SlipDetector()
    feed(detector, 20.0, vx=0.0, wz_wheel=0.0, wz_gyro=0.0)
    assert not detector.slipping()


def test_rotating_in_place_is_not_flagged():
    """Turning on the spot: no forward claim, and the gyro confirms the yaw."""
    detector = SlipDetector()
    steps = 200
    for i in range(1, steps + 1):
        t = i / 10.0
        detector.add(t, 0.0, 0.4, 0.4, (0.0, 0.0, 0.4 * t))
    assert not detector.slipping()


def test_wheels_claim_rotation_the_gyro_does_not_see():
    """Wheels spinning differentially against a pinned chassis: claimed yaw rate with no measured yaw rate is slip even though nothing tilts."""
    detector = SlipDetector()
    feed(detector, 20.0, vx=0.2, wz_wheel=0.4, wz_gyro=0.0)
    assert detector.slipping()


def test_short_history_never_declares_slip():
    """A window that is not yet full says nothing rather than guessing."""
    detector = SlipDetector()
    feed(detector, 2.0, vx=0.2, wz_wheel=0.0, wz_gyro=0.0)
    assert detector.features() is None
    assert not detector.slipping()


def test_yaw_wrap_is_not_mistaken_for_a_large_attitude_change():
    """Crossing +-pi must not look like a 6 rad attitude change.

    The detector unwraps yaw. `attitude_span_rad` is the value
    `_body_is_rigid` (retired - see
    test_symmetric_wedge_without_commanded_turn_is_a_known_undetected_gap)
    used to act on; it is asserted directly here rather than through
    `slipping()`, since `slipping()` no longer depends on it at all
    (wz_wheel=0.0 keeps signature 1 out of this too).
    """
    detector = SlipDetector()
    for i in range(1, 201):
        t = i / 10.0
        # Sitting still at a heading right on the wrap point, dithering across it.
        yaw = math.pi - 0.001 if i % 2 else -math.pi + 0.001
        detector.add(t, 0.2, 0.0, 0.0, (0.0, 0.0, yaw))
    features = detector.features()
    assert features["attitude_span_rad"] < 0.01


@pytest.mark.parametrize(
    "observed_fraction,expected_slip",
    [
        (0.08, True),  # bottom of the measured slipping band
        (0.124, True),  # top of it
        (0.169, False),  # bottom of the measured honest-driving band
        (0.85, False),  # median honest driving
    ],
)
def test_measured_rotation_bands(observed_fraction, expected_slip):
    """The threshold sits in the gap between two bands measured on a real run: slipping windows corroborated 8.2-12.4% of the wheels' claimed rotation, honest driving 16.9-173%. Both edges are pinned here so a future tweak to the threshold has to admit it is moving out of the measured gap."""
    detector = SlipDetector()
    wheel_rate = 0.4
    for i in range(1, 201):
        t = i / 10.0
        detector.add(t, 0.2, wheel_rate, wheel_rate * observed_fraction, (0.05, 0.02, 0.1 * t))
    assert detector.slipping() is expected_slip


def test_slip_is_released_only_with_margin():
    """Hysteresis: what declares slip (<= 0.145) must not immediately release at 0.15, or the ZUPT flickers on and off around the boundary."""
    detector = SlipDetector()
    for i in range(1, 201):
        detector.add(i / 10.0, 0.2, 0.4, 0.4 * 0.10, (0.05, 0.02, 0.01 * i))
    assert detector.slipping()
    assert not detector.clearing()

    for i in range(201, 401):  # gyro now corroborates well past the release ratio
        detector.add(i / 10.0, 0.2, 0.4, 0.4 * 0.60, (0.05, 0.02, 0.024 * i))
    assert detector.clearing()


def test_release_uses_recent_evidence_not_the_whole_window():
    """Releasing on the full 15 s window would hold the zero-velocity update for 15 s after the rover breaks free, because the window still contains the wedge - and suppressing a really-moving rover's velocity is the same corruption the ZUPT exists to prevent, in the other direction."""
    detector = SlipDetector()
    for i in range(1, 201):  # 20 s wedged: wheels turning, gyro sees almost none
        detector.add(i / 10.0, 0.2, 0.4, 0.4 * 0.10, (0.05, 0.02, 0.004 * i))
    assert detector.slipping()
    assert not detector.clearing()

    # 6 s of genuinely corroborated rotation - less than half the declaration
    # window, so only a recent-slice release can see it.
    for i in range(201, 261):
        detector.add(i / 10.0, 0.2, 0.4, 0.4 * 0.90, (0.05, 0.02, 0.036 * i))
    assert detector.clearing()


def test_quiet_wheels_do_not_release_the_gate():
    """Regression: observed live as the gate flickering at the /odom message rate - 24 declare/clear pairs in two seconds.

    The rover was wedged (the 15 s window saw the disagreement, so slipping()
    stayed true) but had just been commanded to stop, so the most recent
    seconds claimed almost nothing and the release test read that absence of
    evidence as evidence of recovery. Releasing on quiet wheels is pointless
    anyway: with nothing being claimed there is nothing for the gate to
    suppress, so the safe answer is to stay latched.
    """
    detector = SlipDetector()
    for i in range(1, 151):  # 15 s wedged, wheels spinning, gyro sees ~10%
        detector.add(i / 10.0, 0.2, 0.4, 0.04, (0.05, 0.02, 0.004 * i))
    assert detector.slipping()

    for i in range(151, 201):  # 5 s stopped: wheels claim nothing at all
        detector.add(i / 10.0, 0.0, 0.0, 0.0, (0.05, 0.02, 0.604))
    assert detector.slipping(), "the wedge is still visible over the full window"
    assert not detector.clearing(), "must not release on quiet wheels"


def test_release_needs_positive_evidence_not_just_a_gap_in_data():
    """Immediately after declaring slip there is no recent evidence either way; the safe answer is to keep suppressing, not to release."""
    detector = SlipDetector()
    for i in range(1, 201):
        detector.add(i / 10.0, 0.2, 0.4, 0.4 * 0.10, (0.05, 0.02, 0.004 * i))
    assert detector.slipping()
    detector._samples.clear()  # no recent samples at all
    detector.add(20.1, 0.2, 0.4, 0.0, (0.05, 0.02, 0.08))
    assert not detector.clearing()


def test_wheels_claiming_little_rotation_cannot_trigger_the_rotation_test():
    """A ratio computed from almost no claimed rotation is noise, not evidence."""
    detector = SlipDetector()
    for i in range(1, 201):
        t = i / 10.0
        # 0.01 rad/s over 15 s = 0.15 rad claimed, under the 0.5 rad floor,
        # and the body is visibly tilting, so nothing should fire.
        detector.add(t, 0.2, 0.01, 0.0, (0.004 * t, 0.0, 0.0))
    assert not detector.slipping()


@pytest.mark.parametrize("window_s", [3.0, 15.0])
def test_window_length_no_longer_changes_the_no_commanded_turn_verdict(window_s):
    """The window length no longer changes the no-commanded-turn verdict.

    This used to be the measured reason the window is 15 s and not 3 s: a
    rover driving straight across smooth ground tilts slowly, so a short
    window's attitude span looked "rigid" (signature 2) when a long
    window's didn't. That was entirely a signature-2 argument, and
    signature 2 is retired (see
    test_symmetric_wedge_without_commanded_turn_is_a_known_undetected_gap)
    - with no commanded turn (wz_wheel=0.0) neither signature can fire
    regardless of window length, which this pins directly. The window
    still matters for signature 1 (needs time to accumulate a stable
    rotation ratio - see the module docstring), just not for this case.
    """
    smooth_drive = SlipDetector(window_s=window_s)
    feed(smooth_drive, 25.0, vx=0.2, wz_wheel=0.0, wz_gyro=0.0006, attitude_rate=0.002)
    assert not smooth_drive.slipping()


def test_out_of_order_samples_do_not_crash_the_node():
    """A timestamp that goes backwards must not take the whole launch down.

    This is a regression test for a real failure, not a hypothetical: an M4
    acceptance run died 30 s in with `ValueError: math domain error` from
    `math.sqrt(gyro_sq / span_s)`. features() integrates over consecutive pairs
    with `dt = t1 - t0`, so a single out-of-order sample makes dt negative and
    drives the running sum of SQUARES below zero. This node is launched as
    required, so its crash shut down gazebo, the EKF and the whole run with it -
    and the harness recorded a rover that had travelled 0 m.

    The trigger was load: adding the depth camera and the VO node changed message
    timing enough for /odom and /imu stamps to arrive out of order. The bug was
    always there; it just had not been provoked before.

    The window below is built so that the backwards sample genuinely drives the
    sum negative rather than merely denting it: the run is quiet (gyro 0, so it
    contributes nothing to the sum of squares) except for one lively sample
    immediately before the jump, whose gyro^2 is then multiplied by a negative
    dt. Feeding a uniformly noisy window instead would leave the total positive,
    and the test would pass with or without the fix - which is exactly what the
    first version of this test did.
    """
    detector = SlipDetector()
    feed(detector, 20.0, vx=0.2, wz_wheel=0.0, wz_gyro=0.0)

    latest = detector._samples[-1][0]
    detector.add(latest + 0.1, 0.2, 0.0, 3.0, (0.0, 0.0, 0.0))  # one lively sample
    detector.add(latest - 3.0, 0.2, 0.0, 3.0, (0.0, 0.0, 0.0))  # then 3 s into the past

    features = detector.features()
    assert features is not None
    assert features["gyro_rms_rps"] >= 0.0
    assert math.isfinite(features["gyro_rms_rps"])
    assert detector.dropped_out_of_order == 1
    # And the detector still works afterwards - the guard drops the sample, not the window.
    assert detector.slipping() in (True, False)


def test_duplicate_timestamps_are_dropped():
    """Two samples with the same stamp give dt = 0 and contribute nothing."""
    detector = SlipDetector()
    feed(detector, 20.0, vx=0.2, wz_wheel=0.0, wz_gyro=0.1)

    before = len(detector._samples)
    latest = detector._samples[-1][0]
    detector.add(latest, 0.2, 0.0, 0.1, (0.0, 0.0, 0.0))

    assert len(detector._samples) == before
    assert detector.dropped_out_of_order == 1
