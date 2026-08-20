# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""The final approach must terminate, and must not terminate early.

`goal_tolerance_m` was 1.0 m because a pure pursuit follower told to stop
closer than it can steer will orbit its goal instead of arriving. That cost
M4 two thirds of its 1.5 m bar before drift (PROGRESS.md, "Where M4 actually
stands now"), so the tolerance is being tightened - and the reason it was
generous has to be answered by construction, not by hoping.

Two properties, pulling in opposite directions, are what these tests pin:

  * it ALWAYS terminates, however the rover moves, and within a bound that
    can be written down; and
  * it does NOT terminate early - a rover still genuinely closing on its goal
    is never stopped short, including while creeping the last half metre or
    being carried away and back by a recovery maneuver.

`TerminalApproach` takes sim-time seconds as an argument rather than reading a
clock, which is what lets an orbit be simulated here at full length instead of
being waited out in a real run.
"""

import math

from regolith_vehicle_interface.pure_pursuit_node import APPROACHING
from regolith_vehicle_interface.pure_pursuit_node import ARRIVED
from regolith_vehicle_interface.pure_pursuit_node import CLOSEST_APPROACH
from regolith_vehicle_interface.pure_pursuit_node import FAR
from regolith_vehicle_interface.pure_pursuit_node import TerminalApproach

# The shipped settings.
RADIUS_M = 3.0
TOLERANCE_M = 0.35
GIVEBACK_M = 0.5
PATIENCE_S = 15.0
STEP_S = 0.1  # control_period_s
SPEED_MPS = 0.2  # base_speed_mps


def approach() -> TerminalApproach:
    return TerminalApproach(RADIUS_M, TOLERANCE_M, GIVEBACK_M, PATIENCE_S)


def test_a_straight_creep_arrives_and_is_never_cut_short():
    """The ordinary case: drive at the goal until inside tolerance."""
    ta = approach()
    distance = RADIUS_M - 0.01
    t = 0.0
    while distance > TOLERANCE_M:
        outcome = ta.update(distance, t)
        assert outcome == APPROACHING, f"stopped short at {distance:.2f} m"
        distance -= SPEED_MPS * STEP_S
        t += STEP_S
    assert ta.update(distance, t) == ARRIVED


def test_a_slow_creep_is_not_cut_short_by_patience():
    """Crawling at a tenth of cruise still counts as progress: 0.02 m/s clears the 0.05 m improvement threshold well inside the 15 s patience window."""
    ta = approach()
    distance, t = 2.0, 0.0
    while distance > TOLERANCE_M:
        assert ta.update(distance, t) == APPROACHING, f"stopped short at {distance:.2f} m"
        distance -= 0.02 * STEP_S
        t += STEP_S
    assert ta.update(distance, t) == ARRIVED


def test_an_orbit_terminates_instead_of_circling_forever():
    """The failure the 1.0 m tolerance was avoiding: a rover that circles the goal at a distance it never improves on. It must stop, not lap."""
    ta = approach()
    t = 0.0
    for _ in range(int(600 / STEP_S)):  # ten minutes of sim time
        # Constant-radius orbit: distance never improves, never recedes much.
        outcome = ta.update(1.2, t)
        if outcome == CLOSEST_APPROACH:
            assert t <= PATIENCE_S + STEP_S, f"took {t:.1f} s to give up on an orbit"
            assert ta.closest_m == 1.2
            return
        t += STEP_S
    raise AssertionError("orbited for ten minutes without terminating")


def test_an_asymptotic_creep_that_never_arrives_still_terminates():
    """The case `improvement_m` exists for, and the one a "did it get closer?" test would wave through: every step DOES get closer, by less and less, converging on 0.90 m - a goal the rover approaches forever and never reaches. Losing traction on the last stretch looks exactly like this, and this world has produced a not-root-caused traction failure on benign ground already (PROGRESS.md, "Retraction"). Improvement has to be worth the clock it renews, or the rover creeps at the goal until the mission budget runs out."""
    ta = approach()
    t, limit_m, gap_m = 0.0, 0.90, 0.60
    for _ in range(int(600 / STEP_S)):
        if ta.update(limit_m + gap_m, t) == CLOSEST_APPROACH:
            # Not quick - each 0.05 m of ground bought another 15 s window, and
            # the rover kept paying until a window ran out, ~150 s in. That is
            # the honest cost of crediting real progress: bounded, a twelfth of
            # the 1800 s mission budget, and still finite where a "did it get
            # closer at all?" rule would creep here forever.
            assert t < 300.0, f"took {t:.0f} s to give up on a creep"
            assert ta.closest_m > TOLERANCE_M  # it genuinely never arrived
            return
        gap_m *= 0.999  # always closer, never enough
        t += STEP_S
    raise AssertionError("an approach that never arrives never terminated")


def test_estimator_jitter_at_a_standstill_terminates():
    """The same clock, against a stalled rover whose position estimate wobbles by a couple of centimetres."""
    ta = approach()
    t = 0.0
    jitter = [0.0, -0.02, 0.01, -0.015, 0.02, -0.01]  # deterministic, |e| <= 2 cm
    for i in range(int(600 / STEP_S)):
        if ta.update(1.2 + jitter[i % len(jitter)], t) == CLOSEST_APPROACH:
            assert t <= 2 * PATIENCE_S, f"took {t:.1f} s to give up on a jittering stall"
            return
        t += STEP_S
    raise AssertionError("jitter renewed the patience clock indefinitely")


def test_being_pushed_away_from_the_goal_stops_at_the_closest_approach():
    """Receding is decided against the best distance ACHIEVED, not the distance on entry - a rover that got to 0.6 m and is now at 1.2 m has been carried away from an arrival it nearly had."""
    ta = approach()
    t = 0.0
    for distance in [2.0, 1.5, 1.0, 0.6]:
        assert ta.update(distance, t) == APPROACHING
        t += STEP_S
    assert ta.closest_m == 0.6
    assert ta.update(0.6 + GIVEBACK_M - 0.01, t) == APPROACHING  # just inside
    assert ta.update(0.6 + GIVEBACK_M + 0.01, t) == CLOSEST_APPROACH


def test_termination_is_bounded_for_any_movement_whatsoever():
    """The guarantee, against adversarial motion rather than a plausible path. Each patience reset costs 0.05 m of real improvement and distance cannot go below the tolerance, so the approach cannot outlast (radius - tolerance) / improvement windows however the rover moves."""
    ta = approach()
    bound_s = (RADIUS_M - TOLERANCE_M) / ta.improvement_m * (PATIENCE_S + STEP_S)
    # Worst case: improve by just enough to renew the clock, as late as possible.
    distance, t = RADIUS_M - 0.01, 0.0
    for _ in range(int(bound_s / STEP_S) + 100):
        outcome = ta.update(distance, t)
        if outcome in (ARRIVED, CLOSEST_APPROACH):
            assert t <= bound_s, f"terminated at {t:.0f} s, past the {bound_s:.0f} s bound"
            return
        if outcome == APPROACHING and math.isclose(t % (PATIENCE_S - STEP_S), 0.0, abs_tol=1e-9):
            distance -= ta.improvement_m + 1e-6  # the cheapest legal renewal
        t += STEP_S
    raise AssertionError("never terminated")


def test_leaving_the_radius_forgets_the_approach():
    """A rover that drives back out - a detour around a late obstacle, or a recovery maneuver - gets a clean approach when it returns, rather than being judged against a closest approach it made minutes ago."""
    ta = approach()
    assert ta.update(1.0, 0.0) == APPROACHING
    assert ta.closest_m == 1.0
    assert ta.update(RADIUS_M + 5.0, 1.0) == FAR
    assert ta.closest_m is None
    # Back inside, at a distance that would have read as "receded" before.
    assert ta.update(2.0, 100.0) == APPROACHING
    for t in range(101, 101 + int(PATIENCE_S)):
        assert ta.update(2.0, float(t)) == APPROACHING or t >= 100 + PATIENCE_S


def test_the_tolerance_is_what_decides_arrival():
    """The parameter under test actually moves the stopping point: what arrives at 1.0 m is still approaching at 0.35 m."""
    loose = TerminalApproach(RADIUS_M, 1.0, GIVEBACK_M, PATIENCE_S)
    tight = TerminalApproach(RADIUS_M, 0.35, GIVEBACK_M, PATIENCE_S)
    assert loose.update(0.9, 0.0) == ARRIVED
    assert tight.update(0.9, 0.0) == APPROACHING
    assert tight.update(0.3, 0.1) == ARRIVED


def test_closest_approach_is_never_worse_than_the_loose_tolerance_would_have_been():
    """The safety argument for tightening: a CLOSEST_APPROACH stop can only happen after the rover has already been closer than it is now, so the fallback cannot leave it further out than the old 1.0 m stop would have."""
    ta = approach()
    t = 0.0
    for distance in [2.5, 1.8, 0.9, 0.55]:
        ta.update(distance, t)
        t += STEP_S
    while True:
        outcome = ta.update(0.55, t)
        if outcome == CLOSEST_APPROACH:
            break
        t += STEP_S
        assert t < 600, "did not terminate"
    assert ta.closest_m <= 1.0
