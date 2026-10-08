# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""choose_excursion_point - where the rover goes to re-localise before it claims an
arrival on low confidence."""

import math

from regolith_vehicle_interface.pure_pursuit_node import _same_xy, choose_excursion_point


def free(x, y):
    return 0.0


def test_straight_ahead_when_clear():
    x, y = choose_excursion_point((1.0, 2.0), math.radians(90), 8.0, free)
    assert abs(x - 1.0) < 1e-9 and abs(y - 10.0) < 1e-9


def test_turns_away_from_a_lethal_cell_ahead():
    def wall_ahead(x, y):
        return 100.0 if y > 5.0 else 0.0  # everything "north" is lethal
    x, y = choose_excursion_point((0.0, 0.0), math.radians(90), 8.0, wall_ahead)
    assert y <= 5.0  # it found a bearing that is not into the wall


def test_none_when_boxed_in():
    assert choose_excursion_point((0.0, 0.0), 0.0, 8.0, lambda x, y: 100.0) is None


def test_same_xy_tolerance():
    assert _same_xy((1.0, 2.0), (1.0004, 1.9996))
    assert not _same_xy((1.0, 2.0), (1.01, 2.0))
