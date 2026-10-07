# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""flip_recovery_node.escape_tilt_exceeded - the escape maneuver's tilt limit."""

import importlib.util
import math
from pathlib import Path
import sys

NODE = Path(__file__).resolve().parent.parent / "scripts/flip_recovery_node.py"
spec = importlib.util.spec_from_file_location("flip_recovery_node_tilt", NODE)
mod = importlib.util.module_from_spec(spec)
sys.modules["flip_recovery_node_tilt"] = mod
spec.loader.exec_module(mod)
LIMIT = math.radians(25.0)


def test_level_and_moderate_tilt_continue():
    assert not mod.escape_tilt_exceeded(0.0, 0.0, LIMIT)
    assert not mod.escape_tilt_exceeded(math.radians(15), math.radians(-20), LIMIT)


def test_the_recorded_flip_trajectory_is_stopped_before_it_flips():
    # Roll/pitch climbing toward the recorded rollover (31 / 45 deg at the flip);
    # the guard must trip on the way up, well before it.
    assert mod.escape_tilt_exceeded(math.radians(24), math.radians(26), LIMIT)
    assert mod.escape_tilt_exceeded(math.radians(-27), 0.0, LIMIT)
