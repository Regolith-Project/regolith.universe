# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Statistical tests for imu_error_model_node.ImuErrorModel.

Asserted on what the model OUTPUTS (the orientation it reports, compared with
the truth it was given), over many seeds, against the closed-form error the
parameters imply - not on the model's internal fields, which a bug in apply()
could leave correct while the published sensor is wrong.
"""

import importlib.util
import math
from pathlib import Path
import statistics
import sys

NODE = Path(__file__).resolve().parent.parent / "scripts/imu_error_model_node.py"
spec = importlib.util.spec_from_file_location("imu_error_model_node", NODE)
mod = importlib.util.module_from_spec(spec)
sys.modules["imu_error_model_node"] = mod
spec.loader.exec_module(mod)


def rpy(q):
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


TRUE_Q = mod.quat_from_rpy(0.02, -0.03, 1.1)


def run(seed, seconds, rate_hz=10.0, **kw):
    m = mod.ImuErrorModel(seed, **kw)
    out = None
    for i in range(int(seconds * rate_hz) + 1):
        out = m.apply(i / rate_hz, TRUE_Q, (0.0, 0.0, 0.0), (0.0, 0.0, mod.LUNAR_G))
    return out


def test_zero_parameters_are_the_identity():
    q, gyro, accel = run(3, 5.0, gyro_bias_dph=0, gyro_noise_density_dps_rthz=0, accel_bias_g=0,
                         accel_noise_density_g_rthz=0, initial_heading_sigma_deg=0, tilt_noise_deg=0)
    assert all(abs(a - b) < 1e-12 for a, b in zip(rpy(q), rpy(TRUE_Q)))
    assert gyro == [0.0, 0.0, 0.0] and accel == [0.0, 0.0, mod.LUNAR_G]


def test_same_seed_same_errors_different_seed_different():
    assert run(7, 3.0) == run(7, 3.0)
    assert run(7, 3.0) != run(8, 3.0)


def test_heading_error_after_600_s_matches_the_parameters():
    # sigma^2 = initial^2 + (bias * t)^2 + ARW^2 * t, with the defaults:
    # 0.5 deg, 3 deg/h over 600 s = 0.5 deg, 0.0025 deg/s/rtHz -> 0.061 deg.
    t = 600.0
    expected = math.sqrt(0.5 ** 2 + (3.0 / 3600 * t) ** 2 + 0.0025 ** 2 * t)
    errors = []
    for seed in range(300):
        q, _, _ = run(seed, t, rate_hz=5.0)
        errors.append(math.degrees(rpy(q)[2] - rpy(TRUE_Q)[2]))
    sd = statistics.pstdev(errors)
    assert 0.85 * expected < sd < 1.15 * expected, (sd, expected)
    assert abs(statistics.mean(errors)) < 0.15


def test_tilt_error_is_accel_bias_over_lunar_gravity():
    expected = math.degrees(math.atan2(0.001 * mod.STANDARD_G, mod.LUNAR_G))  # ~0.35 deg
    rolls = []
    for seed in range(400):
        q, _, _ = run(seed, 0.0)
        rolls.append(math.degrees(rpy(q)[0] - rpy(TRUE_Q)[0]))
    sd = statistics.pstdev(rolls)
    assert 0.85 * expected < sd < 1.15 * expected, (sd, expected)


def test_tilt_error_does_not_grow():
    q0, _, _ = run(11, 0.0)
    q1, _, _ = run(11, 600.0)
    # Same seed: roll differs only by the small white tilt noise, not by drift.
    assert abs(math.degrees(rpy(q1)[0] - rpy(q0)[0])) < 0.2
