#!/usr/bin/env python3
# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Turns gz-sim's ideal IMU into one a real rover could carry.

gz-sim's IMU reports the simulator's own orientation, exactly - including an
absolute, drift-free yaw - and noiseless rates and accelerations. The EKF fuses
that orientation (config/ekf*.yaml, imu0_config), so every localisation result
in PROGRESS.md was measured with perfect heading, and terrain_relative_node
matched terrain slope against a perfect attitude. No lunar rover has either:
there is no usable magnetic field, so heading comes from an initial absolute fix
(sun sensor or star tracker) propagated by gyro integration, and tilt comes from
the accelerometer's view of gravity - which on the Moon is six times weaker, so
the same accelerometer bias produces six times the tilt error it would on Earth.
docs/TRL4_BASELINE.md, gap G3 and section 3.3.

With imu_error_model:=true, hello_moon.launch.py bridges the ideal IMU to
/sim/imu_ideal and this node publishes /imu, so every consumer (the EKF via
sensor_covariance_relay, wheel_slip_node, flip_recovery_node,
terrain_relative_node) reads the modelled sensor without being changed.

THE MODEL, all errors drawn once per run from imu_noise_seed:

  rates          true + per-axis residual bias N(0, gyro_bias_dph) + white
                 noise of density gyro_noise_density_dps_rthz
  acceleration   true + per-axis residual bias N(0, accel_bias_g) + white noise
                 of density accel_noise_density_g_rthz
  roll / pitch   true + the tilt an AHRS infers from the biased accelerometer
                 (bias / lunar g, constant) + small white noise. An AHRS keeps
                 re-levelling against gravity, so tilt error does not grow.
  yaw            true + initial heading error N(0, initial_heading_sigma_deg)
                 + the integral of the z-gyro's bias and white noise: it drifts
                 for the whole run and nothing corrects it.

"Residual" bias means after the pre-mission calibration any real rover does at
rest; the defaults are an industrial-grade MEMS unit (gyro 3 deg/h, accel 1 mg),
conservative against a tactical-grade one. This node is simulator-side - a
sensor model, like gz's own IMU - which is why it may read /sim/imu_ideal; the
acceptance harness's R5 audit allow-lists it by name and nothing else.
"""

import math
import random

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

LUNAR_G = 1.62
STANDARD_G = 9.80665


def quat_multiply(a, b):
    """Hamilton product of (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class ImuErrorModel:
    """The error model, with no ROS in it, so it can be tested directly."""

    def __init__(self, seed: int, gyro_bias_dph=3.0, gyro_noise_density_dps_rthz=0.0025,
                 accel_bias_g=0.001, accel_noise_density_g_rthz=0.0001,
                 initial_heading_sigma_deg=0.5, tilt_noise_deg=0.02, gravity=LUNAR_G):
        rng = random.Random(seed)
        self._rng = rng
        deg = math.radians
        self.gyro_bias = [rng.gauss(0.0, deg(gyro_bias_dph) / 3600.0) for _ in range(3)]
        self.accel_bias = [rng.gauss(0.0, accel_bias_g * STANDARD_G) for _ in range(3)]
        self.initial_heading_error = rng.gauss(0.0, deg(initial_heading_sigma_deg))
        self.gyro_density = deg(gyro_noise_density_dps_rthz)
        self.accel_density = accel_noise_density_g_rthz * STANDARD_G
        self.tilt_noise = deg(tilt_noise_deg)
        # Tilt an AHRS infers from a biased accelerometer at rest: the bias
        # reads as a component of gravity along that axis.
        self.roll_bias = math.atan2(self.accel_bias[1], gravity)
        self.pitch_bias = -math.atan2(self.accel_bias[0], gravity)
        self.yaw_error = self.initial_heading_error
        self._last_t = None

    def apply(self, t: float, q_true, gyro, accel):
        """Return (q, gyro, accel) as the modelled sensor reports them at sim time t."""
        dt = 0.0
        if self._last_t is not None and t > self._last_t:
            dt = min(t - self._last_t, 1.0)  # a stalled sim must not inject a huge step
        self._last_t = t if self._last_t is None or t > self._last_t else self._last_t
        rng = self._rng
        # White noise per sample: density / sqrt(dt), the discrete equivalent of a
        # continuous noise density at this sample interval.
        g_sigma = self.gyro_density / math.sqrt(dt) if dt > 0 else 0.0
        a_sigma = self.accel_density / math.sqrt(dt) if dt > 0 else 0.0
        gyro_out = [w + b + rng.gauss(0.0, g_sigma) for w, b in zip(gyro, self.gyro_bias)]
        accel_out = [a + b + rng.gauss(0.0, a_sigma) for a, b in zip(accel, self.accel_bias)]
        # Heading error: integral of the z-gyro error. Using the realised noise
        # sample keeps rate and heading errors consistent with each other.
        self.yaw_error += (gyro_out[2] - gyro[2]) * dt
        roll_err = self.roll_bias + rng.gauss(0.0, self.tilt_noise)
        pitch_err = self.pitch_bias + rng.gauss(0.0, self.tilt_noise)
        q = quat_multiply(quat_from_rpy(0.0, 0.0, self.yaw_error),
                          quat_multiply(q_true, quat_from_rpy(roll_err, pitch_err, 0.0)))
        return q, gyro_out, accel_out


class ImuErrorModelNode(Node):
    def __init__(self):
        super().__init__("regolith_imu_error_model")
        self.declare_parameter("imu_noise_seed", 0)
        self.declare_parameter("gyro_bias_dph", 3.0)
        self.declare_parameter("gyro_noise_density_dps_rthz", 0.0025)
        self.declare_parameter("accel_bias_g", 0.001)
        self.declare_parameter("accel_noise_density_g_rthz", 0.0001)
        self.declare_parameter("initial_heading_sigma_deg", 0.5)
        self.declare_parameter("tilt_noise_deg", 0.02)
        p = self.get_parameter
        self._model = ImuErrorModel(
            seed=int(p("imu_noise_seed").value),
            gyro_bias_dph=p("gyro_bias_dph").value,
            gyro_noise_density_dps_rthz=p("gyro_noise_density_dps_rthz").value,
            accel_bias_g=p("accel_bias_g").value,
            accel_noise_density_g_rthz=p("accel_noise_density_g_rthz").value,
            initial_heading_sigma_deg=p("initial_heading_sigma_deg").value,
            tilt_noise_deg=p("tilt_noise_deg").value,
        )
        m = self._model
        self.get_logger().info(
            "IMU ERROR MODEL (seed {}): gyro bias [{}] deg/h, accel bias [{}] mg, initial "
            "heading error {:.3f} deg, tilt bias roll {:.3f} / pitch {:.3f} deg".format(
                int(p("imu_noise_seed").value),
                ", ".join(f"{math.degrees(b) * 3600:.2f}" for b in m.gyro_bias),
                ", ".join(f"{b / STANDARD_G * 1000:.3f}" for b in m.accel_bias),
                math.degrees(m.initial_heading_error),
                math.degrees(m.roll_bias), math.degrees(m.pitch_bias)))
        self._pub = self.create_publisher(Imu, "/imu", 20)
        self.create_subscription(Imu, "/sim/imu_ideal", self._on_imu, 20)
        self._count = 0

    def _on_imu(self, msg: Imu) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        o, w, a = msg.orientation, msg.angular_velocity, msg.linear_acceleration
        q, gyro, accel = self._model.apply(t, (o.x, o.y, o.z, o.w), (w.x, w.y, w.z), (a.x, a.y, a.z))
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gyro
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = accel
        self._pub.publish(msg)
        self._count += 1
        if self._count % 6000 == 0:  # once a simulated minute at 100 Hz
            self.get_logger().info(
                f"IMU ERROR MODEL: heading error now {math.degrees(self._model.yaw_error):.3f} deg "
                f"at sim t={t:.0f}s")


def main() -> None:
    rclpy.init()
    node = ImuErrorModelNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
