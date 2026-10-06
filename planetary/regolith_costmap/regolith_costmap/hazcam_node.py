# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Onboard hazard camera: depth images in, obstacle points in odom out (gap G10).

The geometry is in hazcam.py and is tested there on analytically rendered depth
images. This file is the ROS shell: decode the depth image, get the camera mount
and the rover pose from TF and the tilt from the IMU, call the detector, publish
obstacle cell centres on /hazcam/obstacles for costmap_node.

WHERE EACH PIECE OF POSE COMES FROM, AND WHY.
  * camera_link in base_link: TF, read once (a fixed joint), as visual odometry
    does - moving the camera in the xacro cannot leave this node behind.
  * Roll and pitch: /imu. The EKF runs in two_d_mode, so odom -> base_link
    carries no tilt at all; on a 3 deg slope that is a 3 deg error in every ray.
    The IMU's attitude (biased by about 0.35 deg in RC-1) is the best tilt the
    rover has. The step test barely depends on it anyway (hazcam.py).
  * x, y, yaw: TF odom -> base_link at the image's stamp - the EKF, the same
    estimate the planner routes in, so a detection lands where the planner
    thinks the rock is even when the estimate has drifted from the truth.

It never reads ground truth (RC-1 oracle rule, docs/TRL4_BASELINE.md).

RATE. Frames are processed at most once per `period_s` of sim time (default
0.5 s, i.e. every 5th depth frame). About 11 ms of CPU per frame measured on
this machine (test_hazcam_geometry), so 2 frames per sim second at 0.1-0.6x real
time costs 0.2-1.3% of a core.
"""

import math
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2
import tf2_ros

from regolith_costmap.hazcam import back_project
from regolith_costmap.hazcam import build_elevation_grid
from regolith_costmap.hazcam import depth_image_to_array
from regolith_costmap.hazcam import detect_steps
from regolith_costmap.hazcam import ground_edge_bias
from regolith_costmap.hazcam import HazcamConfig
from regolith_costmap.hazcam import level_to_odom
from regolith_costmap.hazcam import optical_to_level
from regolith_costmap.hazcam import quaternion_to_matrix
from regolith_costmap.hazcam import roll_pitch_yaw
from regolith_costmap.hazcam_cloud import make_cloud


def _stamp_s(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class HazcamNode(Node):
    """Depth frames in, /hazcam/obstacles (PointCloud2 in odom) out."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__("hazcam_node")
        defaults = HazcamConfig()
        self.declare_parameter("depth_topic", "/camera/depth")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("output_topic", "/hazcam/obstacles")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("period_s", 0.5)
        # A frame whose pose TF cannot supply within this long is dropped: placing
        # it with a later pose would put its obstacles in the wrong place.
        self.declare_parameter("tf_wait_s", 1.0)
        self.declare_parameter("imu_max_age_s", 1.0)
        for name in ("stride_u", "stride_v", "min_depth_m", "max_range_m", "cell_m",
                     "step_height_m", "step_radius_cells", "min_points_obstacle",
                     "min_points_ground"):
            self.declare_parameter(name, getattr(defaults, name))
        self.cfg = HazcamConfig(
            stride_u=int(self.get_parameter("stride_u").value),
            stride_v=int(self.get_parameter("stride_v").value),
            min_depth_m=float(self.get_parameter("min_depth_m").value),
            max_range_m=float(self.get_parameter("max_range_m").value),
            cell_m=float(self.get_parameter("cell_m").value),
            step_height_m=float(self.get_parameter("step_height_m").value),
            step_radius_cells=int(self.get_parameter("step_radius_cells").value),
            min_points_obstacle=int(self.get_parameter("min_points_obstacle").value),
            min_points_ground=int(self.get_parameter("min_points_ground").value),
        )
        self.period_s = float(self.get_parameter("period_s").value)
        self.tf_wait_s = float(self.get_parameter("tf_wait_s").value)
        self.imu_max_age_s = float(self.get_parameter("imu_max_age_s").value)
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value

        self.k_matrix = None
        self.mount = None
        self.tilt = None  # (roll, pitch, stamp_s) from the IMU
        self.pending = None  # newest depth frame waiting for its TF
        self.last_processed_s = None
        self.counts = {"processed": 0, "with_obstacles": 0, "points_published": 0}
        self.dropped = {}
        self.cpu_s = 0.0
        self.edge_bias = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(
            PointCloud2, self.get_parameter("output_topic").value, 10)
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value, self._on_camera_info, 10)
        self.create_subscription(Imu, self.get_parameter("imu_topic").value, self._on_imu, 20)
        # Best effort, as the bridge publishes: a RELIABLE subscriber never matches it
        # and the node would sit silent with no error (found by visual odometry first).
        self.create_subscription(
            Image, self.get_parameter("depth_topic").value, self._on_depth,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(0.1, self._try_pending)
        self.create_timer(30.0, self._report)
        self.get_logger().info(
            f"Hazcam started: step >= {self.cfg.step_height_m:.2f} m within "
            f"{self.cfg.step_radius_cells * self.cfg.cell_m:.2f} m, out to "
            f"{self.cfg.max_range_m:.1f} m, every {self.period_s:.2f} s of sim time. "
            "Onboard depth, IMU and EKF only; never ground truth.")

    def _drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self.k_matrix is None:
            self.k_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"Camera intrinsics received (fx={self.k_matrix[0, 0]:.1f} px).")

    def _on_imu(self, msg: Imu) -> None:
        q = msg.orientation
        if q.x == 0.0 and q.y == 0.0 and q.z == 0.0 and q.w == 0.0:
            return  # no orientation in this message
        roll, pitch, _ = roll_pitch_yaw(q.x, q.y, q.z, q.w)
        self.tilt = (roll, pitch, _stamp_s(msg.header.stamp))

    def _lookup_mount(self) -> bool:
        if self.mount is not None:
            return True
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, Time()).transform
        except tf2_ros.TransformException:
            return False
        r = tf.rotation
        self.mount = (quaternion_to_matrix(r.x, r.y, r.z, r.w),
                      np.array([tf.translation.x, tf.translation.y, tf.translation.z]))
        self.get_logger().info(
            f"Camera mount from TF: offset {np.round(self.mount[1], 3).tolist()} m in "
            f"{self.base_frame}.")
        return True

    def _on_depth(self, msg: Image) -> None:
        stamp_s = _stamp_s(msg.header.stamp)
        if self.last_processed_s is not None and stamp_s - self.last_processed_s < self.period_s:
            return
        if self.pending is not None:
            self._drop("superseded while waiting for TF")
        self.pending = msg
        self._try_pending()

    def _try_pending(self) -> None:
        msg = self.pending
        if msg is None:
            return
        if self.k_matrix is None or not self._lookup_mount():
            self.pending = None
            self._drop("no intrinsics or camera mount yet")
            return
        stamp = Time.from_msg(msg.header.stamp)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, stamp, timeout=Duration(seconds=0.0))
        except tf2_ros.ExtrapolationException:
            # The EKF has not published this far yet: keep the frame for a later tick,
            # unless sim time has moved on long past it.
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if now_s - _stamp_s(msg.header.stamp) > self.tf_wait_s:
                self.pending = None
                self._drop("no odom pose at the image stamp")
            return
        except tf2_ros.TransformException:
            self.pending = None
            self._drop("odom -> base_link unavailable")
            return
        self.pending = None
        self._process(msg, tf.transform)

    def _process(self, msg: Image, odom_base) -> None:
        started = time.thread_time()
        try:
            depth = depth_image_to_array(msg.encoding, msg.height, msg.width, msg.step,
                                         bool(msg.is_bigendian), bytes(msg.data))
        except ValueError as error:
            self._drop(str(error))
            return
        stamp_s = _stamp_s(msg.header.stamp)
        q = odom_base.rotation
        tf_roll, tf_pitch, yaw = roll_pitch_yaw(q.x, q.y, q.z, q.w)
        if self.tilt is not None and abs(stamp_s - self.tilt[2]) <= self.imu_max_age_s:
            roll, pitch = self.tilt[0], self.tilt[1]
        else:
            # No recent IMU: fall back to TF's tilt (zero under two_d_mode). The step
            # test tolerates it on gentle ground, but it is counted, not hidden.
            roll, pitch = tf_roll, tf_pitch
            self._drop("no recent IMU tilt (used TF tilt; frame still processed)")

        rotation, offset = self.mount
        points = optical_to_level(back_project(depth, self.k_matrix, self.cfg),
                                  rotation, offset, roll, pitch)
        grid = build_elevation_grid(points, self.cfg)
        obstacle, step = detect_steps(grid, self.cfg)
        bias = ground_edge_bias(points)
        if math.isfinite(bias):
            self.edge_bias.append(bias)
            self.edge_bias = self.edge_bias[-200:]

        self.last_processed_s = stamp_s
        self.counts["processed"] += 1
        ix, iy = np.nonzero(obstacle)
        if len(ix):
            xy_level = np.column_stack([grid.x_min + (ix + 0.5) * grid.cell_m,
                                        grid.y_min + (iy + 0.5) * grid.cell_m])
            xy_odom = level_to_odom(xy_level, odom_base.translation.x, odom_base.translation.y,
                                    yaw)
            self.publisher.publish(
                make_cloud(xy_odom, step[ix, iy], self.odom_frame, msg.header.stamp))
            self.counts["with_obstacles"] += 1
            self.counts["points_published"] += len(ix)
        self.cpu_s += time.thread_time() - started

    def _report(self) -> None:
        n = self.counts["processed"]
        if n == 0:
            self.get_logger().warn(
                "Hazcam has processed no frames - check /camera/depth (nav_cameras:=true), "
                "/camera/camera_info and the odom -> base_link TF. Dropped: "
                f"{self.dropped or 'none'}")
            return
        bias = (f"{np.median(self.edge_bias):+.3f} m" if self.edge_bias else "n/a")
        self.get_logger().info(
            f"Hazcam: {n} frames, {self.counts['with_obstacles']} with obstacles, "
            f"{self.counts['points_published']} obstacle cells published, "
            f"{1e3 * self.cpu_s / n:.1f} ms CPU per frame. Near-ground edge-minus-centre "
            f"height {bias} (planar depth reads ~0; ray-length depth would read about "
            f"-0.03). Dropped/flagged: {self.dropped or 'none'}")


def main() -> None:
    """Console entry point."""
    rclpy.init()
    node = HazcamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
