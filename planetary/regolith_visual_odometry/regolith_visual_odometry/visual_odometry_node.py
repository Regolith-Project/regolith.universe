# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Publishes body-frame velocity from the rover's RGB + depth cameras.

The geometry lives in vo_core.py (and is tested there, without ROS). This file is
the thin ROS shell: subscribe, pair frames, call the estimator, publish.

WHAT IT PUBLISHES, AND WHY ONLY THAT. `/vo/odom` is a nav_msgs/Odometry carrying
TWIST ONLY - linear x/y and yaw rate in base_link. Its pose is left at identity
with an enormous covariance, and the EKF is configured to ignore it.

That is the whole design decision, so it is worth stating plainly. VO could
integrate its own pose and publish that instead, but a relative sensor's
integrated pose drifts, and handing the EKF two independently drifting position
estimates makes them fight. Fusing velocity keeps each sensor to what it
actually measures and lets the filter do the integrating once:

    wheel odometry   vx, vyaw          cannot represent vy at all
    IMU              orientation, wxyz cannot observe position error
    visual odometry  vy                <- the term nothing else could see

Only vy is fused, and that is a measured decision rather than a tidy one. Against
ground truth over 130 real frame pairs from this world, VO's lateral channel came
out unbiased (+0.000 +- 0.018 m/s) while its forward channel was biased 0.060 m/s
LOW - a 30% under-report at cruise - against gated wheel odometry's ~1%. vx is
still published, because it is worth plotting and because saying so is cheaper
than rediscovering it, but ekf.yaml does not fuse it: a filter that believes it
has travelled less than it has drives past its goal.

The vy slot is the entire point. A differential-drive model asserts vy = 0 by
construction, so ~10% of this rover's motion over boulder-strewn regolith was
invisible to the estimator and accumulated as an uncorrected random walk (see
vo_core.py's header and PROGRESS.md's M4 error budget).

IT NEVER TOUCHES GROUND TRUTH. Same rule as wheel_slip_node.py: this is a
localization input, and an acceptance number produced by a localization input
that consulted /ground_truth/pose would be meaningless. Its inputs are the two
cameras and nothing else.
"""

import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from regolith_visual_odometry.vo_core import VoConfig, estimate_motion

# Covariance robot_localization reads as "no information here". Large but finite:
# an inf or NaN anywhere in the matrix poisons the filter's own arithmetic.
IGNORE_COVARIANCE = 1e6

# The derived sigma describes the LATERAL channel well - measured against ground
# truth over 130 real frame pairs, vy error was +0.000 +- 0.018 m/s against a
# reported sigma of 0.020. The forward channel is a different story: it came out
# biased 0.060 m/s low, a 30% under-report at cruise, which no amount of variance
# describes honestly because it is a bias and not noise. vx is published because
# it is worth plotting, and inflated here so that anything which does consume it
# weights it by something closer to its real error than to vy's.
# ekf.yaml deliberately fuses vy only.
FORWARD_SIGMA_INFLATION = 4.0


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Kept local rather than pulling in a transforms library for one function."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class VisualOdometryNode(Node):
    def __init__(self) -> None:
        super().__init__("visual_odometry_node")

        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("depth_topic", "/camera/depth")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("output_topic", "/vo/odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_link")
        # Estimate against a keyframe ~0.4 s old rather than the previous frame:
        # adjacent frames carry about 1 px of flow against 0.74 px of tracker
        # noise, which measures nothing. See vo_core.py's header for the sweep.
        self.declare_parameter("min_baseline_s", 0.4)
        self.declare_parameter("max_baseline_s", 1.5)
        self.declare_parameter("min_inliers", VoConfig.min_inliers)
        self.declare_parameter("max_depth_m", VoConfig.max_depth_m)

        self.cfg = VoConfig(
            min_inliers=int(self.get_parameter("min_inliers").value),
            max_depth_m=float(self.get_parameter("max_depth_m").value),
        )
        self.min_baseline_s = float(self.get_parameter("min_baseline_s").value)
        self.max_baseline_s = float(self.get_parameter("max_baseline_s").value)
        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value

        self.bridge = CvBridge()
        self.k_matrix: np.ndarray | None = None
        self.mount: tuple[np.ndarray, np.ndarray] | None = None
        self.keyframe: tuple[np.ndarray, np.ndarray, float] | None = None
        self.n_published = 0
        self.n_refused = 0
        # Counted per reason: "VO refused 39 pairs" is not actionable, "39 pairs
        # refused, all for too few features with usable depth" points straight at
        # the depth topic. Found the hard way on the first live run.
        self.refusals: dict[str, int] = {}
        self.last_inliers = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.publisher = self.create_publisher(Odometry, self.get_parameter("output_topic").value, 10)
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value, self._on_camera_info, 10
        )

        # Sensor-data QoS: the bridge publishes best-effort, and a RELIABLE
        # subscriber simply never matches it - the node would sit silent with no
        # error anywhere. Depth is 10 Hz against the RGB camera's 30, so the slop
        # has to admit up to about half a depth period.
        image_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sync = ApproximateTimeSynchronizer(
            [
                Subscriber(self, Image, self.get_parameter("image_topic").value, qos_profile=image_qos),
                Subscriber(self, Image, self.get_parameter("depth_topic").value, qos_profile=image_qos),
            ],
            queue_size=10,
            slop=0.05,
        )
        self.sync.registerCallback(self._on_frame_pair)

        self.create_timer(10.0, self._report)
        self.get_logger().info(
            "Visual odometry started - publishing body-frame velocity (vx, vy, vyaw) on "
            f"{self.get_parameter('output_topic').value}. Onboard cameras only; never ground truth."
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self.k_matrix is None:
            self.k_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"Camera intrinsics received (fx={self.k_matrix[0, 0]:.1f} px).")

    def _lookup_mount(self) -> bool:
        """base_link <- camera_link, from TF rather than hardcoded URDF numbers.

        Read once and cached; it is a fixed joint. Taking it from TF means moving
        the camera in the xacro cannot silently leave this node computing motion
        for a camera that is no longer where it thinks it is.
        """
        if self.mount is not None:
            return True
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, rclpy.time.Time()
            ).transform
        except tf2_ros.TransformException:
            return False
        rotation = quaternion_to_rotation_matrix(
            tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w
        )
        offset = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
        self.mount = (rotation, offset)
        self.get_logger().info(
            f"Camera mount from TF: offset {np.round(offset, 3).tolist()} m in {self.base_frame}."
        )
        return True

    def _on_frame_pair(self, image_msg: Image, depth_msg: Image) -> None:
        if self.k_matrix is None or not self._lookup_mount():
            return

        gray = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="mono8")
        depth = np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1"), dtype=np.float32)
        stamp_s = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9

        if self.keyframe is None:
            self.keyframe = (gray, depth, stamp_s)
            return

        key_gray, key_depth, key_stamp_s = self.keyframe
        dt_s = stamp_s - key_stamp_s
        if dt_s < self.min_baseline_s:
            # Not enough parallax yet - keep accumulating against this keyframe.
            return
        if dt_s > self.max_baseline_s:
            # Sim time jumped, or frames stopped arriving. Motion over an unknown
            # interval is not a measurement; start a fresh keyframe instead.
            self.keyframe = (gray, depth, stamp_s)
            return

        rotation, offset = self.mount
        estimate = estimate_motion(
            key_gray, key_depth, gray, self.k_matrix, dt_s, rotation, offset, self.cfg
        )
        self._publish(estimate, image_msg.header.stamp)
        self.keyframe = (gray, depth, stamp_s)

    def _publish(self, estimate, stamp) -> None:
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "odom"
        msg.child_frame_id = self.base_frame

        msg.pose.covariance = [0.0] * 36
        msg.twist.covariance = [0.0] * 36
        for i in range(6):
            msg.pose.covariance[i * 7] = IGNORE_COVARIANCE
            msg.twist.covariance[i * 7] = IGNORE_COVARIANCE

        if estimate.valid:
            variance = estimate.velocity_sigma_mps**2
            msg.twist.twist.linear.x = float(estimate.linear_mps[0])
            msg.twist.twist.linear.y = float(estimate.linear_mps[1])
            msg.twist.twist.angular.z = float(estimate.angular_rps[2])
            msg.twist.covariance[0] = variance * FORWARD_SIGMA_INFLATION**2
            msg.twist.covariance[7] = variance
            self.n_published += 1
            self.last_inliers = estimate.n_inliers
        else:
            self.refusals[estimate.reason] = self.refusals.get(estimate.reason, 0) + 1
            # Zero twist under an ignore-covariance, which the EKF discards. NOT a
            # measured standstill - see VoEstimate's docstring for why that
            # distinction is the difference between a gap and a corrupted filter.
            self.n_refused += 1

        self.publisher.publish(msg)

    def _report(self) -> None:
        total = self.n_published + self.n_refused
        if total == 0:
            self.get_logger().warn(
                "No frame pairs processed yet - check that the RGB and depth topics are "
                "both publishing and that their stamps agree."
            )
            return
        detail = ", ".join(f"{reason} x{count}" for reason, count in sorted(self.refusals.items()))
        message = (
            f"Visual odometry: {self.n_published}/{total} frame pairs used "
            f"(last estimate had {self.last_inliers} inliers)."
        )
        if self.refusals:
            message += f" Refused: {detail}."
        # A run where nothing is usable is a silent localization failure - the EKF
        # simply reverts to the wheel+IMU behaviour that scored 0/3 - so say so at
        # WARN rather than letting it read as normal progress.
        if self.n_published == 0:
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)


def main() -> None:
    rclpy.init()
    node = VisualOdometryNode()
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
