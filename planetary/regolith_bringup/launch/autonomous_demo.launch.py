# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""The actual hello-world: generates terrain, spawns the rover, localizes it (EKF),
builds a traversability costmap from the known terrain, and autonomously plans +
drives to wherever you click "2D Goal Pose" in RViz.

    ros2 launch regolith_bringup autonomous_demo.launch.py seed:=42

Then in RViz, use the "2D Goal Pose" tool to click a goal.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.actions import TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

ROVER_NAME = "rover"
WORLD_NAME = "regolith_moon"


def _generate_and_launch(context, *args, **kwargs):
    import json

    from regolith_terrain_gen.cli import default_output_dir
    from regolith_terrain_gen.config import TerrainConfig
    from regolith_terrain_gen.generate import generate_world
    import xacro

    raw_seed = LaunchConfiguration("seed").perform(context)
    try:
        seed = int(raw_seed)
        if seed < 0:
            raise ValueError
    except ValueError:
        raise RuntimeError(
            f"Launch argument 'seed' must be a non-negative integer, got '{raw_seed}'"
        ) from None
    cfg = TerrainConfig(seed=seed)
    output_dir = default_output_dir(seed)
    world_sdf_path = generate_world(cfg, output_dir, start_paused=False)
    manifest_path = output_dir / "manifest.json"

    # Spawn height must be above the ACTUAL local terrain elevation, not a fixed
    # guess - see PROGRESS.md M2 and heightmap.py's build_terrain_collision_boxes_sdf.
    manifest = json.loads(manifest_path.read_text())
    spawn_z = manifest["spawn_zone"]["elevation_m"] + 0.5

    xacro_path = (
        FindPackageShare("regolith_rover_description").find("regolith_rover_description")
        + "/urdf/regolith_rover.urdf.xacro"
    )
    urdf_xml = xacro.process_file(xacro_path).toxml()
    rover_urdf_path = output_dir / "rover.urdf"
    rover_urdf_path.write_text(urdf_xml)

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]),
        launch_arguments={"gz_args": f"-r {world_sdf_path}"}.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": urdf_xml, "use_sim_time": True}],
    )

    # A short delay gives Gazebo a moment to finish starting up before the spawn
    # service call arrives.
    spawn_rover = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "run",
                    "ros_gz_sim",
                    "create",
                    "-world",
                    WORLD_NAME,
                    "-file",
                    str(rover_urdf_path),
                    "-name",
                    ROVER_NAME,
                    "-x",
                    "0",
                    "-y",
                    "0",
                    "-z",
                    str(spawn_z),
                ],
                output="screen",
            )
        ],
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/camera@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            f"/world/{WORLD_NAME}/model/{ROVER_NAME}/joint_state@sensor_msgs/msg/JointState@gz.msgs.Model",
            # Ground truth, comparison only - deliberately NOT fed into the EKF, and
            # NOT bridged to /tf (the EKF below is the sole publisher of odom -> base_link).
            f"/model/{ROVER_NAME}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
        ],
        remappings=[
            ("/odometry", "/odom"),
            ("/camera", "/camera/image"),
            ("/camera_info", "/camera/camera_info"),
            (f"/world/{WORLD_NAME}/model/{ROVER_NAME}/joint_state", "/joint_states"),
            (f"/model/{ROVER_NAME}/pose", "/ground_truth/pose"),
        ],
    )

    # See PROGRESS.md M3 for why this relay is necessary (gz-sim publishes all-zero
    # sensor covariance, which robot_localization treats as "ignore this measurement";
    # the IMU's frame_id also needs fixing to match the URDF's TF tree).
    sensor_covariance_relay = Node(
        package="regolith_bringup",
        executable="sensor_covariance_relay.py",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    ekf_config = FindPackageShare("regolith_bringup").find("regolith_bringup") + "/config/ekf.yaml"
    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_config],
    )

    costmap_node = Node(
        package="regolith_costmap",
        executable="costmap_node",
        output="screen",
        parameters=[
            {
                "manifest_path": str(manifest_path),
                "resolution_m": 1.0,
                "rover_radius_m": 0.3,
                "slope_lethal_deg": 20.0,
                "use_sim_time": True,
            }
        ],
    )

    planner_node = Node(
        package="regolith_planner",
        executable="planner_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    pure_pursuit_node = Node(
        package="regolith_vehicle_interface",
        executable="pure_pursuit_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return [
        gz_sim,
        robot_state_publisher,
        spawn_rover,
        bridge,
        sensor_covariance_relay,
        ekf_node,
        costmap_node,
        planner_node,
        pure_pursuit_node,
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "seed", default_value="42", description="Terrain generation seed"
            ),
            OpaqueFunction(function=_generate_and_launch),
        ]
    )
