# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates lunar terrain, spawns the Regolith rover, and bridges ROS <-> Gazebo topics.

    ros2 launch regolith_bringup teleop_demo.launch.py seed:=42

Drive with, in another terminal:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
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

    seed = int(LaunchConfiguration("seed").perform(context))
    cfg = TerrainConfig(seed=seed)
    output_dir = default_output_dir(seed)
    world_sdf_path = generate_world(cfg, output_dir, start_paused=False)

    # Spawn height must be above the ACTUAL local terrain elevation, not a fixed
    # guess: the terrain's collision boxes sit right at that elevation, and spawning
    # below/inside solid collision geometry produces erratic physics (see the long
    # note in heightmap.py's build_terrain_collision_boxes_sdf).
    manifest = json.loads((output_dir / "manifest.json").read_text())
    spawn_z = manifest["spawn_zone"]["elevation_m"] + 0.5

    xacro_path = FindPackageShare("regolith_rover_description").find(
        "regolith_rover_description"
    ) + "/urdf/regolith_rover.urdf.xacro"
    urdf_xml = xacro.process_file(xacro_path).toxml()
    rover_urdf_path = output_dir / "rover.urdf"
    rover_urdf_path.write_text(urdf_xml)

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
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
                    "ros2", "run", "ros_gz_sim", "create",
                    "-world", WORLD_NAME,
                    "-file", str(rover_urdf_path),
                    "-name", ROVER_NAME,
                    "-x", "0", "-y", "0", "-z", str(spawn_z),
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
            "/pose[tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            "/camera@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            f"/world/{WORLD_NAME}/model/{ROVER_NAME}/joint_state@sensor_msgs/msg/JointState@gz.msgs.Model",
        ],
        remappings=[
            ("/odometry", "/odom"),
            ("/pose", "/tf"),
            ("/camera", "/camera/image"),
            ("/camera_info", "/camera/camera_info"),
            (f"/world/{WORLD_NAME}/model/{ROVER_NAME}/joint_state", "/joint_states"),
        ],
    )

    return [gz_sim, robot_state_publisher, spawn_rover, bridge]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("seed", default_value="42", description="Terrain generation seed"),
            OpaqueFunction(function=_generate_and_launch),
        ]
    )
