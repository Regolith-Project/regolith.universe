# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Generates a procedural lunar terrain world for the given seed and opens it in Gazebo.

    ros2 launch regolith_bringup terrain_only.launch.py seed:=42
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def _generate_and_launch(context, *args, **kwargs):
    # Imported lazily so `ros2 launch` argument parsing doesn't require the package at import time.
    from regolith_terrain_gen.cli import default_output_dir
    from regolith_terrain_gen.config import TerrainConfig
    from regolith_terrain_gen.generate import generate_world

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
    world_sdf_path = generate_world(cfg, default_output_dir(seed))

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={"gz_args": str(world_sdf_path)}.items(),
    )
    return [gz_sim]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("seed", default_value="42", description="Terrain generation seed"),
            OpaqueFunction(function=_generate_and_launch),
        ]
    )
