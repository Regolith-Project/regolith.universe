# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Assemble the generated heightmap, textures, and rock scatter into a Gazebo world SDF,
plus a JSON manifest that downstream packages (costmap, planner) can consume."""

import json
from pathlib import Path

import numpy as np

from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.craters import Crater
from regolith_terrain_gen.scatter import RockInstance

_GUI_BLOCK = """    <gui fullscreen="false">
      <plugin filename="MinimalScene" name="3D View">
        <gz-gui>
          <title>3D View</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="string" key="state">docked</property>
        </gz-gui>
        <engine>ogre2</engine>
        <scene>scene</scene>
        <ambient_light>0.4 0.4 0.4</ambient_light>
        <background_color>0.01 0.01 0.02</background_color>
        <camera_pose>{camera_pose}</camera_pose>
      </plugin>
      <plugin filename="GzSceneManager" name="Scene Manager">
        <gz-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </gz-gui>
      </plugin>
      <plugin filename="InteractiveViewControl" name="Interactive view control">
        <gz-gui>
          <property key="resizable" type="bool">false</property>
          <property key="width" type="double">5</property>
          <property key="height" type="double">5</property>
          <property key="state" type="string">floating</property>
          <property key="showTitleBar" type="bool">false</property>
        </gz-gui>
      </plugin>
      <plugin filename="WorldControl" name="World control">
        <gz-gui>
          <title>World control</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">72</property>
          <property type="double" key="width">121</property>
          <property type="double" key="z">1</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View">
            <line own="left" target="left"/>
            <line own="bottom" target="bottom"/>
          </anchors>
        </gz-gui>
        <play_pause>true</play_pause>
        <step>true</step>
        <start_paused>true</start_paused>
      </plugin>
      <plugin filename="WorldStats" name="World stats">
        <gz-gui>
          <title>World stats</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">110</property>
          <property type="double" key="width">290</property>
          <property type="double" key="z">1</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View">
            <line own="right" target="right"/>
            <line own="bottom" target="bottom"/>
          </anchors>
        </gz-gui>
        <sim_time>true</sim_time>
        <real_time>true</real_time>
        <real_time_factor>true</real_time_factor>
        <iterations>true</iterations>
      </plugin>
    </gui>
"""


def _sun_direction(elevation_deg: float, azimuth_deg: float) -> tuple:
    elevation = np.deg2rad(elevation_deg)
    azimuth = np.deg2rad(azimuth_deg)
    dx = np.cos(elevation) * np.cos(azimuth)
    dy = np.cos(elevation) * np.sin(azimuth)
    dz = -np.sin(elevation)
    return dx, dy, dz


def _rock_model_sdf(rock: RockInstance, index: int, mesh_dir: Path) -> str:
    mesh_uri = f"file://{mesh_dir / (rock.variant + '.obj')}"
    return f"""    <model name="rock_{index}">
      <static>true</static>
      <pose>{rock.x_m:.3f} {rock.y_m:.3f} {rock.z_m:.3f} 0 0 {rock.yaw_rad:.4f}</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <mesh>
              <uri>{mesh_uri}</uri>
              <scale>{rock.scale_m:.3f} {rock.scale_m:.3f} {rock.scale_m:.3f}</scale>
            </mesh>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>{mesh_uri}</uri>
              <scale>{rock.scale_m:.3f} {rock.scale_m:.3f} {rock.scale_m:.3f}</scale>
            </mesh>
          </geometry>
          <material>
            <diffuse>0.32 0.30 0.29 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
            <pbr>
              <metal>
                <roughness>0.92</roughness>
                <metalness>0.0</metalness>
              </metal>
            </pbr>
          </material>
        </visual>
      </link>
    </model>
"""


def build_world_sdf(
    cfg: TerrainConfig,
    heightmap_png: Path,
    texture_pngs: dict,
    rocks: list,
    rock_mesh_dir: Path,
) -> str:
    dx, dy, dz = _sun_direction(cfg.sun_elevation_deg, cfg.sun_azimuth_deg)
    # Elevated oblique viewpoint well outside the crater field, looking down across it
    # so both the crater field and the low-sun long shadows are visible at once.
    camera_pose = "-110 -110 35 0 0.28 0.78"

    rock_models = "\n".join(_rock_model_sdf(rock, i, rock_mesh_dir) for i, rock in enumerate(rocks))

    return f"""<?xml version="1.0"?>
<sdf version="1.10">
  <world name="regolith_moon">
    <gravity>0 0 -1.62</gravity>
    <physics name="default_physics" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <scene>
      <ambient>0.06 0.06 0.07 1</ambient>
      <background>0.01 0.01 0.02 1</background>
      <shadows>true</shadows>
      <grid>false</grid>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <direction>{dx:.4f} {dy:.4f} {dz:.4f}</direction>
      <diffuse>1 0.97 0.90 1</diffuse>
      <specular>0.02 0.02 0.02 1</specular>
      <attenuation>
        <range>2000</range>
        <constant>1.0</constant>
        <linear>0</linear>
        <quadratic>0</quadratic>
      </attenuation>
    </light>

    <model name="moon_terrain">
      <static>true</static>
      <link name="terrain_link">
        <collision name="terrain_collision">
          <geometry>
            <heightmap>
              <uri>file://{heightmap_png}</uri>
              <size>{cfg.world_size_m} {cfg.world_size_m} {cfg.height_range_m}</size>
              <pos>0 0 0</pos>
            </heightmap>
          </geometry>
        </collision>
        <visual name="terrain_visual">
          <geometry>
            <heightmap>
              <uri>file://{heightmap_png}</uri>
              <size>{cfg.world_size_m} {cfg.world_size_m} {cfg.height_range_m}</size>
              <pos>0 0 0</pos>
              <texture>
                <diffuse>file://{texture_pngs['albedo']}</diffuse>
                <normal>file://{texture_pngs['normal']}</normal>
                <size>20</size>
              </texture>
            </heightmap>
          </geometry>
        </visual>
      </link>
    </model>

{rock_models}
{_GUI_BLOCK.format(camera_pose=camera_pose)}
  </world>
</sdf>
"""


def write_manifest(
    path: Path,
    cfg: TerrainConfig,
    craters: list,
    rocks: list,
    heightmap_png: Path,
    world_sdf: Path,
) -> None:
    manifest = {
        "seed": cfg.seed,
        "world_size_m": cfg.world_size_m,
        "height_range_m": cfg.height_range_m,
        "heightmap_resolution_px": cfg.heightmap_resolution_px,
        "heightmap_png": str(heightmap_png),
        "world_sdf": str(world_sdf),
        "spawn_zone": {
            "x_m": cfg.spawn_zone_center[0],
            "y_m": cfg.spawn_zone_center[1],
            "radius_m": cfg.spawn_zone_radius_m,
        },
        "craters": [
            {
                "x_m": c.x_m,
                "y_m": c.y_m,
                "diameter_m": c.diameter_m,
                "depth_m": c.depth_m,
                "rim_height_m": c.rim_height_m,
            }
            for c in craters
        ],
        "rocks": [
            {
                "x_m": r.x_m,
                "y_m": r.y_m,
                "z_m": r.z_m,
                "yaw_rad": r.yaw_rad,
                "scale_m": r.scale_m,
                "variant": r.variant,
            }
            for r in rocks
        ],
    }
    path.write_text(json.dumps(manifest, indent=2))
