# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Assemble the generated heightmap, textures, and rock scatter into a Gazebo world SDF, plus a JSON manifest that downstream packages (costmap, planner) can consume."""

import json
from pathlib import Path

import numpy as np
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.earth import earth_model_sdf
from regolith_terrain_gen.farfield import farfield_model_sdf
from regolith_terrain_gen.scatter import RockInstance

# See _rock_model_sdf's docstring: a constant extra sink applied ONLY to the new visual
# rock mesh's own local pose, never to the frozen collision ellipsoid or rock.z_m.
VISUAL_SEATING_MARGIN_M = 0.06
from regolith_terrain_gen.sky import sky_model_sdf

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
        <start_paused>{start_paused}</start_paused>
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


def _rock_model_sdf(
    rock: RockInstance, index: int, mesh_dir: Path, visual_mesh_paths: dict = None, tint: tuple = None
) -> str:
    """Collision stays an ELLIPSOID, unconditionally, sized from `rock.collision_radii_m`
    - <mesh> collision is silently a no-op in this gz-physics install (verified - a probe
    drops straight through one). See rocks.RockVariant.

    The VISUAL mesh, if `visual_mesh_paths` supplies one for this rock's variant, is the
    new shape-varied mesh from rocks_visual.py - fit inside this exact ellipsoid by
    construction (see rocks_visual.fit_to_collision_envelope and
    test_rock_visual_meshes_fit_frozen_collision). Falls back to the original
    displaced-icosphere mesh in `mesh_dir` if no visual variant is supplied, so this
    function still works standalone (tests, older callers).

    `tint` (r, g, b multipliers) gives each rock instance its own diffuse variation
    instead of one flat grey for all 190 - see rocks_visual.rock_albedo_tints.

    VISUAL_SEATING_MARGIN_M: only applied when a NEW visual mesh is in play, and only to
    the <visual>'s own local pose - never to <collision> or to rock.z_m, both of which
    stay exactly as frozen. It exists because a differently-shaped visual mesh (not the
    exact mesh seat_rock_z computed rock.z_m against) can, for the same frozen z_m/
    rotation, come up a little short of the original mesh's own lowest reach: measured on
    seed 7, 2 of 190 rocks came up 1.5-1.8 cm short even with the shape-family and
    ellipsoid-fit precautions in rocks_visual.py (see that module's docstring). Sinking
    the VISUAL mesh a further constant amount absorbs that residual with margin - it is
    the same kind of deliberate under-drawing embed_frac already does, just applied to
    the newer, second mesh instead of the original. Verified by
    test_no_visual_rock_hangs_above_the_drawn_ground (test_rock_visual_seating_and_
    containment.py): 0 rocks float across seeds 42/7/123 with this in place at 0.06 m -
    the surface-budget rebalance in terrain_detail.py moved the residual enough that the
    margin needed re-measuring and was raised from an earlier 0.04 m; see PROGRESS.md's
    terrain-realism-pass note for the actual per-seed worst-case gap.
    """
    if visual_mesh_paths and rock.variant in visual_mesh_paths:
        mesh_uri = f"file://{visual_mesh_paths[rock.variant]}"
        visual_pose = f"<pose>0 0 {-VISUAL_SEATING_MARGIN_M:.3f} 0 0 0</pose>"
    else:
        mesh_uri = f"file://{mesh_dir / (rock.variant + '.obj')}"
        visual_pose = ""
    rx, ry, rz = rock.collision_radii_m
    tr, tg, tb = tint if tint is not None else (1.0, 1.0, 1.0)
    # Base raised from (0.32, 0.30, 0.29): measured against renders, that base plus a
    # narrow +/-10% tint left all 190 boulders reading as near-black silhouettes against
    # the ~0.40-0.54 albedo ground - real lunar basalt boulders span a wider range than
    # that, some notably brighter (weathered/dusty) than the soil around them, some
    # darker (fresher, less regolith cover). rock_albedo_variation is now wide enough
    # (see config.py) that tint alone produces that spread; the base only needed to move
    # up, not the spread widened further here.
    dr, dg, db = 0.40 * tr, 0.38 * tg, 0.36 * tb
    return f"""    <model name="rock_{index}">
      <static>true</static>
      <pose>{rock.x_m:.3f} {rock.y_m:.3f} {rock.z_m:.3f} {rock.roll_rad:.4f} {rock.pitch_rad:.4f} {rock.yaw_rad:.4f}</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <ellipsoid>
              <radii>{rx:.3f} {ry:.3f} {rz:.3f}</radii>
            </ellipsoid>
          </geometry>
          <surface><friction><ode><mu>1.1</mu><mu2>1.1</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          {visual_pose}
          <geometry>
            <mesh>
              <uri>{mesh_uri}</uri>
              <scale>{rock.scale_m:.3f} {rock.scale_m:.3f} {rock.scale_m:.3f}</scale>
            </mesh>
          </geometry>
          <material>
            <diffuse>{dr:.3f} {dg:.3f} {db:.3f} 1</diffuse>
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


def _pebble_field_model_sdf(pebble_field_obj: Path) -> str:
    """Thousands of small stones welded into ONE mesh - one draw call, no collision at
    all (visual-only ground break-up at rover scale; see rocks_visual.save_pebble_field_obj
    and config.py's pebble_* fields for why that is safe)."""
    return f"""    <model name="pebble_field">
      <static>true</static>
      <link name="link">
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>file://{pebble_field_obj}</uri>
            </mesh>
          </geometry>
          <material>
            <!-- Brighter than the ground (0.40-0.54 albedo) and noticeably brighter than
                 the boulders (0.29-0.45 - see _rock_model_sdf): real fine regolith/dust
                 scatters light more diffusely per unit area than a boulder's more
                 shadowed facets, and at the size these render (a handful of pixels at
                 orbital/terrain range) anything darker than the ground disappears into
                 flat black specks against it, which is what an earlier, dimmer value
                 (0.36-0.44) measured as in a render - "black specks" rather than stones.
                 Kept out of the true 0.40-0.54 ground albedo band so pebbles still read
                 as distinct grains rather than invisible against the terrain. -->
            <diffuse>0.52 0.50 0.47 1</diffuse>
            <specular>0.12 0.12 0.12 1</specular>
            <pbr>
              <metal>
                <roughness>0.78</roughness>
                <metalness>0.0</metalness>
              </metal>
            </pbr>
          </material>
        </visual>
      </link>
    </model>
"""


# Opening GUI viewpoint, expressed RELATIVE to the spawn point rather than as an
# absolute pose: back off this far along -x/-y and sit this high above the ground.
# Sized so the 0.4 m rover is unmistakable the moment the window opens - at ~7 m it
# spans ~60 px in a 1200 px-wide window. Two earlier absolute poses were both too far
# out to see it at all: "-110 -110 35" (~155 m, rover 2-3 px, reported as "Gazebo
# shows nothing but terrain") and then "-22 -22 13" (~32 m, rover ~13 px, still
# reported as not showing the rover - measured directly off a GUI screenshot).
_CAM_BACK_M = 4.5
_CAM_HEIGHT_M = 3.0
# Aim at the chassis rather than its contact patch, so the rover sits just above the
# frame centre instead of on the horizon line.
_CAM_TARGET_HEIGHT_M = 0.17


def _gui_camera_pose(cfg: TerrainConfig, elevation_lookup=None) -> str:
    """Compute the opening GUI camera pose, placed relative to the actual terrain height at the spawn point.

    The pose used to be a hardcoded absolute string, which silently assumed a terrain
    elevation: spawn elevation is seed-dependent (5.2 m for seed 42, 6.1 m for seed 7,
    and the fBm range is 10 m), so a fixed z is a different height above the ground for
    every seed - and at these much closer camera distances that error is no longer
    cosmetic. It is also sampled at the CAMERA's own (x, y), not just the spawn point:
    a camera placed a fixed height above the spawn elevation can end up underground if
    the terrain rises behind the rover (that exact mistake - camera at z=2.5 against
    ~5.2 m local terrain, rendering the underside of the terrain - cost time during the
    investigation in PROGRESS.md).

    Both camera and target sit inside the spawn zone, which place_craters and
    scatter_rocks both keep clear, so nothing can occlude the opening shot.
    """
    sx, sy = cfg.spawn_zone_center
    cam_x, cam_y = sx - _CAM_BACK_M, sy - _CAM_BACK_M

    if elevation_lookup is None:
        spawn_ground = camera_ground = 0.0
    else:
        spawn_ground = elevation_lookup(sx, sy)
        camera_ground = elevation_lookup(cam_x, cam_y)

    cam_z = max(spawn_ground, camera_ground) + _CAM_HEIGHT_M
    horizontal = float(np.hypot(sx - cam_x, sy - cam_y))
    pitch = float(np.arctan2(cam_z - (spawn_ground + _CAM_TARGET_HEIGHT_M), horizontal))
    yaw = float(np.arctan2(sy - cam_y, sx - cam_x))
    return f"{cam_x:.3f} {cam_y:.3f} {cam_z:.3f} 0 {pitch:.3f} {yaw:.3f}"


def build_world_sdf(
    cfg: TerrainConfig,
    texture_pngs: dict,
    rocks: list,
    rock_mesh_dir: Path,
    terrain_collision_sdf: str,
    terrain_mesh_obj: Path,
    elevation_lookup=None,
    start_paused: bool = True,
    sky_texture_path: Path = None,
    sky_mesh_obj: Path = None,
    farfield_mesh_obj: Path = None,
    rock_visual_mesh_paths: dict = None,
    rock_tints=None,
    pebble_field_obj: Path = None,
) -> str:
    # No heightmap_png / z_min / z_span here any more: the ground is drawn from
    # terrain_mesh_obj, which carries its own absolute world coordinates. The PNG's
    # min/max-stretch encoding still matters - it is how the costmap decodes elevations -
    # but it is no longer part of the world SDF. See save_heightmap_png and terrain_mesh.
    dx, dy, dz = _sun_direction(cfg.sun_elevation_deg, cfg.sun_azimuth_deg)
    camera_pose = _gui_camera_pose(cfg, elevation_lookup)

    rock_models = "\n".join(
        _rock_model_sdf(
            rock,
            i,
            rock_mesh_dir,
            visual_mesh_paths=rock_visual_mesh_paths,
            tint=tuple(rock_tints[i]) if rock_tints is not None else None,
        )
        for i, rock in enumerate(rocks)
    )
    if pebble_field_obj is not None:
        rock_models += "\n" + _pebble_field_model_sdf(pebble_field_obj)

    # Sky/Earth/far-field horizon - all VISUAL ONLY (no <collision>, never written to
    # manifest.json - see sky.py/earth.py/farfield.py). Each is gated on both its own
    # cfg.*_enabled flag and generate.py actually having built the asset it needs
    # (sky_texture_path / farfield_mesh_obj), so build_world_sdf still works standalone
    # (e.g. in tests) without those assets on disk.
    extra_models = ""
    if cfg.sky_enabled and sky_texture_path is not None and sky_mesh_obj is not None:
        extra_models += sky_model_sdf(cfg, sky_texture_path, sky_mesh_obj)
    if cfg.earth_enabled:
        extra_models += earth_model_sdf(cfg)
    if cfg.farfield_enabled and farfield_mesh_obj is not None:
        extra_models += farfield_model_sdf(cfg, farfield_mesh_obj, texture_pngs)

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
      <ambient>{cfg.scene_ambient[0]:.3f} {cfg.scene_ambient[1]:.3f} {cfg.scene_ambient[2]:.3f} 1</ambient>
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
        <!-- Collision is a box grid approximating the terrain, and NOT the mesh used
             for the visual below: both <heightmap> and <mesh> collision construction
             from SDF are unimplemented for dartsim/bullet/bullet-featherstone in this
             gz-physics 7.8.0 install (verified empirically across all three engines -
             see heightmap.py). The visual being free of physics duty is what made
             swapping it for a mesh a purely rendering-side change. -->
{terrain_collision_sdf}
        <!-- The ground is a MESH, not a <heightmap>. A <heightmap> visual is drawn by
             Ogre-Next's Terra, which point-samples it coarser the further it is from
             the camera; the rocks standing on it are meshes and keep their exact
             placement, so at the horizon they hang visibly in the air while every
             placement test - all of which grade the data on disk - stays green. A
             <mesh> has no LOD in this stack and is drawn as authored at every range.
             See terrain_mesh.py for the measurement. heightmap.png is still written
             and still ships; the costmap reads it, but it is no longer drawn. -->
        <visual name="terrain_visual">
          <geometry>
            <mesh>
              <uri>file://{terrain_mesh_obj}</uri>
            </mesh>
          </geometry>
          <material>
            <diffuse>1 1 1 1</diffuse>
            <specular>0.05 0.05 0.05 1</specular>
            <pbr>
              <metal>
                <albedo_map>file://{texture_pngs['albedo']}</albedo_map>
                <normal_map>file://{texture_pngs['normal']}</normal_map>
                <roughness_map>file://{texture_pngs['roughness']}</roughness_map>
                <metalness>0.0</metalness>
              </metal>
            </pbr>
          </material>
        </visual>
      </link>
    </model>

{rock_models}
{extra_models}
{_GUI_BLOCK.format(camera_pose=camera_pose, start_paused=str(start_paused).lower())}
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
    spawn_elevation_m: float,
    heightmap_z_min_m: float,
    heightmap_z_span_m: float,
    terrain_mesh_obj: Path,
    terrain_mesh_stats: dict,
) -> None:
    manifest = {
        "seed": cfg.seed,
        "world_size_m": cfg.world_size_m,
        "height_range_m": cfg.height_range_m,
        # The real elevations pixel 0 and pixel 65535 decode to, straight from
        # save_heightmap_png - NOT height_range_m, which is the range the generator was
        # ALLOWED to use, not the one this seed's surface actually occupies (typically
        # ~8 m of the configured 10). Anyone decoding the PNG must use these: assuming
        # height_range_m overstates every slope by span/height_range_m, which is how the
        # costmap ran a ~16 deg effective lethal threshold against its configured 20.
        # See PROGRESS.md "costmap decodes the wrong height span".
        "heightmap_z_min_m": heightmap_z_min_m,
        "heightmap_z_span_m": heightmap_z_span_m,
        "heightmap_resolution_px": cfg.heightmap_resolution_px,
        "heightmap_png": str(heightmap_png),
        # The geometry that is actually DRAWN, and the surface anything standing on the
        # ground is seated against. heightmap_png above is the same surface sampled at
        # every post, kept for the costmap; the mesh is what gz renders.
        "terrain_mesh_obj": str(terrain_mesh_obj),
        "terrain_mesh": terrain_mesh_stats,
        "world_sdf": str(world_sdf),
        "spawn_zone": {
            "x_m": cfg.spawn_zone_center[0],
            "y_m": cfg.spawn_zone_center[1],
            "radius_m": cfg.spawn_zone_radius_m,
            "elevation_m": spawn_elevation_m,
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
                "roll_rad": r.roll_rad,
                "pitch_rad": r.pitch_rad,
                "yaw_rad": r.yaw_rad,
                "scale_m": r.scale_m,
                "variant": r.variant,
                # True horizontal footprint of the collision ellipsoid, so the costmap
                # can mark what the rover can actually hit instead of assuming a
                # scale_m-radius circle (scale_m is the mesh BOUNDING radius, which
                # overstates the thin axes of these deliberately anisotropic boulders).
                "collision_radii_m": list(r.collision_radii_m),
            }
            for r in rocks
        ],
    }
    path.write_text(json.dumps(manifest, indent=2))
