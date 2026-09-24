# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Earth, fixed in the lunar sky.

Earth is placed at a fixed (azimuth, elevation) - real lunar-surface Earth does
not rise or set (the Moon is tidally locked, so from a fixed point on the near
side Earth only ever librates a few degrees), so a static pose is not a
simplification, it is the correct behaviour.

PHASE COMES FOR FREE FROM ORDINARY LIGHTING - deliberately not baked into a
texture. Earth is an ordinary *lit* sphere (unlike the sky dome, its material
uses the scene's default lighting=true), textured with the real NASA Blue Marble
albedo map. gz's directional "sun" light - the same light and direction
worldgen.py computes from cfg.sun_elevation_deg/sun_azimuth_deg - shades it with
the same physically-based N-dot-L falloff it shades the terrain with. That means:

  - the illuminated crescent/gibbous shape is automatically consistent with
    whatever sun direction the scene is using, with no separate phase
    computation or texture-blending code to keep in sync with worldgen's sun
    vector;
  - the night side receives only the scene's ambient term (near-black by
    default - see TerrainConfig.scene_ambient), so it renders genuinely dark
    rather than a flat unlit grey disc;
  - there's no hard day/night edge artefact from a baked terminator texture,
    because the terminator here is a real per-pixel lighting computation on a
    smooth sphere, which is inherently soft across a few pixels the way a
    physically lit sphere's terminator actually looks.

Earth does not cast a shadow (`cast_shadows=false` on its visual): at these
distances and this light's flat SDF `directional` model it would only ever
throw a shadow into empty space anyhow, so it costs nothing to disable and
saves a large-radius object from ever being considered by the shadow pass.
"""

from pathlib import Path

import numpy as np
from regolith_terrain_gen.config import TerrainConfig

ASSET_DIR = Path(__file__).parent / "assets"
# PNG, not the JPEG NASA ships it as: a first render pass with the JPEG came back
# with Earth rendering as a flat black disc - every other textured material in this
# codebase (terrain albedo/normal/roughness, rock albedo) is a PNG, and this JPEG
# was the one exception. Re-encoded losslessly from the same source pixels (see
# docs/media/README.md for provenance) rather than risk it being a format-support
# gap in this Ogre2/gz-rendering install.
EARTH_ALBEDO_PATH = ASSET_DIR / "earth_albedo.png"


def earth_position_m(cfg: TerrainConfig) -> tuple:
    """Fixed world-frame (x, y, z) of Earth's centre, from its configured azimuth/elevation/distance.

    Same azimuth/elevation convention as worldgen._sun_direction (0 deg azimuth
    = +x, elevation measured up from the local horizontal) so the two are easy
    to reason about together, even though Earth's position is independent of
    the sun's.
    """
    el = np.deg2rad(cfg.earth_elevation_deg)
    az = np.deg2rad(cfg.earth_azimuth_deg)
    d = cfg.earth_distance_m
    x = d * np.cos(el) * np.cos(az)
    y = d * np.cos(el) * np.sin(az)
    z = d * np.sin(el)
    return float(x), float(y), float(z)


def earth_radius_m(cfg: TerrainConfig) -> float:
    """Sphere radius giving the configured angular diameter at earth_distance_m.

    Real Earth-from-the-Moon subtends ~1.9 deg (Earth's 12742 km diameter at the
    Moon's ~384400 km mean distance); this defaults to 2.0 deg, "about four times
    the Moon seen from Earth" (~0.5 deg), matching the brief.
    """
    half_angle = np.deg2rad(cfg.earth_angular_diameter_deg) / 2.0
    return float(cfg.earth_distance_m * np.tan(half_angle))


def earth_model_sdf(cfg: TerrainConfig) -> str:
    x, y, z = earth_position_m(cfg)
    radius = earth_radius_m(cfg)
    # Yaw the sphere so the albedo map's prime-meridian seam faces roughly away
    # from the camera's usual approach (spawn sits near the origin looking
    # +x/+y) - purely cosmetic, does not affect phase/lighting.
    yaw = np.deg2rad(200.0)
    return f"""    <model name="earth">
      <static>true</static>
      <pose>{x:.2f} {y:.2f} {z:.2f} 0 0 {yaw:.4f}</pose>
      <link name="link">
        <!-- Visual only: no collision geometry, not written to manifest.json - see
             sky.py's docstring for why that matters to the costmap/planner. -->
        <visual name="earth_visual">
          <cast_shadows>false</cast_shadows>
          <geometry>
            <sphere><radius>{radius:.3f}</radius></sphere>
          </geometry>
          <material>
            <!-- Deliberately lighting=true (the default) - see module docstring:
                 this is what gives Earth a phase that automatically matches the
                 scene's sun direction, for free, with a genuinely dark night side. -->
            <ambient>1 1 1 1</ambient>
            <diffuse>1 1 1 1</diffuse>
            <specular>0.05 0.05 0.05 1</specular>
            <pbr>
              <metal>
                <albedo_map>file://{EARTH_ALBEDO_PATH}</albedo_map>
                <roughness>0.92</roughness>
                <metalness>0.0</metalness>
              </metal>
            </pbr>
          </material>
        </visual>
      </link>
    </model>
"""
