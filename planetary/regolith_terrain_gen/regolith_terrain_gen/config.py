# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Tunable parameters for procedural lunar terrain generation."""

from dataclasses import dataclass, field


@dataclass
class TerrainConfig:
    seed: int = 42

    # World / heightmap
    world_size_m: float = 200.0
    heightmap_resolution_px: int = 513  # 2^n + 1, standard heightmap convention
    height_range_m: float = 10.0

    # Base roughness (fractal Brownian motion)
    fbm_octaves: int = 5
    fbm_base_cell_px: float = 96.0
    fbm_lacunarity: float = 2.0
    fbm_gain: float = 0.5
    fbm_weight_m: float = 0.9  # contribution of base roughness before final normalization

    # Regional slope
    regional_slope_deg: float = 1.5

    # Crater field (power-law size-frequency distribution)
    # crater_count and rock_count were raised from their original values (60 / 130)
    # and spawn_zone_radius_m lowered from 12.0: measured against the actual costmap
    # lethality (not just raw obstacle footprints), the original density let a
    # majority of short (10-20 m) straight lines pass completely clear of any
    # obstacle - e.g. the shipped tour_mission.py's fixed 5-leg route crossed an
    # obstacle on only 1 of 5 legs, so the rover barely had to turn. These values
    # were chosen by measuring straight-line-blocked fraction and A* reachability
    # together across several seeds (60-100 m and 10-20 m goals) rather than eyeballed
    # - see PROGRESS.md's "Terrain density increase" note for the actual numbers
    # and the trade-off against a small increase in unreachable-goal risk.
    crater_count: int = 100
    crater_diameter_min_m: float = 2.0
    crater_diameter_max_m: float = 40.0
    crater_size_exponent: float = 2.0  # N(>D) ~ D^-exponent
    crater_depth_to_diameter: float = 0.055
    crater_rim_height_frac: float = 0.35  # rim height as a fraction of crater depth
    crater_rim_width_frac: float = 0.18  # rim gaussian width as a fraction of crater radius

    # Spawn zone (guaranteed traversable, kept clear of craters/rocks)
    spawn_zone_radius_m: float = 9.0
    spawn_zone_center: tuple = (0.0, 0.0)

    # Rocks
    rock_count: int = 190
    rock_variant_count: int = 4
    rock_scale_min_m: float = 0.3
    rock_scale_max_m: float = 2.4
    rock_subdivisions: int = 1

    # Lighting
    sun_elevation_deg: float = 12.0
    sun_azimuth_deg: float = 235.0

    # Surface texture
    texture_resolution_px: int = 512

    def __post_init__(self) -> None:
        if self.heightmap_resolution_px % 2 == 0:
            raise ValueError("heightmap_resolution_px should be odd (2^n + 1)")
