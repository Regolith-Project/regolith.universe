# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Procedural regolith surface textures: grey albedo, high-frequency normal map,
high (and slightly varying) roughness."""

from pathlib import Path

import numpy as np
from PIL import Image

from regolith_terrain_gen.noise import value_noise_2d


def _to_uint8(normalized: np.ndarray) -> np.ndarray:
    return (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)


def generate_albedo(resolution: int, rng: np.random.Generator) -> np.ndarray:
    """Grey regolith base with subtle low-frequency brightness variation."""
    variation = value_noise_2d((resolution, resolution), resolution / 6.0, rng)
    variation = (variation - variation.min()) / (variation.max() - variation.min() + 1e-9)
    grey = 0.42 + 0.12 * variation  # dark lunar regolith, ~0.42-0.54 albedo
    rgb = np.stack([grey, grey, grey * 1.01], axis=-1)  # imperceptibly cool grey, not literally flat
    return _to_uint8(rgb)


def _small_crater_pits(resolution: int, rng: np.random.Generator, tile_size_m: float,
                       count: int) -> np.ndarray:
    """A height field of small bowl-and-rim pits, for craters too small to be geometry.

    The rendered/collided surface is the coarse collision grid, so craters below roughly
    2x the collision cell size (~8 m) simply do not survive into it - see config's
    crater_count note. Putting sub-8 m pitting into the tiling surface texture gets that
    scale back visually for free, with no physics resolution spent.

    Caveat worth knowing: this texture TILES (every tile_size_m), so these pits repeat on
    that period. They are deliberately kept shallow and small - they should read as
    surface pitting from rover height, not as recognisable landmarks whose repetition
    gives the tiling away.
    """
    px_per_m = resolution / tile_size_m
    field = np.zeros((resolution, resolution))
    yy, xx = np.mgrid[0:resolution, 0:resolution]
    for _ in range(count):
        # 0.4-2.5 m across: below the ~8 m the collision surface can represent.
        radius_px = rng.uniform(0.4, 2.5) / 2.0 * px_per_m
        cx, cy = rng.uniform(0, resolution, size=2)
        # Wrap distance so pits crossing the tile edge stay seamless.
        dx = np.abs(xx - cx); dx = np.minimum(dx, resolution - dx)
        dy = np.abs(yy - cy); dy = np.minimum(dy, resolution - dy)
        x_norm = np.hypot(dx, dy) / max(radius_px, 1e-6)
        bowl = np.where(x_norm <= 1.0, -(1.0 - x_norm**2), 0.0)
        rim = np.exp(-(((x_norm - 1.0) / 0.35) ** 2))
        field += bowl + 0.3 * rim
    return field


def generate_normal_map(resolution: int, rng: np.random.Generator, strength: float = 0.35,
                        tile_size_m: float = 20.0, pit_count: int = 26) -> np.ndarray:
    """High-frequency micro-detail (small rocks/regolith grain) plus sub-resolution
    crater pitting, encoded as a tangent-space normal map."""
    detail = value_noise_2d((resolution, resolution), 18.0, rng) + 0.5 * value_noise_2d(
        (resolution, resolution), 9.0, rng
    )
    pits = _small_crater_pits(resolution, rng, tile_size_m, pit_count)
    # Scaled relative to the noise range so pits read as shape, not as a stamped pattern.
    detail = detail + 0.55 * (detail.max() - detail.min()) * pits
    gy, gx = np.gradient(detail)
    nx, ny = -gx * strength, -gy * strength
    nz = np.ones_like(nx)
    length = np.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny, nz = nx / length, ny / length, nz / length
    rgb = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1)
    return _to_uint8(rgb)


def generate_roughness_map(resolution: int, rng: np.random.Generator) -> np.ndarray:
    """High roughness overall (loose regolith has no specular highlight), slight variation."""
    variation = value_noise_2d((resolution, resolution), resolution / 8.0, rng)
    variation = (variation - variation.min()) / (variation.max() - variation.min() + 1e-9)
    roughness = 0.82 + 0.13 * variation
    return _to_uint8(np.stack([roughness] * 3, axis=-1))


def generate_textures(output_dir: Path, resolution: int, rng: np.random.Generator,
                      tile_size_m: float = 20.0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, array in (
        ("albedo", generate_albedo(resolution, rng)),
        ("normal", generate_normal_map(resolution, rng, tile_size_m=tile_size_m)),
        ("roughness", generate_roughness_map(resolution, rng)),
    ):
        path = output_dir / f"{name}.png"
        Image.fromarray(array, mode="RGB").save(path)
        paths[name] = path
    return paths
