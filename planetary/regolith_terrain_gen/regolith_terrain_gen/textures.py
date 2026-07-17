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


def generate_normal_map(resolution: int, rng: np.random.Generator, strength: float = 0.35) -> np.ndarray:
    """High-frequency micro-detail (small rocks/regolith grain) encoded as a tangent-space normal map."""
    detail = value_noise_2d((resolution, resolution), 18.0, rng) + 0.5 * value_noise_2d(
        (resolution, resolution), 9.0, rng
    )
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


def generate_textures(output_dir: Path, resolution: int, rng: np.random.Generator) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, array in (
        ("albedo", generate_albedo(resolution, rng)),
        ("normal", generate_normal_map(resolution, rng)),
        ("roughness", generate_roughness_map(resolution, rng)),
    ):
        path = output_dir / f"{name}.png"
        Image.fromarray(array, mode="RGB").save(path)
        paths[name] = path
    return paths
