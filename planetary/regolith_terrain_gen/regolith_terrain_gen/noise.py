# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Fractal Brownian motion base terrain via smoothed value noise.

No external noise library is required: each octave is a coarse random
lattice smoothly upsampled with cubic interpolation, which is a standard,
dependency-free stand-in for simplex/Perlin noise at this fidelity.
"""

import numpy as np
from scipy import ndimage


def value_noise_2d(shape: tuple, cell_size: float, rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    cell_size = max(2.0, cell_size)
    lattice_h = int(h / cell_size) + 3
    lattice_w = int(w / cell_size) + 3
    lattice = rng.uniform(-1.0, 1.0, size=(lattice_h, lattice_w))
    zoomed = ndimage.zoom(lattice, cell_size, order=3, mode="reflect")
    return zoomed[:h, :w]


def fbm(
    shape: tuple,
    rng: np.random.Generator,
    octaves: int = 5,
    base_cell: float = 96.0,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Sum of progressively finer, weaker value-noise octaves, normalized to roughly [-1, 1]."""
    result = np.zeros(shape, dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0
    cell = base_cell
    for _ in range(octaves):
        result += amplitude * value_noise_2d(shape, cell, rng)
        total_amplitude += amplitude
        amplitude *= gain
        cell /= lacunarity
    return result / total_amplitude
