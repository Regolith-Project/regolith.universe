# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Low-poly procedural rock meshes: an icosphere, subdivided and displaced into an
irregular boulder, exported as flat-shaded OBJ (no external mesh library needed)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PHI = (1.0 + 5.0**0.5) / 2.0

_ICOSAHEDRON_VERTICES = np.array(
    [
        [-1, _PHI, 0], [1, _PHI, 0], [-1, -_PHI, 0], [1, -_PHI, 0],
        [0, -1, _PHI], [0, 1, _PHI], [0, -1, -_PHI], [0, 1, -_PHI],
        [_PHI, 0, -1], [_PHI, 0, 1], [-_PHI, 0, -1], [-_PHI, 0, 1],
    ],
    dtype=np.float64,
)
_ICOSAHEDRON_VERTICES /= np.linalg.norm(_ICOSAHEDRON_VERTICES[0])

_ICOSAHEDRON_FACES = [
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
]


def _subdivide(vertices: list, faces: list) -> tuple:
    midpoint_cache = {}

    def midpoint(i: int, j: int) -> int:
        key = tuple(sorted((i, j)))
        if key in midpoint_cache:
            return midpoint_cache[key]
        mid = (vertices[i] + vertices[j]) / 2.0
        mid /= np.linalg.norm(mid)
        vertices.append(mid)
        idx = len(vertices) - 1
        midpoint_cache[key] = idx
        return idx

    new_faces = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return vertices, new_faces


def icosphere(subdivisions: int) -> tuple:
    vertices = [v.copy() for v in _ICOSAHEDRON_VERTICES]
    faces = list(_ICOSAHEDRON_FACES)
    for _ in range(subdivisions):
        vertices, faces = _subdivide(vertices, faces)
    return np.array(vertices), faces


def displace_rock(vertices: np.ndarray, rng: np.random.Generator, roughness: float = 0.32) -> np.ndarray:
    """Smooth pseudo-random radial displacement (sum of a few random-phase standing waves) —
    a lightweight stand-in for 3D noise, adequate at this vertex count."""
    displacement = np.ones(len(vertices))
    for _ in range(4):
        freq = rng.uniform(1.3, 3.2)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        amp = rng.uniform(0.08, roughness) / 4.0
        wave = (
            np.sin(freq * vertices[:, 0] + phase[0])
            * np.cos(freq * vertices[:, 1] + phase[1])
            * np.sin(freq * vertices[:, 2] + phase[2])
        )
        displacement += amp * wave
    displacement = np.clip(displacement, 0.55, 1.5)

    # Random anisotropic stretch so rocks aren't all lumpy spheres.
    stretch = rng.uniform(0.55, 1.35, size=3)
    return vertices * displacement[:, None] * stretch[None, :]


def generate_rock_variant(rng: np.random.Generator, subdivisions: int = 1) -> tuple:
    vertices, faces = icosphere(subdivisions)
    vertices = displace_rock(vertices, rng)
    # Re-center and normalize so every variant fits a unit bounding radius before scaling at placement time.
    vertices -= vertices.mean(axis=0)
    radius = np.max(np.linalg.norm(vertices, axis=1))
    vertices /= radius
    return vertices, faces


def write_obj(path: Path, vertices: np.ndarray, faces: list, name: str) -> None:
    """Flat-shaded export: each face gets its own vertex copies and one face normal,
    which reads as a faceted low-poly rock rather than a smoothed blob."""
    lines = [f"# Regolith procedural rock variant: {name}", f"o {name}"]
    vertex_lines, normal_lines, face_lines = [], [], []
    vi = 1
    for a, b, c in faces:
        v0, v1, v2 = vertices[a], vertices[b], vertices[c]
        normal = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
        normal_lines.append("vn {:.5f} {:.5f} {:.5f}".format(*normal))
        for v in (v0, v1, v2):
            vertex_lines.append("v {:.5f} {:.5f} {:.5f}".format(*v))
        ni = len(normal_lines)
        face_lines.append(f"f {vi}//{ni} {vi+1}//{ni} {vi+2}//{ni}")
        vi += 3
    lines += vertex_lines + normal_lines + face_lines
    path.write_text("\n".join(lines) + "\n")


@dataclass
class RockVariant:
    """A generated rock mesh, plus what placement and collision need from its geometry.

    ``vertices`` (not just the name) is what lets scatter_rocks compute where a rock's
    actual underside is: normalization above divides by the BOUNDING RADIUS, but
    displace_rock's anisotropic stretch means a variant's lowest vertex sits anywhere
    from ~0.51 to 1.0 units below the mesh origin, so no single constant offset can seat
    every variant on the ground. See scatter.seat_rock_z.

    ``collision_radii`` is the ellipsoid that stands in for the mesh as COLLISION
    geometry. <mesh> collision silently does nothing in this gz-sim 8 / gz-physics 7.8.0
    install - the same dartsim limitation heightmap.py records for terrain - which was
    verified here directly: a probe dropped onto a rock-mesh collision falls straight
    through it (final z -38.3 m vs 3.1 m for a box control). So the shipped rocks were
    decoration the rover drove through, while the costmap planned around them.
    This is purely a correctness fix and buys no speed: measured interleaved in one
    session at res24, 190 mesh rocks give RTF 0.499 and 190 ellipsoids 0.488, against
    0.568 with no rock collisions at all - i.e. the rocks cost ~12% either way, and mesh
    vs ellipsoid is within noise. (An earlier session recorded mesh rocks at 0.134 and
    concluded they burned ~70% of the budget; that figure does not reproduce and the
    conclusion drawn from it was wrong.) An ellipsoid is used rather than a sphere because these boulders are
    deliberately anisotropic; a sphere at the bounding radius would stick out well past
    the visible rock on its thin axes.
    """

    name: str
    vertices: np.ndarray
    collision_radii: np.ndarray


def fit_collision_ellipsoid(vertices: np.ndarray) -> np.ndarray:
    """Semi-axes of an ellipsoid approximating this rock, in unit-mesh coordinates.

    Per-axis max |component|, i.e. the tight axis-aligned bounding half-extents. For an
    undisplaced icosphere this is exactly the unit sphere; for the displaced, stretched
    blobs it hugs the silhouette and sits slightly INSIDE the mesh along diagonals -
    the direction to err, since a collision shape poking out past the visible rock
    would stop the rover against thin air.
    """
    return np.max(np.abs(vertices), axis=0)


def generate_rock_variants(output_dir: Path, count: int, rng: np.random.Generator, subdivisions: int = 1) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for i in range(count):
        vertices, faces = generate_rock_variant(rng, subdivisions)
        name = f"rock_{i}"
        write_obj(output_dir / f"{name}.obj", vertices, faces, name)
        variants.append(
            RockVariant(name=name, vertices=vertices, collision_radii=fit_collision_ellipsoid(vertices))
        )
    return variants
