# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Visual-only rock shape variety and a merged pebble field.

``rocks.py`` (icosphere + radial displacement) and ``scatter.py`` own the frozen
collision ellipsoids AND consume the one shared RNG that also places every crater and
rock - see config.py's "physics is frozen" note. Nothing in this module may touch that
RNG or those functions: every rock's collision_radii_m in the manifest has to come out
identical to what the pre-existing generator produces for the same seed.

What this module adds instead:

  * Several new, more varied VISUAL mesh archetypes (angular shard, tabular slab, jagged
    fractured blob, alongside a rounder blob like the original) - all built on an
    INDEPENDENT RNG passed in by the caller, never the shared one.
  * Each new mesh is fit, per axis, inside the ORIGINAL variant's frozen
    ``collision_radii`` (shrunk by ``cfg.rock_visual_shrink`` for margin) - see
    ``fit_to_collision_envelope``. This is the "verify, don't assume" requirement in the
    task brief: the fit is exact by construction (target/actual per-axis scale), and
    ``test_rock_visual_meshes_fit_frozen_collision`` checks it directly against the
    shipped OBJs and manifest rather than trusting the arithmetic here.
  * A merged, single-draw-call pebble field: thousands of small (3-10 cm) stones with NO
    collision, seated on the DRAWN surface (not elevation_lookup - they are new objects
    added on top of the ground, not a change to the terrain height field, so the 0.03 m
    surface budget does not apply to them at all). Purely decorative rover-scale ground
    break-up; a non-collidable pebble the rover silently drives over is not a physics
    concern.
"""

from pathlib import Path

import numpy as np
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.rocks import icosphere


def fit_to_collision_envelope(vertices: np.ndarray, target_radii, shrink: float) -> np.ndarray:
    """Fit `vertices` inside the ellipsoid with semi-axes `target_radii`, per-vertex,
    without shrinking the whole mesh uniformly.

    TWO THINGS TRIED AND MEASURED WRONG, worth recording:

    1. Match only the per-axis bounding extremes (max(|x|) == target_radii[0], etc) and
       stop there. A vertex off-axis (e.g. a shard's corner) can satisfy every per-axis
       maximum individually while its ellipsoid-normalized radius
       sqrt((x/rx)^2+(y/ry)^2+(z/rz)^2) still exceeds 1 - it pokes outside the ellipsoid.
       Measured (seed 42, shrink 0.97): 6 of 190 rocks floated, up to 4.5 cm.

    2. Fix that by finding the worst such vertex and uniformly shrinking the WHOLE mesh
       so every vertex clears the ellipsoid. This is a real containment guarantee, but it
       is the wrong fix: rocks.RockVariant's own docstring records that the ellipsoid
       ("the tight axis-aligned bounding half-extents... sits slightly INSIDE the mesh
       along diagonals") is deliberately SMALLER than the original displaced-icosphere
       mesh at diagonal directions - that is what keeps the invisible collision volume
       from poking out past the visible original rock. So a mesh uniformly shrunk to fit
       entirely inside that ellipsoid ends up SMALLER than the original mesh almost
       everywhere, including at the vertex that originally defined the rock's resting
       height (seat_rock_z) - which does not move, because z_m is frozen. Measured: this
       made floating dramatically WORSE (43-176 of 190 rocks, up to 0.5 m) - a uniform
       shrink punishes every vertex, including the ones actually touching the ground, for
       the sake of a handful of outliers.

    The fix that actually measures clean: match the per-axis bounding extremes (so the
    mesh's overall size/depth matches what the ORIGINAL mesh - and thus the frozen
    seating height - assumed), then pull back ONLY the individual vertices that exceed
    the ellipsoid, radially, to its surface. Every other vertex, including whichever one
    is actually lowest for a given rock's rotation, is left untouched. See
    test_rock_visual_meshes_fit_frozen_collision for the containment check this
    guarantees, and PROGRESS.md's terrain-realism-pass note for the measured result
    across all three seeds with this version.
    """
    target = np.asarray(target_radii, dtype=float)
    actual = np.max(np.abs(vertices), axis=0)
    actual = np.where(actual < 1e-9, 1e-9, actual)
    axis_scale = target / actual
    candidate = vertices * axis_scale[None, :]

    ratio = np.sqrt(np.sum((candidate / target[None, :]) ** 2, axis=1))
    excess = ratio > shrink
    if excess.any():
        pull = shrink / ratio[excess]
        candidate[excess] = candidate[excess] * pull[:, None]
    return candidate


def _recenter(vertices: np.ndarray) -> np.ndarray:
    return vertices - vertices.mean(axis=0)


def _blob(rng: np.random.Generator, subdivisions: int = 2) -> tuple:
    """Rounded boulder - the same family the original rocks.py displace_rock produces,
    regenerated here on the independent visual RNG."""
    vertices, faces = icosphere(subdivisions)
    displacement = np.ones(len(vertices))
    for _ in range(4):
        freq = rng.uniform(1.2, 3.0)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        amp = rng.uniform(0.10, 0.34) / 4.0
        wave = (
            np.sin(freq * vertices[:, 0] + phase[0])
            * np.cos(freq * vertices[:, 1] + phase[1])
            * np.sin(freq * vertices[:, 2] + phase[2])
        )
        displacement += amp * wave
    displacement = np.clip(displacement, 0.55, 1.5)
    stretch = rng.uniform(0.6, 1.3, size=3)
    return _recenter(vertices * displacement[:, None] * stretch[None, :]), faces


\
# All four archetypes below stay in the SAME family as the original displace_rock: radial
# noise applied to an icosphere, plus a mild axis stretch - never a hard planar cut or an
# extreme (>~1.3x) axis stretch. That is deliberate, not a missed opportunity for more
# dramatic shapes: a first version of _shard/_slab used random cut PLANES and elongation
# up to 1.9x, and while each individual new mesh still fit its own axis-aligned bounding
# box, its ellipsoid-normalized overshoot (see fit_to_collision_envelope) reached 1.5-1.7x
# at corner vertices, against ~1.0-1.06x for the ORIGINAL displace_rock meshes (measured,
# same seed, both generated fresh - see PROGRESS.md's terrain-realism-pass note). Fitting
# THAT shape's axis extremes to the frozen ellipsoid, while also being asked to reach as
# deep as the original mesh did (the seating height is frozen and does not move), pulled
# corner vertices in hard enough that six of seed 42's rocks floated up to 4.5 cm, and one
# seed-7 rock - a badly-shaped _shard instance - floated 0.66 m once a uniform-shrink fix
# was tried (see fit_to_collision_envelope's docstring for why that fix was itself wrong).
# Staying in the same shape family as the mesh the ellipsoid was measured from keeps the
# overshoot in the same ~1.0-1.1x range the shipped system already tolerates, without a
# geometric conflict between "reach as deep as the original" and "stay inside the
# ellipsoid" for any vertex. Visual variety comes from noise FREQUENCY/character and a
# mild stretch instead.


def _shard(rng: np.random.Generator, subdivisions: int = 1) -> tuple:
    """Angular: low subdivision (flat-shaded facets already read as angular) plus
    higher-frequency, sharper-clipped radial noise than _blob, so facets read as jagged
    rather than smoothly rounded."""
    vertices, faces = icosphere(max(0, subdivisions))
    displacement = np.ones(len(vertices))
    for _ in range(5):
        freq = rng.uniform(2.5, 5.0)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        amp = rng.uniform(0.14, 0.32) / 5.0
        wave = (
            np.sin(freq * vertices[:, 0] + phase[0])
            * np.cos(freq * vertices[:, 1] + phase[1])
            * np.sin(freq * vertices[:, 2] + phase[2])
        )
        displacement += amp * wave
    displacement = np.clip(displacement, 0.55, 1.45)
    stretch = rng.uniform(0.75, 1.3, size=3)
    return _recenter(vertices * displacement[:, None] * stretch[None, :]), faces


def _slab(rng: np.random.Generator, subdivisions: int = 1) -> tuple:
    """Tabular: one axis moderately flattened (not extreme - see module note), blob-like
    radial noise, so it reads as a flatter boulder rather than a perfect sphere."""
    vertices, faces = icosphere(max(0, subdivisions))
    displacement = np.ones(len(vertices))
    for _ in range(4):
        freq = rng.uniform(1.3, 3.0)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        amp = rng.uniform(0.08, 0.22) / 4.0
        wave = (
            np.sin(freq * vertices[:, 0] + phase[0])
            * np.cos(freq * vertices[:, 1] + phase[1])
            * np.sin(freq * vertices[:, 2] + phase[2])
        )
        displacement += amp * wave
    displacement = np.clip(displacement, 0.7, 1.35)
    axis = rng.integers(0, 3)
    stretch = rng.uniform(0.85, 1.2, size=3)
    stretch[axis] = rng.uniform(0.45, 0.62)
    return _recenter(vertices * displacement[:, None] * stretch[None, :]), faces


def _jagged(rng: np.random.Generator, subdivisions: int = 2) -> tuple:
    """Fractured, crumbly-looking blob: like _blob but with much higher-frequency
    multi-octave vertex noise and a fracture cut, for the roughest-looking archetype."""
    vertices, faces = icosphere(subdivisions)
    displacement = np.ones(len(vertices))
    for _ in range(6):
        freq = rng.uniform(3.0, 7.0)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        amp = rng.uniform(0.12, 0.30) / 6.0
        wave = (
            np.sin(freq * vertices[:, 0] + phase[0])
            * np.cos(freq * vertices[:, 1] + phase[1])
            * np.sin(freq * vertices[:, 2] + phase[2])
        )
        displacement += amp * wave
    displacement = np.clip(displacement, 0.55, 1.45)
    stretch = rng.uniform(0.8, 1.25, size=3)
    return _recenter(vertices * displacement[:, None] * stretch[None, :]), faces


_SHAPE_FUNCS = (_blob, _shard, _slab, _jagged)


def _write_obj_merged(path: Path, verts_list, faces_list, name: str) -> dict:
    """Flat-shaded export of several already-placed meshes welded into one OBJ (one draw
    call). Mirrors rocks.write_obj's per-face vertex-copy convention."""
    lines = [f"# Regolith procedural mesh: {name}", f"o {name}"]
    vertex_lines, normal_lines, face_lines = [], [], []
    vi = 0
    for verts, faces in zip(verts_list, faces_list):
        for a, b, c in faces:
            v0, v1, v2 = verts[a], verts[b], verts[c]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal = normal / norm
            normal_lines.append("vn {:.5f} {:.5f} {:.5f}".format(*normal))
            for v in (v0, v1, v2):
                vertex_lines.append("v {:.5f} {:.5f} {:.5f}".format(*v))
            ni = len(normal_lines)
            base = vi + 1
            face_lines.append(f"f {base}//{ni} {base+1}//{ni} {base+2}//{ni}")
            vi += 3
    lines += vertex_lines + normal_lines + face_lines
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return {"vertices": len(vertex_lines), "triangles": len(face_lines)}


def generate_visual_rock_variants(
    output_dir: Path, original_variants: list, rng: np.random.Generator, shrink: float
) -> dict:
    """One new, shape-varied visual mesh per ORIGINAL variant slot, each fit inside that
    variant's own frozen collision ellipsoid. Returns {variant_name: obj_path}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for i, variant in enumerate(original_variants):
        shape_fn = _SHAPE_FUNCS[i % len(_SHAPE_FUNCS)]
        vertices, faces = shape_fn(rng)
        vertices = fit_to_collision_envelope(vertices, variant.collision_radii, shrink)
        name = f"{variant.name}_visual"
        path = output_dir / f"{name}.obj"
        _write_obj_merged(path, [vertices], [faces], name)
        paths[variant.name] = path
    return paths


def rock_albedo_tints(count: int, rng: np.random.Generator, variation: float) -> np.ndarray:
    """Per-instance diffuse multiplier around 1.0, so 190 boulders are not all the
    identical grey - real lunar rocks vary in LIGHTNESS with weathering/dust cover, not
    in hue: the Moon has no coloured rock.

    An earlier version drew an INDEPENDENT +/-variation multiplier per r/g/b channel.
    Mathematically that is a saturation change, not a brightness one, and at
    variation=0.28 it was a large one - measured against a render, boulders came out in
    saturated purple, maroon, teal and blue, an obvious and immediate tell that the
    scene is not real. The fix is to draw ONE scalar per rock and apply it equally to
    every channel, which by construction cannot introduce a colour cast: r/g/b keep the
    same ratio as worldgen's base diffuse, only their shared magnitude moves.
    """
    scalar = 1.0 + rng.uniform(-variation, variation, size=count)
    return np.repeat(scalar[:, None], 3, axis=1)


def _bilinear_sample(surface_yx: np.ndarray, cfg: TerrainConfig, x, y):
    n = surface_yx.shape[0]
    half = cfg.world_size_m / 2.0
    per_m = (n - 1) / cfg.world_size_m
    px = np.clip((x + half) * per_m, 0.0, n - 1.0)
    py = np.clip((y + half) * per_m, 0.0, n - 1.0)
    x0, y0 = int(np.floor(px)), int(np.floor(py))
    x1, y1 = min(x0 + 1, n - 1), min(y0 + 1, n - 1)
    fx, fy = px - x0, py - y0
    z00, z10 = surface_yx[y0, x0], surface_yx[y0, x1]
    z01, z11 = surface_yx[y1, x0], surface_yx[y1, x1]
    return (z00 * (1 - fx) + z10 * fx) * (1 - fy) + (z01 * (1 - fx) + z11 * fx) * fy


def save_pebble_field_obj(path: Path, cfg: TerrainConfig, drawn_surface: np.ndarray, rng: np.random.Generator) -> dict:
    """Thousands of small non-collidable stones seated on the DRAWN surface, welded into
    one OBJ. Not placed against elevation_lookup and not subject to the surface budget -
    see the module docstring for why that is safe."""
    base_verts, base_faces = icosphere(0)  # 12 verts / 20 faces - cheap per instance
    half = cfg.world_size_m / 2.0
    spawn_cx, spawn_cy = cfg.spawn_zone_center

    verts_list, faces_list = [], []
    for _ in range(cfg.pebble_count):
        x = rng.uniform(-half + 0.3, half - 0.3)
        y = rng.uniform(-half + 0.3, half - 0.3)
        if np.hypot(x - spawn_cx, y - spawn_cy) < cfg.spawn_zone_radius_m * 0.9:
            continue
        radius = rng.uniform(cfg.pebble_radius_min_m, cfg.pebble_radius_max_m)
        stretch = rng.uniform(0.55, 1.3, size=3)
        yaw = rng.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        v = (base_verts * stretch[None, :] * radius) @ rot.T

        ground_z = _bilinear_sample(drawn_surface, cfg, x, y)
        lowest = float(v[:, 2].min())
        # Embed a little so it reads as sitting IN the regolith, matching rocks.py's own
        # embed convention rather than perching pebbles on top of the surface.
        z_origin = ground_z - lowest - 0.25 * radius
        verts_list.append(v + np.array([x, y, z_origin]))
        faces_list.append(base_faces)

    stats = _write_obj_merged(path, verts_list, faces_list, "pebble_field")
    stats["count"] = len(verts_list)
    return stats
