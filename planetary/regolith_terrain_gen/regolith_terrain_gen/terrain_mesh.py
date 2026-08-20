# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Export the visual terrain surface as an OBJ mesh.

WHY THE GROUND IS A MESH AND NOT A ``<heightmap>``
--------------------------------------------------
gz renders a ``<heightmap>`` visual through Ogre-Next's **Terra**, a GPU terrain system
with distance-based LOD: a terrain cell far from the camera is built by point-sampling
the heightmap at a stride that grows with range, and interpolating only between the
posts it kept. The rocks standing on that ground are ordinary meshes and take no part
in it, so they keep their exact placement while the ground under them is drawn from a
coarser sample of itself - and at the horizon they visibly hang in the air.

Measured on the shipped surface (0.39 m posts, 190 rocks), reconstructing the ground
from every Nth post the way Terra does:

    stride  post spacing   rocks lifted off   worst gap
         1         0.39 m           0 / 190      seated
         8         3.12 m           0 / 190      seated
        16         6.25 m         2-5 / 190       0.08 m
        32        12.50 m       25-29 / 190       0.47 m
        64        25.00 m       38-64 / 190       1.12 m

That is the "floating rocks" report: it is a *rendering* artefact, which is why three
rounds of increasingly careful placement tests - all of which grade the data on disk -
kept coming back green while the user kept seeing rocks in the sky.

A ``<mesh>`` visual has no LOD in this stack. It is drawn as authored at every range,
so the surface the user sees is the surface the tests measure, and the whole class of
bug goes away rather than being tuned around. The mesh is also written in plain world
coordinates, which retires the transpose convention that caused an earlier round of the
same report (see ``save_heightmap_png``).

``heightmap.png`` is still written and still ships - the costmap reads it as its
elevation source (``regolith_costmap.costmap_node``). It is no longer what gets drawn.
"""

from pathlib import Path

import numpy as np
from regolith_terrain_gen.config import TerrainConfig


def terrain_mesh_posts(surface_yx: np.ndarray, cfg: TerrainConfig, stride: int) -> np.ndarray:
    """Indices of the heightmap posts kept as mesh vertices, along one axis.

    The last post is always kept, so the mesh spans the full world even when `stride`
    does not divide the grid evenly.
    """
    n = surface_yx.shape[0]
    idx = np.arange(0, n, stride)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _grid_lookup(xs: np.ndarray, ys: np.ndarray, z: np.ndarray):
    """elevation(x_m, y_m) over a regular grid of posts, evaluated on the TRIANGLES
    the mesh is built from - two per quad, split on the a->c diagonal (u == v).

    A rock rests on a triangle, not on a bilinear patch between four posts, and the
    difference between those two is precisely the sub-post detail that has hidden
    floating rocks before. Shared by the generator (which seats rocks) and by anything
    reading a shipped terrain.obj back, so the two cannot disagree.
    """

    def elevation(x_m, y_m):
        x = np.clip(np.asarray(x_m, dtype=float), xs[0], xs[-1])
        y = np.clip(np.asarray(y_m, dtype=float), ys[0], ys[-1])
        cx = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
        cy = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, len(ys) - 2)
        u = (x - xs[cx]) / (xs[cx + 1] - xs[cx])
        v = (y - ys[cy]) / (ys[cy + 1] - ys[cy])
        za, zb, zc, zd = z[cy, cx], z[cy, cx + 1], z[cy + 1, cx + 1], z[cy + 1, cx]
        out = np.where(
            v <= u,
            za + (zb - za) * u + (zc - zb) * v,
            za + (zd - za) * v + (zc - zd) * u,
        )
        return float(out) if np.ndim(x_m) == 0 else out

    return elevation


def load_drawn_surface(obj_path: Path):
    """elevation(x_m, y_m) read back from a shipped ``terrain.obj``.

    For anything that has to put an object on the ground *after* generation - the
    mission flags, a debugging probe - without reproducing the generator's maths. The
    OBJ is in world metres, so this needs no manifest, no axis convention and no z
    decode; it is the whole reason the ground ships as a mesh.
    """
    verts = np.array(
        [
            [float(t) for t in line.split()[1:4]]
            for line in Path(obj_path).read_text().splitlines()
            if line.startswith("v ")
        ]
    )
    if not len(verts):
        raise ValueError(f"{obj_path} contains no vertices")
    xs, ys = np.unique(verts[:, 0]), np.unique(verts[:, 1])
    if len(xs) * len(ys) != len(verts):
        raise ValueError(f"{obj_path} is not a regular grid of posts")
    z = np.empty((len(ys), len(xs)))
    z[np.searchsorted(ys, verts[:, 1]), np.searchsorted(xs, verts[:, 0])] = verts[:, 2]
    return _grid_lookup(xs, ys, z)


def mesh_surface_lookup(surface_yx: np.ndarray, cfg: TerrainConfig, stride: int = None):
    """elevation(x_m, y_m) evaluated on the TRIANGLES the terrain mesh is made of.

    Anything seated on the ground has to be seated on the ground that is actually
    drawn. The mesh keeps every `stride`-th post and splits each quad into two
    triangles, so its surface is piecewise-LINEAR over those triangles - not bilinear
    over every heightmap post. Sampling the dense array instead would put a rock up to
    a stride's worth of relief off the drawn surface, which is exactly the class of
    mismatch that produced the last two rounds of floating rocks.
    """
    stride = cfg.terrain_mesh_stride if stride is None else stride
    idx = terrain_mesh_posts(surface_yx, cfg, stride)
    n = surface_yx.shape[0]
    coords = -cfg.world_size_m / 2.0 + idx / ((n - 1) / cfg.world_size_m)
    # Same evaluator load_drawn_surface uses, so seating a rock during generation and
    # reading the shipped OBJ back later cannot give two different answers.
    return _grid_lookup(coords, coords, surface_yx[np.ix_(idx, idx)])


def save_terrain_mesh_obj(
    surface_yx: np.ndarray, cfg: TerrainConfig, path: Path, stride: int = None
) -> dict:
    """Write the visual surface as an OBJ in world coordinates.

    `surface_yx` is indexed [row = y, col = x] in absolute metres, exactly as
    ``build_heightmap`` returns it. Vertices carry their own world x/y, so nothing
    downstream has to know an axis convention.

    Texture coordinates tile the surface textures every ``cfg.texture_tile_size_m``,
    which is what the ``<heightmap><texture><size>`` element used to do.
    """
    stride = cfg.terrain_mesh_stride if stride is None else stride
    idx = terrain_mesh_posts(surface_yx, cfg, stride)
    n = surface_yx.shape[0]
    half = cfg.world_size_m / 2.0
    per_m = (n - 1) / cfg.world_size_m

    coords = -half + idx / per_m  # world metres of each retained post, both axes
    z = surface_yx[np.ix_(idx, idx)]  # [row = y, col = x]
    rows = cols = len(idx)

    xs = np.broadcast_to(coords[None, :], (rows, cols))
    ys = np.broadcast_to(coords[:, None], (rows, cols))

    # Per-vertex normals from the surface gradient, so the ground shades smoothly
    # instead of showing every triangle. np.gradient handles the uneven final step.
    dzdx = np.gradient(z, coords, axis=1)
    dzdy = np.gradient(z, coords, axis=0)
    normals = np.stack([-dzdx, -dzdy, np.ones_like(z)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    uv = coords / cfg.texture_tile_size_m

    lines = [
        "# Regolith lunar terrain - drawn surface, world coordinates, metres.",
        "# Generated by regolith_terrain_gen.terrain_mesh; see that module for why the",
        "# ground ships as a mesh rather than a <heightmap> visual.",
        f"o terrain",
    ]
    lines += [
        "v {:.4f} {:.4f} {:.4f}".format(x, y, h)
        for x, y, h in zip(xs.ravel(), ys.ravel(), z.ravel())
    ]
    lines += ["vt {:.5f} {:.5f}".format(u, v) for v in uv for u in uv]
    lines += ["vn {:.4f} {:.4f} {:.4f}".format(*nrm) for nrm in normals.reshape(-1, 3)]

    # Two triangles per quad, counter-clockwise seen from +z so the lit side faces up.
    r0 = np.arange(rows - 1)[:, None] * cols
    r1 = r0 + cols
    c0 = np.arange(cols - 1)[None, :]
    c1 = c0 + 1
    a = (r0 + c0).ravel() + 1  # OBJ indices are 1-based
    b = (r0 + c1).ravel() + 1
    c = (r1 + c1).ravel() + 1
    d = (r1 + c0).ravel() + 1
    faces = np.empty((len(a) * 2, 3), dtype=np.int64)
    faces[0::2] = np.stack([a, b, c], axis=1)
    faces[1::2] = np.stack([a, c, d], axis=1)
    lines += ["f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}".format(*f) for f in faces]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")

    return {
        "posts": int(rows),
        "stride": int(stride),
        "post_spacing_m": float(coords[1] - coords[0]),
        "triangles": int(len(faces)),
        "vertices": int(rows * cols),
    }
