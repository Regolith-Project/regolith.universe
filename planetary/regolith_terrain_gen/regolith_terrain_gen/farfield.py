# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Visual-only far-field terrain: extends the ground far beyond the 200 m physics
footprint so it reads as a horizon meeting the sky, instead of stopping dead at a
hard edge.

FROZEN PHYSICS, VISUAL-ONLY GEOMETRY
-------------------------------------
Everything here produces a `<visual>`-only static model with no `<collision>`,
built and written entirely separately from build_heightmap/build_terrain_collision_
boxes_sdf/write_manifest. It never touches the physics terrain, the rng stream
those consume, or manifest.json - the costmap and planner, which only ever read
the manifest and the collision geometry, cannot see this mesh exists.

HOW IT JOINS THE PHYSICS TERRAIN WITHOUT A SEAM
-------------------------------------------------
The physics world is a 200 m square (|x|,|y| <= world_size_m/2). This mesh is a
sequence of concentric SQUARE "shells" (like picture-frame mats nested inside
each other) starting exactly on that boundary and growing outward to
`farfield_outer_radius_m`. Shell spacing grows with a power curve (`_SHELL_POWER`)
so shells are densely packed near the boundary (for a smooth blend) and sparse far
out (where fine detail is invisible anyway, so it would only cost triangles).

Every shell uses the *same* angular parametrisation (`_square_ring_xy`), so shell
i's j-th vertex sits on the straight radial line from the origin through shell
0's j-th vertex. Shell 0's vertices are, by construction, exactly on the physics
boundary, so whatever `elevation_lookup` the caller passes in (generate.py hands
in a lookup built from the actual drawn/rendered surface - see build_farfield_mesh's
own docstring) gives their true height directly - zero z-discontinuity at the seam
by construction, not by tuning. Moving outward from shell 0, height blends from a
short linear extrapolation of the boundary
(matching the physics terrain's own local slope for the first
`farfield_blend_width_m` or so) to an independent large-scale noise field, via a
smoothstep - so the join is not just value-continuous at d=0, it stays close to
the real terrain's slope for a few metres past the edge too, rather than kinking
immediately into unrelated noise.

Farther out, `_far_relief` (a coarse fbm field, amplitude `farfield_relief_m`) and
a handful of large synthetic crater-rim mounds (`farfield_crater_count`) give the
horizon silhouette interest - hills and rim shapes to break the skyline, per the
brief. The outermost shells are curved downward (`farfield_horizon_drop_m`) so the
mesh's own far edge sinks out of view instead of showing as a hard rectangular
line against the sky - similar in spirit to how a real horizon drops away with
distance on a curved body, and it means the far boundary never needs to be inside
every camera's frustum to look right.
"""

from pathlib import Path

import numpy as np
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.noise import fbm

_SHELL_POWER = 3.0
_INSET_M = 2.0  # how far inward of the boundary the slope-matching sample is taken


def _square_ring_xy(half_width: float, segments_per_side: int) -> tuple:
    """(x, y) of every vertex on a square ring of the given half-width, `4 * segments_per_side` points, walked CCW starting at (+half_width, -half_width).

    Every shell is built with this same angular indexing (scalar `half_width`
    per call), so shell i's vertex j is always the radial projection of shell
    0's vertex j through the origin - see module docstring.
    """
    m = segments_per_side
    k = np.arange(4 * m)
    side = k // m
    t = (k % m) / m
    local = half_width * (2.0 * t - 1.0)
    x = np.select(
        [side == 0, side == 1, side == 2, side == 3],
        [half_width, -local, -half_width, local],
    )
    y = np.select(
        [side == 0, side == 1, side == 2, side == 3],
        [local, half_width, -local, -half_width],
    )
    return x, y


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _far_relief_field(cfg: TerrainConfig, rng: np.random.Generator, outer_half: float):
    """A coarse, independent large-scale height field over the whole far-field square, sampled by bilinear interpolation - cheap, smooth, and deterministic per-seed without ever touching the terrain generator's own rng stream (see module docstring)."""
    res = 192
    grid = fbm((res, res), rng, octaves=5, base_cell=max(8.0, res / 10.0), lacunarity=2.1, gain=0.55)

    def sample(x, y):
        # grid[row=y, col=x], both in [0, res-1], matching every other module here.
        fx = (np.asarray(x) + outer_half) / (2.0 * outer_half) * (res - 1)
        fy = (np.asarray(y) + outer_half) / (2.0 * outer_half) * (res - 1)
        fx = np.clip(fx, 0.0, res - 1)
        fy = np.clip(fy, 0.0, res - 1)
        x0 = np.floor(fx).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, res - 1)
        y1 = np.clip(y0 + 1, 0, res - 1)
        tx = fx - x0
        ty = fy - y0
        a = grid[y0, x0] * (1 - tx) + grid[y0, x1] * tx
        b = grid[y1, x0] * (1 - tx) + grid[y1, x1] * tx
        return a * (1 - ty) + b * ty

    return sample


def _add_crater_rims(x, y, h, cfg: TerrainConfig, rng: np.random.Generator, inner_half, outer_half):
    """Add a handful of large rim-shaped mounds for horizon silhouette interest - the far-field analogue of the near terrain's crater field, at a scale (hundreds of metres) and count (a few dozen) appropriate to what would actually read as distinct shapes on a multi-km horizon.

    Each rim is also given a coarse angular "ridge/terrace" wobble (`_rim_angular_profile`)
    instead of a perfectly smooth ring, and its height is scaled by the same
    distance ramp `_relief_ramp` uses - close-in craters (just past the blend band)
    stay LOW, full-height rims only appear well out towards the horizon. A first
    pass put full-height (up to ~150 m) rims right past the 100 m physics edge,
    which read as a close-in wall/overhang rather than a receding horizon.
    """
    n = cfg.farfield_crater_count
    if n <= 0:
        return h
    cx = rng.uniform(-outer_half, outer_half, n)
    cy = rng.uniform(-outer_half, outer_half, n)
    # Keep centres out past the near blend band so they never interfere with the
    # boundary seam.
    far_enough = np.hypot(cx, cy) > (inner_half + cfg.farfield_blend_width_m * 3.0)
    cx, cy = cx[far_enough], cy[far_enough]
    diam = rng.uniform(220.0, 850.0, cx.shape[0])
    rim_h = rng.uniform(0.45, 1.15, cx.shape[0]) * cfg.farfield_relief_m * 0.6
    phase = rng.uniform(0.0, 2.0 * np.pi, cx.shape[0])
    lobes = rng.integers(4, 9, cx.shape[0])
    ramp = _relief_ramp(np.hypot(cx, cy) - inner_half, cfg)
    for i in range(cx.shape[0]):
        dx, dy = x - cx[i], y - cy[i]
        r = np.hypot(dx, dy)
        rad = diam[i] / 2.0
        # Angular wobble so the rim reads as an irregular arc/terrace rather than
        # a perfect circle - real crater rims are not round.
        wobble = 1.0 + 0.30 * np.cos(lobes[i] * np.arctan2(dy, dx) + phase[i])
        # Raised rim near the crater's edge, falling off both inward and outward -
        # the same qualitative shape craters.py sculpts up close, simplified since
        # this only ever needs to read correctly in silhouette. Narrower than a
        # first pass (0.30 -> 0.22 of the radius) for a crisper, more angular rim
        # edge at horizon scale - a wide Gaussian read as a soft dune, not a rim.
        h = h + rim_h[i] * ramp[i] * wobble * np.exp(
            -(((r - 0.88 * rad) / (0.22 * rad)) ** 2)
        )
    return h


_RELIEF_FLOOR = 0.05


def _relief_ramp(d_from_boundary: np.ndarray, cfg: TerrainConfig) -> np.ndarray:
    """`_RELIEF_FLOOR` just past the blend band, ramping smoothly to 1 by `farfield_relief_ramp_m` further out.

    Applied to both the coarse noise field and the crater rims so relief height
    grows with distance instead of standing at full amplitude right past the 100 m
    physics edge - full-height hills a few hundred metres out read as a close-in
    wall/overhang, not a horizon (see module docstring / PROGRESS.md for the
    render this was diagnosed from).

    0.05, not 0 or the 0.25 a previous pass used. The history here matters:
    - An original pass ramped from literally 0. Before `noise`'s base level was
      anchored to h_edge (see build_farfield_mesh), that base was a flat global
      value, so ramp=0 meant a long, dead-flat trough at the WRONG height -
      visible as a gap from an elevated camera.
    - Anchoring to h_edge fixed that: at ramp=0, `noise` now equals `h_edge`
      exactly, which is where `extrap` (the slope-matching term) is ALSO headed
      at that same distance - i.e. ramp=0 is already seam-continuous, no trough,
      by construction. That made a large floor unnecessary, but a later pass
      still had it at 0.25 - which turned out to be actively harmful: at
      d=blend_width, ramp is only ~floor, so up to `_far_relief_m * floor`
      (55 m at floor=0.25) of UNCORRELATED random noise leaks into the height
      right at the seam, independent of local terrain shape. Traced (by
      printing the actual per-shell profile) to a real, if modest and smooth,
      dip-then-recover a few shells past a seam-adjacent local peak - not a
      cliff (the silhouette floor below already bounds those), but still a
      visible gap of sky from an elevated camera, because the near terrain's
      own profile keeps climbing right up to the boundary while this dip
      immediately follows it. 0.05 keeps a trace of independent variation
      (so the immediate post-seam terrain is not perfectly flat) while
      making that leak small enough to matter far less.
    """
    start = cfg.farfield_blend_width_m
    span = max(1e-6, cfg.farfield_relief_ramp_m - start)
    t = _smoothstep((np.asarray(d_from_boundary) - start) / span)
    return _RELIEF_FLOOR + (1.0 - _RELIEF_FLOOR) * t


def build_farfield_mesh(cfg: TerrainConfig, elevation_lookup, path: Path) -> dict:
    """Write the far-field visual mesh as an OBJ in world coordinates and return stats.

    `elevation_lookup(x, y)` must be whatever surface has to match at the seam -
    generate.py passes a lookup built from `drawn_surface` (the surface actually
    rendered by terrain_mesh_obj, see terrain_mesh.mesh_surface_lookup), not the
    coarser collision surface, so the far field meets exactly what is on screen at
    the boundary. See module docstring for the join maths.
    """
    inner_half = cfg.world_size_m / 2.0
    outer_half = cfg.farfield_outer_radius_m
    n_shells = cfg.farfield_shell_count
    m = cfg.farfield_segments_per_side

    frac = np.linspace(0.0, 1.0, n_shells)
    half_widths = inner_half + (outer_half - inner_half) * frac**_SHELL_POWER

    x0, y0 = _square_ring_xy(half_widths[0], m)
    h0 = np.asarray(elevation_lookup(x0, y0), dtype=np.float64)
    r0 = np.hypot(x0, y0)
    r0_safe = np.where(r0 > 1e-6, r0, 1.0)
    ux, uy = x0 / r0_safe, y0 / r0_safe
    xin = x0 - ux * _INSET_M
    yin = y0 - uy * _INSET_M
    hin = np.asarray(elevation_lookup(xin, yin), dtype=np.float64)
    grad_out = (h0 - hin) / _INSET_M  # d(height)/d(outward distance) at the boundary

    rng = np.random.default_rng((int(cfg.seed) * 1_000_003) ^ 0x5EAF1E1D)
    far_relief = _far_relief_field(cfg, rng, outer_half)

    # Height `extrap` actually reaches at d == blend_width, i.e. exactly where the
    # (1-w)/w crossfade below finishes handing off from `extrap` to `noise`. This,
    # not the raw boundary height h0, is `noise`'s local anchor - see the fix note
    # below for why using h0 directly left a real value jump at the blend edge.
    h_edge = h0 + grad_out * cfg.farfield_blend_width_m
    global_mean_edge = float(np.mean(h_edge))
    local_bias = h_edge - global_mean_edge

    curve_start = 0.55
    xs = np.empty((n_shells, 4 * m))
    ys = np.empty((n_shells, 4 * m))
    hs = np.empty((n_shells, 4 * m))  # pre-drop height, for the silhouette-floor pass below
    ds = np.empty((n_shells, 4 * m))  # distance from the boundary, same - for that pass too
    for i in range(n_shells):
        xi, yi = _square_ring_xy(half_widths[i], m)
        d = np.hypot(xi - x0, yi - y0)
        ds[i] = d
        w = _smoothstep(d / cfg.farfield_blend_width_m)
        extrap = h0 + grad_out * d
        # `d` is already the raw distance from the boundary (same quantity
        # _add_crater_rims passes) - _relief_ramp does its own "- blend_width"
        # internally, so passing d directly here (not d - blend_width again) is
        # what actually makes the ramp start at the blend edge, not 2x further
        # out. A double subtraction here was a second, independent bug from the
        # value-jump one above - harmless on its own (just shifted the ramp's
        # start/full-amplitude distances outward) but fixed alongside it.
        ramp = _relief_ramp(d, cfg)
        # Base level the noise rides on tracks THIS point's own extrapolated
        # edge height (local_bias, anchored at h_edge - see above) near the seam,
        # fading to the boundary's global mean only as relief ramps in (same
        # `ramp`) - not a flat jump to the global mean right past the blend band.
        # TWO bugs stacked here, both real, both visible as a black gap-with-stars
        # between the near terrain and the far field from an elevated camera
        # (never visible from ground level, which is why it passed earlier
        # ground-level renders - the gap sits below the horizon from there).
        # (1) anchoring to h0 (the raw boundary height) instead of h_edge (what
        # `extrap` actually reaches at d=blend_width): since w reaches 1.0
        # (noise fully replaces extrap) exactly at d=blend_width, anchoring to h0
        # while extrap had already moved away from it by grad_out*blend_width
        # left a real value jump AT the blend edge on every boundary point with a
        # nonzero outward slope - the larger of the two. (2) flattening straight
        # to a global mean right past the blend, with no local tracking at all,
        # opened a further, smaller gap at any LOCAL HIGH point on the boundary
        # (e.g. a crater rim near the edge). Both help but do NOT fully close the
        # gap on their own - the noise field itself can still dip below a nearby
        # peak (real terrain is not monotonic) - see the silhouette floor below.
        base = h_edge - ramp * local_bias  # == h_edge*(1-ramp) + global_mean_edge*ramp
        noise = far_relief(xi, yi) * cfg.farfield_relief_m * ramp + base
        xs[i], ys[i], hs[i] = xi, yi, (1.0 - w) * extrap + w * noise

    hs = _add_crater_rims(xs, ys, hs, cfg, rng, inner_half, outer_half)
    # Re-apply the boundary row exactly (crater rims are kept clear of it already,
    # but this keeps the seam exact even if a future tuning shrinks that margin).
    hs[0] = h0

    # SILHOUETTE FLOOR: never let a shell's height fall more than `allowed_dip(d)`
    # below the running maximum height already reached at that same angular
    # position (same k - same radial direction from the origin). Without this,
    # real noise-field variation can rise near the seam and then genuinely dip -
    # not a bug in any single formula above, just ordinary terrain variation -
    # and from an elevated camera that dip can permanently fall below the angle
    # a nearer peak already established, showing as a gap of visible sky that
    # nothing further out ever closes (confirmed by measuring the actual
    # elevation-angle profile from the `orbit` preset's camera: it rises for
    # several shells past the seam, then falls and never recovers).
    #
    # `allowed_dip` GROWS with distance from the boundary (a constant metres-of-
    # dip budget, tried first, has to either stay tight - which just relocates
    # the same cliff to wherever the budget runs out, confirmed by rendering:
    # shells pinned dead flat at the budget's ceiling, then an abrupt ~40 m drop
    # the moment the clamp stopped applying - or be loosened by hand-picking
    # where it turns off, which is exactly the fragile per-seed tuning a real
    # fix should not need). A real height dip of a given size matters less, the
    # farther away it is - its angular size from any given camera shrinks
    # roughly as 1/distance - so scaling the allowed dip with distance keeps
    # its VISUAL (angular) impact roughly constant instead of its metric size,
    # and never needs an explicit on/off boundary: applied uniformly to every
    # shell, it naturally stops being the binding constraint once real relief
    # variation (or the deliberate horizon_drop, added after this) exceeds it.
    allowed_dip = cfg.farfield_max_dip_m + ds * np.tan(np.deg2rad(cfg.farfield_max_dip_slope_deg))
    running_max = hs[0].copy()
    for i in range(1, n_shells):
        hs[i] = np.maximum(hs[i], running_max - allowed_dip[i])
        running_max = np.maximum(running_max, hs[i])

    t_drop = _smoothstep((frac[:, None] - curve_start) / max(1e-6, 1.0 - curve_start))
    drop = -(t_drop**2) * cfg.farfield_horizon_drop_m
    zs = hs + drop

    verts = np.stack([xs, ys, zs], axis=-1)  # (n_shells, 4m, 3)

    # Per-vertex normals: average the geometric normal of every triangle touching
    # a vertex. Robust to the shell/angular grid not being axis-aligned (unlike
    # terrain_mesh.py's Cartesian np.gradient approach, which assumes exactly that).
    rows, cols = n_shells, 4 * m
    normals = np.zeros_like(verts)

    def _accumulate(a_idx, b_idx, c_idx):
        a, b, c = verts[a_idx], verts[b_idx], verts[c_idx]
        fn = np.cross(b - a, c - a)
        for idx in (a_idx, b_idx, c_idx):
            np.add.at(normals, idx, fn)

    r0i, r1i = np.arange(rows - 1)[:, None], np.arange(rows - 1)[:, None] + 1
    c0i, c1i = np.arange(cols)[None, :], (np.arange(cols)[None, :] + 1) % cols
    a_idx = (np.broadcast_to(r0i, (rows - 1, cols)), np.broadcast_to(c0i, (rows - 1, cols)))
    b_idx = (np.broadcast_to(r0i, (rows - 1, cols)), np.broadcast_to(c1i, (rows - 1, cols)))
    c_idx = (np.broadcast_to(r1i, (rows - 1, cols)), np.broadcast_to(c1i, (rows - 1, cols)))
    d_idx = (np.broadcast_to(r1i, (rows - 1, cols)), np.broadcast_to(c0i, (rows - 1, cols)))
    _accumulate(a_idx, b_idx, c_idx)
    _accumulate(a_idx, c_idx, d_idx)

    norm_len = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm_len = np.where(norm_len > 1e-9, norm_len, 1.0)
    normals /= norm_len
    flip = normals[..., 2] < 0.0
    normals[flip] *= -1.0

    lines = [
        "# Regolith lunar terrain - visual-only far-field horizon extension.",
        "# Generated by regolith_terrain_gen.farfield; joins the physics terrain",
        "# exactly at |x|,|y| == world_size_m/2 (see that module for the maths).",
        "# NOT collision geometry, NOT read by the costmap/planner.",
        "o farfield",
    ]
    flat_v = verts.reshape(-1, 3)
    flat_n = normals.reshape(-1, 3)
    # UV in the SAME world-space frequency as the near terrain's own 1:1 bake
    # (terrain_mesh.save_terrain_mesh_obj's uv_world_size_m path: u = x/world_size_m
    # + 0.5), so the apparent grain size does not jump at the seam. The far field
    # spans up to +/-15 tile periods from the origin (outer_half ~3000 m over a
    # 200 m tile), so this MUST repeat, not clamp - a first pass left u/v
    # unwrapped (raw values up to +/-15) and the far field rendered as a flat,
    # near-white, texture-less mass: this Ogre2/gz-rendering build evidently
    # clamps PBR map addressing outside [0, 1] rather than defaulting to wrap, so
    # everything past the first tile sampled one fixed edge texel. `np.mod` here
    # forces the wrap explicitly rather than relying on any renderer default.
    # Trade-off, accepted: a triangle that straddles a tile boundary gets a UV
    # seam (interpolates across the whole texture instead of wrapping) - far
    # preferable to zero detail everywhere, and `farfield_texture_tile_scale`
    # (below) both softens it and makes it rarer.
    #
    # `farfield_texture_tile_scale` > 1 stretches that same bake over a larger
    # area than the near terrain's own 1:1 mapping: a first pass tiled at the
    # near terrain's exact 200 m frequency, which read as "crumpled foil" once
    # actually visible (after the wrap fix) - fine detail (ejecta rays, sub-
    # metre noise) that reads correctly at near-field range is exactly the wrong
    # spatial frequency once repeated across a multi-km horizon, aliasing into a
    # high-contrast marbled/creased look under minification. Distant terrain
    # should be low-contrast and large-scale (crater-scale structure, not fine
    # noise) - stretching the UV is a cheap stand-in for a proper distance-based
    # LOD/blur the asset pipeline does not otherwise have.
    half_world = cfg.world_size_m * cfg.farfield_texture_tile_scale
    flat_uv = np.mod(flat_v[:, :2] / half_world + 0.5, 1.0)
    lines += ["v {:.3f} {:.3f} {:.3f}".format(*v) for v in flat_v]
    lines += ["vt {:.5f} {:.5f}".format(*uv) for uv in flat_uv]
    lines += ["vn {:.4f} {:.4f} {:.4f}".format(*n) for n in flat_n]

    a = (a_idx[0] * cols + a_idx[1]).ravel() + 1  # 1-based OBJ indices
    b = (b_idx[0] * cols + b_idx[1]).ravel() + 1
    c = (c_idx[0] * cols + c_idx[1]).ravel() + 1
    d = (d_idx[0] * cols + d_idx[1]).ravel() + 1
    lines += ["f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}".format(*f) for f in np.stack([a, b, c], axis=1)]
    lines += ["f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}".format(*f) for f in np.stack([a, c, d], axis=1)]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")

    return {
        "shells": int(n_shells),
        "segments_per_side": int(m),
        "vertices": int(flat_v.shape[0]),
        "triangles": int(2 * (rows - 1) * cols),
        "inner_half_width_m": float(inner_half),
        "outer_half_width_m": float(outer_half),
        "boundary_max_abs_error_m": 0.0,  # exact by construction; see test_farfield_boundary_match.py
    }


def farfield_model_sdf(cfg: TerrainConfig, mesh_path: Path, texture_pngs: dict = None) -> str:
    """`<model>` block for the far-field terrain.

    REVERTED from reusing the near terrain's albedo/roughness texture. That
    looked right in isolation (matched tone/detail statistics to the near
    ground) but reads badly once actually seen tiled across a multi-km horizon:
    fine detail authored for close-range viewing (crater-ejecta rays, per-metre
    noise) does not survive being minified this far - it aliases into a high-
    contrast marbled/creased "crumpled foil" look with visible stair-step
    terracing, confirmed across two independent review passes and not fixed by
    softening the tile frequency alone. `texture_pngs` is still accepted (so
    worldgen.py's call site does not need to change) but is now IGNORED - kept
    as a documented dead parameter rather than removed, in case a properly
    distance-blurred/LOD'd texture is worth revisiting later.

    Flat, near-terrain-toned colour instead. The FIRST value tried here (0.42/
    0.40/0.38) was picked from textures_world.py's formula for its near-field
    base grey BEFORE that formula's own darkening terms (slope/concavity/dust-
    streak darkening) - reasonable on paper, wrong in practice: measuring the
    near terrain's actual baked albedo bake gives a much brighter mean (~0.78)
    than that formula alone suggests, and yet the near terrain still renders
    visibly DARKER on screen than a flat 0.42 far field under `cine_light` -
    because the near terrain's fine bump/crater microtexture self-shadows
    heavily under a low sun, while a perfectly flat, un-textured far field has
    no microtexture to self-shadow at all and shows its raw diffuse response
    directly. Matching a flat material's tone to a richly self-shadowed one by
    formula does not work; this value was picked by direct A/B comparison of
    actual `cine_light` renders (the far field read as a distinctly brighter,
    warmer "beige" band above a darker grey near field at the first value) and
    darkened until the two read as continuous.

    "Broad tonal variation and crater-scale structure" - the actual brief for
    how distant terrain should look - comes from the geometry itself: real
    per-vertex normals off the hills/crater-rim relief (see
    build_farfield_mesh) under the scene's directional sun, which is
    inherently low-frequency/large-scale in a way a reused near-field texture
    is not.
    """
    material_body = """<diffuse>0.13 0.125 0.12 1</diffuse>
            <specular>0.02 0.02 0.02 1</specular>
            <pbr>
              <metal>
                <roughness>0.97</roughness>
                <metalness>0.0</metalness>
              </metal>
            </pbr>"""
    return f"""    <model name="farfield_terrain">
      <static>true</static>
      <link name="link">
        <!-- Visual only: no collision geometry. Not written to manifest.json - see the
             module docstring for why the costmap/planner cannot see this. -->
        <visual name="farfield_visual">
          <!-- Off, deliberately: under the shipped 12 deg sun, relief on this
               scale (up to farfield_relief_m tall, out to several km) can throw
               shadows many hundreds of metres long (length ~ height / tan(sun
               elevation)) - long enough to reach back across the physics terrain
               depending on sun azimuth and hill placement. Self-shading from
               surface normals already gives the distant hills/rims a silhouette
               under direct light; this just keeps that light from ever painting
               an unrelated shadow band across the near terrain. -->
          <cast_shadows>false</cast_shadows>
          <geometry>
            <mesh><uri>file://{mesh_path}</uri></mesh>
          </geometry>
          <material>
            {material_body}
          </material>
        </visual>
      </link>
    </model>
"""
