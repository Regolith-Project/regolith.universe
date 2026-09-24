# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Procedural visual meshes + PBR textures for the Regolith rover, in the same spirit as
regolith_terrain_gen/rocks.py: hand-built OBJ export, numpy + PIL only, no mesh library
and no new pip dependency.

This is VISUAL-ONLY tooling. Nothing here may be read by physics: the meshes it writes
are wired into the URDF purely as extra <visual> geometry / replacement <visual>
geometry on the existing chassis and wheel links, never into <collision> or <inertial>.
See regolith_rover_description/urdf/regolith_rover.urdf.xacro for how these files are
consumed, and its header comment for the frozen-physics constraint this all lives under.

Why the parts sharing one link end up sharing one texture atlas each (chassis parts share
"chassis_atlas.png", all four wheels share "wheel_atlas.png"): verified directly against
`xacro <file> | gz sdf -p -` output (see PROGRESS.md-adjacent notes in the urdf file) that
sdformat's urdf2sdf pass applies a `<gazebo reference="LINK"><visual><material>...`
override to EVERY SDF visual belonging to that link, regardless of the `name=` given on
the inner <visual> element - the name is not used to select a target. So a link with
several decorative <visual> sub-parts cannot be given several different PBR materials;
it can only be given one, applied uniformly to whichever visuals exist under it. The
fix used throughout this file is a texture atlas: every part mounted on a given link
samples a different UV region of that link's single shared albedo/normal/roughness/
metalness map set, so the parts still read as different materials even though there is
exactly one SDF <material> block per link.

Run directly: `python3 generate_rover_meshes.py [output_dir]`
(defaults to ../meshes next to this file, which is what is committed to the repo).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# --------------------------------------------------------------------------------------
# Frozen dimensions, copied from regolith_rover.urdf.xacro's xacro:property values.
# These meshes must fill exactly these envelopes - see test/test_mesh_generation.py,
# which parses the live xacro file and checks mesh extents against it directly, so this
# file cannot silently drift from the URDF it is meant to dress.
# --------------------------------------------------------------------------------------
CHASSIS_LENGTH = 0.40
CHASSIS_WIDTH = 0.34
CHASSIS_HEIGHT = 0.11
WHEEL_RADIUS = 0.09
WHEEL_WIDTH = 0.06

RNG_SEED = 20260101  # fixed - determinism is load-bearing, see the regeneration test


# ========================================================================================
# Minimal triangle-mesh builder: positions + UVs + flat per-face normals, OBJ export.
# ========================================================================================


class MeshBuilder:
    """Accumulates a triangle soup (position, uv, flat face normal) and writes OBJ.

    Deliberately not indexed/welded (like rocks.py's write_obj): every triangle gets its
    own 3 vertex copies and one face normal. That costs some vertex count but keeps the
    topology code trivial and gives crisp flat-shaded edges, which reads as "machined
    hardware" rather than a smoothed blob - the same trade rocks.py makes for boulders.
    """

    def __init__(self) -> None:
        self.positions: list[tuple[float, float, float]] = []
        self.uvs: list[tuple[float, float]] = []
        self.normals: list[tuple[float, float, float]] = []
        self.faces: list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = []

    def add_tri(self, p0, p1, p2, uv0, uv1, uv2) -> None:
        a = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
        b = np.asarray(p2, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
        n = np.cross(a, b)
        norm = np.linalg.norm(n)
        n = n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        ni = len(self.normals)
        self.normals.append((float(n[0]), float(n[1]), float(n[2])))
        idxs = []
        for p, uv in ((p0, uv0), (p1, uv1), (p2, uv2)):
            self.positions.append((float(p[0]), float(p[1]), float(p[2])))
            self.uvs.append((float(uv[0]), float(uv[1])))
            idxs.append((len(self.positions) - 1, len(self.uvs) - 1, ni))
        self.faces.append((idxs[0], idxs[1], idxs[2]))

    def add_quad(self, p0, p1, p2, p3, uv0, uv1, uv2, uv3) -> None:
        """p0..p3 in CCW order (outward normal by right-hand rule)."""
        self.add_tri(p0, p1, p2, uv0, uv1, uv2)
        self.add_tri(p0, p2, p3, uv0, uv2, uv3)

    def tri_count(self) -> int:
        return len(self.faces)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        arr = np.array(self.positions)
        return arr.min(axis=0), arr.max(axis=0)

    def merge(self, other: "MeshBuilder", offset=(0.0, 0.0, 0.0), rotation: np.ndarray | None = None,
              uv_rect: tuple[float, float, float, float] | None = None) -> None:
        """Append `other`'s geometry, rigidly transformed and (optionally) UV-remapped
        into a sub-rectangle of this mesh's shared atlas: uv_rect = (u0, v0, u1, v1),
        with other's own [0,1]x[0,1] UV space linearly mapped into it."""
        R = rotation if rotation is not None else np.eye(3)
        off = np.asarray(offset, dtype=np.float64)
        pos_offset = len(self.positions)
        uv_offset = len(self.uvs)
        norm_offset = len(self.normals)
        for p in other.positions:
            wp = R @ np.asarray(p, dtype=np.float64) + off
            self.positions.append((float(wp[0]), float(wp[1]), float(wp[2])))
        for n in other.normals:
            wn = R @ np.asarray(n, dtype=np.float64)
            self.normals.append((float(wn[0]), float(wn[1]), float(wn[2])))
        if uv_rect is None:
            self.uvs.extend(other.uvs)
        else:
            u0, v0, u1, v1 = uv_rect
            for u, v in other.uvs:
                self.uvs.append((u0 + u * (u1 - u0), v0 + v * (v1 - v0)))
        for f in other.faces:
            self.faces.append(tuple(
                (pi + pos_offset, ti + uv_offset, ni + norm_offset) for (pi, ti, ni) in f
            ))

    def write_obj(self, path: Path, name: str) -> None:
        lines = [f"# Regolith rover procedural mesh: {name}", f"o {name}"]
        for p in self.positions:
            lines.append("v {:.6f} {:.6f} {:.6f}".format(*p))
        for uv in self.uvs:
            lines.append("vt {:.6f} {:.6f}".format(*uv))
        for n in self.normals:
            lines.append("vn {:.6f} {:.6f} {:.6f}".format(*n))
        for f in self.faces:
            tokens = " ".join(f"{pi + 1}/{ti + 1}/{ni + 1}" for (pi, ti, ni) in f)
            lines.append(f"f {tokens}")
        path.write_text("\n".join(lines) + "\n")


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ========================================================================================
# Shape primitives
# ========================================================================================


def add_box(mesh: MeshBuilder, center, size, uv_rect=(0.0, 0.0, 1.0, 1.0)) -> None:
    """Axis-aligned box, `size` = (sx, sy, sz), all 6 faces mapped into the same uv_rect
    (each face reads its own local plane coords normalized to [0,1] within that rect) -
    fine for small decorative bosses/clips where a slightly-repeated texture read is
    imperceptible at their scale."""
    cx, cy, cz = center
    sx, sy, sz = size
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    u0, v0, u1, v1 = uv_rect

    def uv(a, b):
        return (u0 + a * (u1 - u0), v0 + b * (v1 - v0))

    # +X, -X, +Y, -Y, +Z, -Z faces, each CCW when viewed from outside.
    mesh.add_quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), uv(0, 0), uv(1, 0), uv(1, 1), uv(0, 1))
    mesh.add_quad((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), uv(0, 0), uv(1, 0), uv(1, 1), uv(0, 1))
    mesh.add_quad((x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1), uv(0, 0), uv(1, 0), uv(1, 1), uv(0, 1))
    mesh.add_quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), uv(0, 0), uv(1, 0), uv(1, 1), uv(0, 1))
    mesh.add_quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), uv(0, 0), uv(1, 0), uv(1, 1), uv(0, 1))
    mesh.add_quad((x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0), uv(0, 0), uv(1, 0), uv(1, 1), uv(0, 1))


def chamfered_rect_profile(length: float, width: float, chamfer: float) -> list[tuple[float, float]]:
    """8-point octagon from a length x width rectangle with its 4 corners cut by `chamfer`,
    wound CCW in the standard x-right/y-up sense (signed shoelace area > 0) - this is
    the winding extrude_polygon's cap-triangle code assumes to get an outward +Z normal
    on the TOP cap. An earlier version of this list was wound CW (verified by computing
    its signed area: negative), which flipped every flat top cap - the chassis deck and
    the hull's own top face - to face down INTO the mesh. Lit only by whatever leaks
    around that backwards normal, every upward-facing surface on the rover rendered as
    a near-black silhouette from any elevated or top-down camera, regardless of albedo
    or metalness - this was the actual cause of "the rover reads as a dark slab from
    hero/orbit", not a material or lighting tuning problem. Fixed here; see
    test_mesh_generation.py for the regression test."""
    L, W, c = length / 2.0, width / 2.0, chamfer
    return [
        (L - c, W), (-L + c, W),
        (-L, W - c), (-L, -W + c),
        (-L + c, -W), (L - c, -W),
        (L, -W + c), (L, W - c),
    ]


def extrude_polygon(profile_xy: list[tuple[float, float]], z0: float, z1: float,
                     cap_bottom: bool = True, cap_top: bool = True,
                     v_range: tuple[float, float] = (0.0, 1.0)) -> MeshBuilder:
    """Extrude a CCW polygon (any simple, roughly-convex polygon - true for every profile
    used in this file) from z0 to z1. Side UV.u = cumulative perimeter fraction,
    UV.v = v_range[0]..v_range[1]. Caps use planar UV normalized to the profile bbox."""
    mesh = MeshBuilder()
    n = len(profile_xy)
    pts = np.array(profile_xy, dtype=np.float64)
    seg_lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    perimeter = seg_lengths.sum()
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    v0, v1 = v_range
    for i in range(n):
        x0, y0 = profile_xy[i]
        x1, y1 = profile_xy[(i + 1) % n]
        ua, ub = cum[i] / perimeter, cum[i + 1] / perimeter
        mesh.add_quad(
            (x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1),
            (ua, v0), (ub, v0), (ub, v1), (ua, v1),
        )
    mins = pts.min(axis=0)
    span = np.maximum(pts.max(axis=0) - mins, 1e-9)

    def planar_uv(x, y):
        return ((x - mins[0]) / span[0], (y - mins[1]) / span[1])

    if cap_bottom:
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        for i in range(n):
            x0, y0 = profile_xy[i]
            x1, y1 = profile_xy[(i + 1) % n]
            # Wound so the outward normal at z0 points -Z: reverse order vs. top cap.
            mesh.add_tri((cx, cy, z0), (x1, y1, z0), (x0, y0, z0),
                         planar_uv(cx, cy), planar_uv(x1, y1), planar_uv(x0, y0))
    if cap_top:
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        for i in range(n):
            x0, y0 = profile_xy[i]
            x1, y1 = profile_xy[(i + 1) % n]
            mesh.add_tri((cx, cy, z1), (x0, y0, z1), (x1, y1, z1),
                         planar_uv(cx, cy), planar_uv(x0, y0), planar_uv(x1, y1))
    return mesh


def revolve_profile(profile_rz: list[tuple[float, float]], segments: int,
                     uv_v_from_arclength: bool = True) -> MeshBuilder:
    """Lathe a (radius, z) profile around the local Z axis into `segments` angular steps.

    Handles poles (r ~ 0) with triangle fans instead of degenerate quads, so a profile
    can open and close cleanly (hub cap -> rim step -> tire sidewall -> tread -> ... ->
    hub cap) in one call. UV.u = angle/2pi; UV.v = cumulative profile arc-length fraction
    (so texture bands - tread vs. sidewall vs. hub - line up with real profile features)
    if uv_v_from_arclength, else linear index fraction.
    """
    mesh = MeshBuilder()
    pts = np.array(profile_rz, dtype=np.float64)
    m = len(pts)
    if uv_v_from_arclength:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = max(seg.sum(), 1e-9)
        v_vals = np.concatenate([[0.0], np.cumsum(seg) / total])
    else:
        v_vals = np.linspace(0.0, 1.0, m)

    thetas = [2.0 * math.pi * j / segments for j in range(segments + 1)]

    def ring_point(i: int, j: int) -> tuple[float, float, float]:
        r, z = pts[i]
        t = thetas[j]
        return (r * math.cos(t), r * math.sin(t), z)

    POLE_EPS = 1e-9
    for i in range(m - 1):
        r0, z0 = pts[i]
        r1, z1 = pts[i + 1]
        v0, v1 = v_vals[i], v_vals[i + 1]
        if r0 <= POLE_EPS and r1 <= POLE_EPS:
            continue
        if r0 <= POLE_EPS:
            # Fan from a single pole point (z0) out to ring i+1.
            pole = (0.0, 0.0, z0)
            for j in range(segments):
                a = ring_point(i + 1, j)
                b = ring_point(i + 1, j + 1)
                ua, ub = j / segments, (j + 1) / segments
                mesh.add_tri(pole, a, b, (ua, v0), (ua, v1), (ub, v1))
        elif r1 <= POLE_EPS:
            pole = (0.0, 0.0, z1)
            for j in range(segments):
                a = ring_point(i, j)
                b = ring_point(i, j + 1)
                ua, ub = j / segments, (j + 1) / segments
                mesh.add_tri(a, pole, b, (ua, v0), (ub, v1), (ub, v0))
        else:
            for j in range(segments):
                a0 = ring_point(i, j)
                a1 = ring_point(i, j + 1)
                b1 = ring_point(i + 1, j + 1)
                b0 = ring_point(i + 1, j)
                ua, ub = j / segments, (j + 1) / segments
                mesh.add_quad(a0, a1, b1, b0, (ua, v0), (ub, v0), (ub, v1), (ua, v1))
    return mesh


def add_radial_block(mesh: MeshBuilder, r_in: float, r_out: float, theta0: float, theta1: float,
                      z0: float, z1: float, uv_rect=(0.0, 0.0, 1.0, 1.0)) -> None:
    """One tread-block-shaped bump: visible outer/side/end faces only (the r_in face is
    embedded in the parent revolve and would never be seen). Used for wheel grousers."""

    def pt(r, t, z):
        return (r * math.cos(t), r * math.sin(t), z)

    p000, p100 = pt(r_in, theta0, z0), pt(r_out, theta0, z0)
    p010, p110 = pt(r_in, theta1, z0), pt(r_out, theta1, z0)
    p001, p101 = pt(r_in, theta0, z1), pt(r_out, theta0, z1)
    p011, p111 = pt(r_in, theta1, z1), pt(r_out, theta1, z1)
    u0, v0, u1, v1 = uv_rect

    def uv(a, b):
        return (u0 + a * (u1 - u0), v0 + b * (v1 - v0))

    # outer (r_out)
    mesh.add_quad(p100, p110, p111, p101, uv(0, 0), uv(1, 0), uv(1, 1), uv(0, 1))
    # theta0 side
    mesh.add_quad(p000, p100, p101, p001, uv(0, 0), uv(1, 0), uv(1, 1), uv(0, 1))
    # theta1 side
    mesh.add_quad(p110, p010, p011, p111, uv(0, 0), uv(1, 0), uv(1, 1), uv(0, 1))
    # z0 end
    mesh.add_quad(p000, p010, p110, p100, uv(0, 0), uv(1, 0), uv(1, 1), uv(0, 1))
    # z1 end
    mesh.add_quad(p101, p111, p011, p001, uv(0, 0), uv(1, 0), uv(1, 1), uv(0, 1))


def offset_profile(profile: list[tuple[float, float]], delta: float) -> list[tuple[float, float]]:
    """Push each vertex of a star-convex-around-the-origin polygon outward by `delta`
    along its own direction from the origin. Good enough for the octagons here (all
    centred on the chassis origin) to build a slightly larger/smaller ring profile."""
    out = []
    for x, y in profile:
        d = math.hypot(x, y)
        if d < 1e-9:
            out.append((x, y))
            continue
        out.append((x + delta * x / d, y + delta * y / d))
    return out


# ========================================================================================
# Rover parts
# ========================================================================================


@dataclass
class PartMesh:
    """A generated part plus the atlas region (in the link's shared texture) its UVs
    were authored against, so the URDF-side comment/manifest can record which region."""

    name: str
    mesh: MeshBuilder
    atlas_region: str


def build_chassis_hull() -> MeshBuilder:
    """Main chassis body: chamfered-rectangle prism (machined corners) with two raised
    trim bands (visible panel-seam ribs) and two small raised access-panel bosses on the
    long side faces. Fills the chassis collision box exactly at its core envelope; the
    trim rings and bosses are the only geometry allowed to poke a few mm past it, which
    is fine since this is VISUAL ONLY (no collision tie)."""
    chamfer = 0.035
    profile = chamfered_rect_profile(CHASSIS_LENGTH, CHASSIS_WIDTH, chamfer)
    z0, z1 = -CHASSIS_HEIGHT / 2.0, CHASSIS_HEIGHT / 2.0

    mesh = extrude_polygon(profile, z0, z1, cap_bottom=True, cap_top=True, v_range=(0.04, 0.94))

    # Two raised trim rings (panel seams), each a thin closed loop pushed slightly
    # outward - built as their own extrusion of an offset profile, no caps (open tube).
    for z_center, v_band in ((-0.020, (0.30, 0.40)), (0.016, (0.62, 0.72))):
        ring_profile = offset_profile(profile, 0.004)
        ring = extrude_polygon(ring_profile, z_center - 0.004, z_center + 0.004,
                                cap_bottom=False, cap_top=False, v_range=v_band)
        mesh.merge(ring)

    # Two small raised "access panel" bosses on the +Y and -Y long faces.
    # uv_rect sits at local v=0.78-0.88: clear of the gold skirt (v<0.42), the placard
    # (v=0.60-0.74) and the hazard stripe (v=0.88-0.97), so these read as plain
    # brushed-aluminum panels rather than accidentally sampling another feature.
    boss_size = (0.10, 0.006, 0.05)
    add_box(mesh, (0.06, CHASSIS_WIDTH / 2.0 + 0.003, 0.012), boss_size, uv_rect=(0.05, 0.78, 0.22, 0.88))
    add_box(mesh, (0.06, -(CHASSIS_WIDTH / 2.0 + 0.003), 0.012), boss_size, uv_rect=(0.05, 0.78, 0.22, 0.88))

    return mesh


DECK_STANDOFF_GAP = 0.012  # deliberate, visible gap between hull top and deck underside


def build_solar_deck() -> MeshBuilder:
    """Thin panel raised on 4 visible corner standoffs above the hull deck, with real
    3D relief on its top face: a raised perimeter rim, 3 structural ribs, a raised
    bezel framing the solar array, and bolt-flange nubs along both long edges.

    Why the relief matters as much as the albedo does: this world's sun sits at only
    12-25 degrees elevation, so a dead-FLAT horizontal surface catches very little
    direct light regardless of how bright its paint is (sin(12deg) ~= 0.2 of full
    irradiance). Raised features change that locally - a rib or rim edge facing the sun
    catches much closer to full irradiance on its near-vertical face, and casts a real
    (small) shadow on the other side, which is exactly the "grazing light rakes across
    hardware" look real rover decks have and a flat plate cannot produce no matter what
    color it is painted. See generate_solar_region's docstring for the paint-albedo
    half of this fix.
    """
    length, width, thick = CHASSIS_LENGTH - 0.05, CHASSIS_WIDTH - 0.05, 0.010
    profile = chamfered_rect_profile(length, width, 0.02)
    mesh = extrude_polygon(profile, 0.0, thick, cap_bottom=True, cap_top=True, v_range=(0.0, 1.0))
    standoff = (0.022, 0.022, DECK_STANDOFF_GAP)
    for sx in (-1, 1):
        for sy in (-1, 1):
            add_box(mesh, (sx * (length / 2 - 0.02), sy * (width / 2 - 0.02), -DECK_STANDOFF_GAP / 2.0),
                    standoff)

    # Raised perimeter rim: a smaller, taller inner plate sitting on top of the base
    # plate (the two-tier construction build_chassis_hull's trim rings use, but here
    # the base plate's own exposed top ledge - visible in the ~12mm gap between the
    # two profiles - IS the rim, rather than a separate thin ring mesh) - a proper
    # watertight raised block with a correct flat (capped) top, not a hand-built sliver.
    rim_h = 0.016
    raised_profile = offset_profile(profile, -0.014)
    raised = extrude_polygon(raised_profile, thick, thick + rim_h,
                              cap_bottom=False, cap_top=True, v_range=(0.0, 1.0))
    mesh.merge(raised)

    # 3 raised structural ribs across the white-plate portion (see array_u0 in
    # generate_solar_region - the plate is roughly the rear half of the deck, x < 0).
    # Sit on top of the RAISED inner plate (top at thick + rim_h), not the base ledge.
    raised_top = thick + rim_h
    for rx in (-0.135, -0.085, -0.035):
        add_box(mesh, (rx, 0.0, raised_top + 0.007), (0.013, width - 0.08, 0.014),
                uv_rect=(0.0, 0.02, 0.46, 0.10))

    # Raised bezel framing the solar array footprint (array_u0/u1 = 0.58/0.83 in
    # generate_solar_region, length=0.35 -> x approx 0.028..0.115), also on the raised
    # plate.
    array_x0, array_x1 = 0.028, 0.115
    bezel_h = 0.012
    bezel_profile = [
        (array_x0 - 0.008, -width / 2 + 0.045), (array_x1 + 0.008, -width / 2 + 0.045),
        (array_x1 + 0.008, width / 2 - 0.045), (array_x0 - 0.008, width / 2 - 0.045),
    ]
    bezel = extrude_polygon(bezel_profile, raised_top, raised_top + bezel_h,
                             cap_bottom=False, cap_top=False, v_range=(0.4, 0.5))
    mesh.merge(bezel)

    # Bolt-flange nubs along both long edges, on the LOWER outer ledge (the ~14mm
    # band between the true edge and the raised inner plate) - read as the fasteners
    # holding the raised panel to the frame below it.
    for bx in np.linspace(-0.15, -0.02, 5):
        for by in (-(width / 2 - 0.008), (width / 2 - 0.008)):
            add_box(mesh, (bx, by, thick + 0.004), (0.012, 0.010, 0.008),
                    uv_rect=(0.0, 0.02, 0.1, 0.1))
    return mesh


def build_equipment_box() -> MeshBuilder:
    """A small MLI-wrapped avionics/battery box - real spacecraft hardware, and a
    second, unmistakable, geometrically-distinct patch of gold thermal blanket so the
    material reads clearly from more angles than just the hull's lower skirt does."""
    size = (0.075, 0.055, 0.045)
    mesh = MeshBuilder()
    add_box(mesh, (0.0, 0.0, 0.0), size, uv_rect=(0.05, 0.05, 0.95, 0.95))
    # A strap line across the top, like a tie-down over the blanket.
    add_box(mesh, (0.0, 0.0, size[2] / 2.0 + 0.002), (size[0] + 0.004, 0.008, 0.004),
            uv_rect=(0.1, 0.02, 0.9, 0.08))
    return mesh


def build_mast_assembly() -> MeshBuilder:
    """Short tapered mast pole + a mounting yoke + a stereo-camera head with a sunshade
    lip and two distinct lens barrels. Lathe axis is local Z (matches the wheel-cylinder
    convention already used in the URDF), so the URDF <origin rpy> alone controls how it
    stands up off the deck.

    Shortened (0.38 m pole -> 0.145 m) after the first pass visually dominated the 0.40 m
    chassis - a mast is meant to read as "sensor head above the deck", not as the tallest
    feature on the vehicle by a wide margin. Total added height above the deck (pole +
    yoke + head + sunshade) is ~0.21 m, well inside the "<= ~0.5 m above the deck" budget.
    """
    pole_height = 0.145
    profile = [
        (0.0, 0.0),
        (0.012, 0.0),
        (0.012, pole_height * 0.08),
        (0.010, pole_height * 0.08),
        (0.010, pole_height * 0.95),
        (0.008, pole_height),
        (0.0, pole_height),
    ]
    mesh = revolve_profile(profile, segments=12)

    # Mounting yoke: a small bracket between the bare pole and the camera head, so the
    # head reads as MOUNTED hardware rather than glued straight onto the pole tip.
    yoke_z = pole_height + 0.008
    add_box(mesh, (0.0, 0.0, yoke_z), (0.028, 0.052, 0.014), uv_rect=(0.0, 0.4, 0.3, 0.5))

    # Camera head: boxy housing, a forward sunshade lip, and two distinct lens barrels -
    # a real stereo pair, not one blob. Faces +X (forward, over the deck).
    head_z = yoke_z + 0.007 + 0.021
    head = MeshBuilder()
    add_box(head, (0.006, 0.0, 0.0), (0.062, 0.11, 0.042), uv_rect=(0.0, 0.7, 1.0, 1.0))
    # Sunshade: a thin lip cantilevered off the top-front edge of the head.
    add_box(head, (0.034, 0.0, 0.026), (0.028, 0.12, 0.006), uv_rect=(0.5, 0.4, 0.8, 0.45))
    lens_profile = [(0.0, 0.0), (0.013, 0.0), (0.013, 0.016), (0.009, 0.021), (0.0, 0.022)]
    for side in (-1, 1):
        lens = revolve_profile(lens_profile, segments=12)
        # Lens barrel built along local Z; rotate so it points along +X (camera boresight).
        head.merge(lens, offset=(0.037, side * 0.032, 0.0), rotation=rot_y(math.pi / 2.0),
                   uv_rect=(0.0, 0.55, 0.45, 0.68))
    mesh.merge(head, offset=(0.0, 0.0, head_z))
    return mesh


def build_dish() -> MeshBuilder:
    """Shallow parabolic high-gain antenna dish on a short feed-horn stalk, lathe axis
    local Z with the dish opening toward +Z (URDF origin rpy tilts it to point out)."""
    radius = 0.052
    depth = 0.020
    rings = 8
    profile = [(0.0, 0.0)]
    for i in range(1, rings + 1):
        r = radius * i / rings
        z = depth * (r / radius) ** 2  # parabola: z ~ r^2
        profile.append((r, z))
    profile.append((radius, depth + 0.004))  # small rim lip
    dish_shell = revolve_profile(profile, segments=20)

    mesh = MeshBuilder()
    mesh.merge(dish_shell)
    # Feed horn: small stalk standing up from the dish vertex.
    horn_profile = [(0.0, 0.0), (0.006, 0.0), (0.004, 0.03), (0.0, 0.032)]
    horn = revolve_profile(horn_profile, segments=8)
    mesh.merge(horn, offset=(0.0, 0.0, -0.032), rotation=rot_x(math.pi))
    return mesh


def build_whip_antenna() -> MeshBuilder:
    """Stubby UHF stub antenna on a round base flange - NOT a needle-thin whip.

    An earlier pass tapered to a 0.5 mm tip over 0.30 m: at typical shot scale that is
    sub-pixel and reads as a stray scratched line, not as hardware. This is deliberately
    short and thick enough to survive down-scaling: base radius 1 cm, tapering to a
    still-visible 4 mm tip over only 12 cm, on a flanged base that reads as a real mount.
    """
    profile = [
        (0.0, 0.0), (0.016, 0.0), (0.016, 0.004),  # base flange
        (0.010, 0.006), (0.010, 0.02),              # collar
        (0.006, 0.02), (0.006, 0.12), (0.004, 0.13), (0.0, 0.13),
    ]
    return revolve_profile(profile, segments=12)


def build_radiator() -> MeshBuilder:
    """Finned radiator panel: a thin backplate with 5 raised fins, built with its thin
    (mounting) axis along local Y so it can sit flush on the chassis' +/-Y side faces
    with a pure yaw rotation (no extra axis remap needed) - see the URDF placement."""
    length, height, thickness = 0.14, 0.085, 0.007
    mesh = extrude_polygon(
        [(-length / 2, -thickness / 2), (length / 2, -thickness / 2),
         (length / 2, thickness / 2), (-length / 2, thickness / 2)],
        0.0, height, cap_bottom=True, cap_top=True,
    )
    # extrude_polygon works in XY + Z-extrusion; rotate this whole panel so the extrusion
    # axis (originally Z) becomes the vertical (Z stays Z is fine actually - the profile
    # above is already in the X/Y=thickness plane and extrudes along Z = "up", which is
    # exactly the panel's height direction). Thin axis is Y, matching the docstring.
    fin_count = 5
    for i in range(fin_count):
        z = height * (i + 0.5) / fin_count
        add_box(mesh, (0.0, 0.0, z), (length - 0.006, thickness + 0.012, 0.006),
                uv_rect=(0.0, 0.8, 1.0, 1.0))
    return mesh


def build_cable_run() -> MeshBuilder:
    """A short cable-harness run: two straight tube segments joined at a bend, with two
    small clip boxes - reads as a wire loom clipped along a chassis edge."""
    seg_a_len, seg_b_len, r = 0.09, 0.07, 0.006
    circle = [(r * math.cos(2 * math.pi * i / 10), r * math.sin(2 * math.pi * i / 10)) for i in range(10)]
    mesh = extrude_polygon(circle, 0.0, seg_a_len, cap_bottom=True, cap_top=False, v_range=(0.0, 0.5))
    mesh_final = MeshBuilder()
    mesh_final.merge(mesh, rotation=rot_y(math.pi / 2.0))  # run along local +X
    seg_b = extrude_polygon(circle, 0.0, seg_b_len, cap_bottom=False, cap_top=True, v_range=(0.5, 1.0))
    mesh_final.merge(seg_b, offset=(seg_a_len, 0.0, 0.0),
                      rotation=rot_y(math.pi / 2.0) @ rot_z(0.6))
    for t in (0.03, seg_a_len + 0.03):
        add_box(mesh_final, (t, 0.0, -r - 0.004), (0.014, 0.02, 0.008), uv_rect=(0.0, 0.0, 1.0, 0.2))
    return mesh_final


def build_wheel_hub() -> tuple[MeshBuilder, list[float]]:
    """Wheel with a stepped hub/rim/sidewall/tread silhouette (built entirely from the
    revolve profile - no extra geometry needed for that part) plus raised grousers
    around the tread. Lathe axis local Z, exactly matching the existing plain-cylinder
    visual's own local frame (radius in XY, length along Z) so the URDF's existing
    `rpy="1.5708 0 0"` origin needs no change. Returns (mesh, v_breakpoints) where
    v_breakpoints are the cumulative-arclength v-fractions of each named profile
    feature, so the texture generator can paint bands that line up with the geometry.
    """
    r, w = WHEEL_RADIUS, WHEEL_WIDTH
    hub_r = r * 0.30
    rim_r = r * 0.62
    tire_r = r
    sidewall_in_z = -w / 2 + 0.006
    tread_in_z = -w / 2 + 0.012
    tread_out_z = w / 2 - 0.012
    sidewall_out_z = w / 2 - 0.006

    profile = [
        (0.0, -w / 2),
        (hub_r, -w / 2),
        (hub_r, sidewall_in_z),
        (rim_r, sidewall_in_z),
        (rim_r, tread_in_z),
        (tire_r, tread_in_z),
        (tire_r, tread_out_z),
        (rim_r, tread_out_z),
        (rim_r, sidewall_out_z),
        (hub_r, sidewall_out_z),
        (hub_r, w / 2),
        (0.0, w / 2),
    ]
    seg = np.linalg.norm(np.diff(np.array(profile), axis=0), axis=1)
    total = seg.sum()
    v_breakpoints = np.concatenate([[0.0], np.cumsum(seg) / total]).tolist()

    segments = 28
    mesh = revolve_profile(profile, segments=segments, uv_v_from_arclength=True)

    grouser_count = 18
    grouser_height = 0.010
    grouser_half_angle = (2 * math.pi / grouser_count) * 0.28
    tread_v0 = v_breakpoints[5]  # start of tread cylinder in the profile above
    tread_v1 = v_breakpoints[6]
    for k in range(grouser_count):
        theta = 2 * math.pi * k / grouser_count
        add_radial_block(
            mesh, tire_r, tire_r + grouser_height,
            theta - grouser_half_angle, theta + grouser_half_angle,
            tread_in_z + 0.002, tread_out_z - 0.002,
            uv_rect=(0.0, tread_v0, 1.0, tread_v1),
        )
    return mesh, v_breakpoints


# ========================================================================================
# Procedural PBR textures (numpy + PIL only)
# ========================================================================================

CHASSIS_ATLAS_PX = 2048
WHEEL_ATLAS_PX = 1024

# Sub-rectangles of the chassis atlas, in OBJ uv space (u right, v UP), shared by every
# <visual> mounted on the chassis link - see the module docstring for why one shared
# atlas per link is the only option this sdformat version actually supports.
CHASSIS_ATLAS_REGIONS = {
    "hull": (0.0, 0.0, 1.0, 0.60),
    "solar": (0.0, 0.60, 0.5, 0.84),
    "white": (0.5, 0.60, 1.0, 0.84),
    "gold": (0.0, 0.84, 1.0, 1.0),
}


def _smooth_noise(h: int, w: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency noise in [0, 1] via a Gaussian-blurred random field (PIL does the
    blur - no scipy needed, matching the numpy+PIL-only constraint)."""
    field = (rng.random((h, w)) * 255).astype(np.uint8)
    blurred = np.asarray(Image.fromarray(field, mode="L").filter(ImageFilter.GaussianBlur(sigma)),
                          dtype=np.float64) / 255.0
    lo, hi = blurred.min(), blurred.max()
    return (blurred - lo) / max(hi - lo, 1e-6)


def _row_v(h: int) -> np.ndarray:
    """Row index -> OBJ v value, row 0 = top = v close to 1 (image top-left origin vs.
    OBJ's bottom-left v=0 convention)."""
    return 1.0 - (np.arange(h) + 0.5) / h


def _height_to_normal(height: np.ndarray, strength: float) -> np.ndarray:
    gy, gx = np.gradient(height)
    nx, ny = -gx * strength, -gy * strength
    nz = np.ones_like(nx)
    length = np.sqrt(nx**2 + ny**2 + nz**2)
    rgb = np.stack([nx / length * 0.5 + 0.5, ny / length * 0.5 + 0.5, nz / length * 0.5 + 0.5], axis=-1)
    return rgb


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def _draw_hazard_stripe(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int, period: int) -> None:
    yy, xx = np.mgrid[y0:y1, x0:x1]
    diag = ((xx - x0) + (yy - y0)) % period
    mask[y0:y1, x0:x1] = np.where(diag < period // 2, 1.0, 0.0)


def generate_hull_region(h: int, w: int, rng: np.random.Generator) -> dict:
    v = _row_v(h)[:, None] * np.ones((1, w))
    base_variation = _smooth_noise(h, w, 6.0, rng)

    # Base metalness is LOW (0.22, was 0.75): this project's shipped lighting is a
    # single harsh directional sun over near-black ambient with no environment/IBL map
    # (verified with an isolated bright-light test render of just this mesh - the maps
    # and UVs were always correct; a HIGH-metalness surface has ~zero diffuse term, so
    # away from the narrow specular highlight it rendered flat black regardless of
    # albedo - that was the "featureless black slab" bug, not a UV/binding bug). Low
    # metalness keeps a real N.L diffuse response so the surface reads under directional
    # light alone, at the cost of being a less "shiny" metal than a studio-lit render
    # would want - the right trade for this renderer.
    albedo = np.zeros((h, w, 3))
    albedo[..., 0] = 0.66 + 0.08 * base_variation
    albedo[..., 1] = 0.68 + 0.08 * base_variation
    albedo[..., 2] = 0.71 + 0.08 * base_variation

    height = 0.15 * base_variation
    roughness = np.full((h, w), 0.45) + 0.10 * _smooth_noise(h, w, 10.0, rng)
    metalness = np.full((h, w), 0.22)

    # A bold, BLOCKY light band near the top of the hull (not a thin line): under this
    # project's harsh low-sun/near-black-ambient lighting, fine tracery (panel lines,
    # rivets) survives up close but disappears at "rover is 150-250 px tall" hero-shot
    # distance - what reads at that scale is a large area of contrasting VALUE. This
    # band, the mid-body, and the gold skirt below give the hull a light/mid/gold
    # three-tone read that still separates even when everything is dim.
    top_band = np.clip((v - 0.74) / 0.10, 0.0, 1.0)
    albedo = albedo * (1 - top_band[..., None]) + 0.90 * top_band[..., None]
    # A crisp dark seam right at the band boundary - the edge itself is what the eye
    # locks onto at a distance, more than either flat tone alone.
    seam = np.clip(1.0 - np.abs(v - 0.745) / 0.012, 0.0, 1.0)
    albedo *= (1.0 - 0.6 * seam[..., None])

    # Panel lines: a handful of horizontal seams (rows) plus sparse vertical seams.
    # Widened and darkened vs. an earlier pass that was too faint to survive a rover
    # that is only ~150 px tall on screen: contrast, not just presence, is what reads
    # at that scale.
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    rng_lines = np.random.default_rng(rng.integers(0, 2**31 - 1))
    line_w = max(3, w // 340)
    for _ in range(9):
        y = int(rng_lines.uniform(0.08, 0.92) * h)
        draw.line([(0, y), (w, y)], fill=255, width=line_w)
    for _ in range(14):
        x = int(rng_lines.uniform(0.0, 1.0) * w)
        y0 = int(rng_lines.uniform(0.05, 0.5) * h)
        y1 = y0 + int(rng_lines.uniform(0.15, 0.4) * h)
        draw.line([(x, y0), (x, y1)], fill=255, width=line_w)
    # Rivets at a sparse grid, bigger and higher-contrast than before.
    rivet_r = max(3, w // 220)
    for gx in np.linspace(0.03, 0.97, 22):
        for gy in np.linspace(0.10, 0.90, 5):
            if rng_lines.random() < 0.55:
                cx, cy = int(gx * w), int(gy * h)
                draw.ellipse([cx - rivet_r, cy - rivet_r, cx + rivet_r, cy + rivet_r], fill=230)
    lines = np.asarray(img, dtype=np.float64) / 255.0
    albedo *= (1.0 - 0.55 * lines[..., None])
    height -= 0.9 * lines
    roughness += 0.2 * lines

    # Warning hazard stripe band (diagonal yellow/black) near one edge - bigger and
    # more saturated so it survives being a small patch on a small on-screen rover.
    hazard = np.zeros((h, w))
    hy0, hy1 = int(0.88 * h), int(0.97 * h)
    _draw_hazard_stripe(hazard, int(0.02 * w), hy0, int(0.34 * w), hy1, period=max(8, w // 45))
    yellow = np.array([0.95, 0.74, 0.05])
    black = np.array([0.04, 0.04, 0.04])
    hz = hazard[hy0:hy1, int(0.02 * w):int(0.34 * w), None]
    albedo[hy0:hy1, int(0.02 * w):int(0.34 * w), :] = hz * yellow + (1 - hz) * black
    roughness[hy0:hy1, int(0.02 * w):int(0.34 * w)] = 0.5
    metalness[hy0:hy1, int(0.02 * w):int(0.34 * w)] = 0.05

    # Placard: light rectangle with a border and a few "text" dashes plus a status dot.
    px0, px1 = int(0.38 * w), int(0.62 * w)
    py0, py1 = int(0.60 * h), int(0.74 * h)
    albedo[py0:py1, px0:px1, :] = np.array([0.85, 0.85, 0.83])
    border = 2
    albedo[py0:py0 + border, px0:px1, :] = 0.15
    albedo[py1 - border:py1, px0:px1, :] = 0.15
    albedo[py0:py1, px0:px0 + border, :] = 0.15
    albedo[py0:py1, px1 - border:px1, :] = 0.15
    for i, ty in enumerate(np.linspace(py0 + 0.2 * (py1 - py0), py1 - 0.25 * (py1 - py0), 3)):
        ty = int(ty)
        albedo[ty:ty + 2, px0 + 4:px1 - 4 - i * 6, :] = 0.2
    dot_cx, dot_cy = px1 + 8, (py0 + py1) // 2
    if dot_cx < w:
        dr = 4
        yy, xx = np.mgrid[max(0, dot_cy - dr):dot_cy + dr, max(0, dot_cx - dr):dot_cx + dr]
        albedo[max(0, dot_cy - dr):dot_cy + dr, max(0, dot_cx - dr):dot_cx + dr, :] = np.array([0.75, 0.1, 0.08])
    roughness[py0:py1, px0:px1] = 0.35
    metalness[py0:py1, px0:px1] = 0.05

    # Gold/amber MLI thermal-blanket skirt, low v = bottom of the extrusion (the
    # underside of the chassis). Raised from a ~14%-tall trim line to a ~40%-tall
    # wrap: on a rover that is a couple hundred px tall on screen, a thin skirt reads
    # as a stray pixel row, not as a material. This is the single most recognisable
    # "flight hardware" signature the brief calls for, so it needs to be unmissable,
    # not tasteful.
    gold_mask = np.clip((0.42 - v) / 0.30, 0.0, 1.0)  # 1 below v=0.12, fades out by v=0.42
    wrinkle = _smooth_noise(h, w, 3.0, rng) * 0.5 + _smooth_noise(h, w, 8.0, rng) * 0.5
    gold_albedo = np.stack([
        0.62 + 0.22 * wrinkle,
        0.42 + 0.16 * wrinkle,
        0.09 + 0.05 * wrinkle,
    ], axis=-1)
    albedo = albedo * (1 - gold_mask[..., None]) + gold_albedo * gold_mask[..., None]
    height += gold_mask * (wrinkle - 0.5) * 0.8
    roughness = roughness * (1 - gold_mask) + (0.32 + 0.30 * wrinkle) * gold_mask
    metalness = metalness * (1 - gold_mask) + (0.30 + 0.12 * wrinkle) * gold_mask

    normal = _height_to_normal(height, strength=0.9)
    return {
        "albedo": _to_uint8(albedo), "normal": _to_uint8(normal),
        "roughness": _to_uint8(np.stack([roughness] * 3, axis=-1)),
        "metalness": _to_uint8(np.stack([metalness] * 3, axis=-1)),
    }


def generate_solar_region(h: int, w: int, rng: np.random.Generator) -> dict:
    """Deck top-cap texture (this is the single largest surface a downward/hero camera
    sees on the whole vehicle - build_solar_deck's flat top face samples this region
    with a plan-view UV: u runs along the deck's LENGTH axis, v along its WIDTH axis).

    An earlier pass made the entire deck a dark blue-black cell field with a thin
    bright frame - correct for a real solar panel, but wrong for THIS vehicle: from
    hero/orbit height the deck is most of the rover's visible silhouette, so a
    deck-sized dark rectangle reads as "the rover is a black slab" regardless of how
    good the contrast is elsewhere (the gold skirt and radiators are on the sides,
    which a downward-looking camera barely sees). Real rover decks are also
    predominantly LIGHT from above (bare/anodised aluminum, MLI, white composite) with
    the solar array as ONE panel among several, not a lid.
    So: the array now covers about a third of the deck length, and the rest is a light
    aluminum plate with rivets and seams plus a gold MLI accent strip - the same
    dark-tyre/light-hub value logic that already makes the wheels read well, applied to
    the deck.
    """
    u = np.linspace(0.0, 1.0, w)[None, :] * np.ones((h, 1))  # length axis
    v = np.linspace(0.0, 1.0, h)[:, None] * np.ones((1, w))  # width axis

    # --- white thermal-paint deck plate (the majority surface) ---
    # Pushed from "light aluminum" (~0.75) to near-white (~0.86): a dim flat surface
    # under a grazing sun only reads as mid-grey if its albedo is high enough to still
    # show SOMETHING at ~20-40% of full irradiance (sin(12deg)..sin(25deg)) - measured
    # directly (see the commit history around this line): at the old 0.75 albedo the
    # deck sampled at ~6/255 in an actual render, versus ~19-78/255 for the surrounding
    # ground. Real rover decks solve this with white thermal-control paint for exactly
    # this reason (Apollo/MER decks never go black under a low sun) - this is that,
    # not a lighting cheat.
    plate_variation = _smooth_noise(h, w, 6.0, rng)
    albedo = np.zeros((h, w, 3))
    albedo[..., 0] = 0.86 + 0.05 * plate_variation
    albedo[..., 1] = 0.87 + 0.05 * plate_variation
    albedo[..., 2] = 0.89 + 0.05 * plate_variation
    height = 0.15 * plate_variation
    roughness = np.full((h, w), 0.44) + 0.08 * plate_variation
    metalness = np.full((h, w), 0.08)

    # Rivets + a couple of seam lines on the plate area, same idiom as the hull.
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    rng_d = np.random.default_rng(rng.integers(0, 2**31 - 1))
    for gx in np.linspace(0.08, 0.46, 9):
        for gy in np.linspace(0.12, 0.88, 6):
            if rng_d.random() < 0.6:
                cx, cy = int(gx * w), int(gy * h)
                r = max(2, w // 260)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=210)
    for sx in (0.20, 0.34):
        xx = int(sx * w)
        draw.line([(xx, int(0.08 * h)), (xx, int(0.92 * h))], fill=255, width=max(2, w // 300))
    detail = np.asarray(img, dtype=np.float64) / 255.0
    albedo *= (1.0 - 0.35 * detail[..., None])
    height -= 0.5 * detail

    # --- solar array: a clearly smaller inset panel (~1/4 of the deck length, was
    # ~1/3), cell grid with bright busbars - the white plate is the DOMINANT surface. ---
    array_u0, array_u1 = 0.58, 0.83
    array_mask = np.clip(np.minimum((u - array_u0) / 0.03, (array_u1 - u) / 0.03), 0.0, 1.0)
    cell_variation = _smooth_noise(h, w, 4.0, rng)
    per_cell = _smooth_noise(h, w, max(2.0, w / 24.0), rng)
    cell_albedo = np.stack([
        0.06 + 0.03 * cell_variation + 0.07 * per_cell,
        0.08 + 0.04 * cell_variation + 0.08 * per_cell,
        0.18 + 0.08 * cell_variation + 0.10 * per_cell,
    ], axis=-1)
    cols, rows = 4, 7
    gimg = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(gimg)
    grid_w = max(2, w // 260)
    for c in range(cols + 1):
        x = int((array_u0 + c / cols * (array_u1 - array_u0)) * w)
        draw.line([(x, 0), (x, h)], fill=255, width=grid_w)
    for r in range(rows + 1):
        y = int(r / rows * h)
        draw.line([(int(array_u0 * w), y), (int(array_u1 * w), y)], fill=255, width=grid_w)
    grid = np.asarray(gimg, dtype=np.float64) / 255.0
    silver = np.array([0.72, 0.75, 0.80])
    cell_albedo = cell_albedo * (1.0 - grid[..., None]) + silver[None, None, :] * grid[..., None]
    cell_roughness = np.full((h, w), 0.22) - 0.10 * grid
    cell_metalness = np.full((h, w), 0.10) + 0.55 * grid
    cell_height = 0.3 * cell_variation - 0.7 * grid

    albedo = albedo * (1 - array_mask[..., None]) + cell_albedo * array_mask[..., None]
    roughness = roughness * (1 - array_mask) + cell_roughness * array_mask
    metalness = metalness * (1 - array_mask) + cell_metalness * array_mask
    height = height * (1 - array_mask) + cell_height * array_mask

    # --- gold MLI accent strip along the rear edge ---
    gold_u0 = 0.88
    gold_mask = np.clip((u - gold_u0) / 0.03, 0.0, 1.0)
    wrinkle = _smooth_noise(h, w, 3.0, rng)
    gold_albedo = np.stack([
        0.60 + 0.18 * wrinkle, 0.40 + 0.13 * wrinkle, 0.10 + 0.05 * wrinkle,
    ], axis=-1)
    albedo = albedo * (1 - gold_mask[..., None]) + gold_albedo * gold_mask[..., None]
    roughness = roughness * (1 - gold_mask) + (0.30 + 0.25 * wrinkle) * gold_mask
    metalness = metalness * (1 - gold_mask) + (0.28 + 0.1 * wrinkle) * gold_mask
    height = height + gold_mask * (wrinkle - 0.5) * 0.6

    # --- bright frame on all 4 edges, on top of everything else ---
    margin = 0.055
    frame = np.clip(np.maximum.reduce([
        (margin - u) / margin, (u - (1 - margin)) / margin,
        (margin - v) / margin, (v - (1 - margin)) / margin,
    ]), 0.0, 1.0)
    frame_albedo = np.array([0.90, 0.91, 0.93])
    albedo = albedo * (1 - frame[..., None]) + frame_albedo[None, None, :] * frame[..., None]
    roughness = roughness * (1 - frame) + 0.38 * frame
    metalness = metalness * (1 - frame) + 0.22 * frame
    height = height * (1 - frame) + 0.25 * frame

    normal = _height_to_normal(height, strength=0.55)
    return {
        "albedo": _to_uint8(albedo), "normal": _to_uint8(normal),
        "roughness": _to_uint8(np.stack([roughness] * 3, axis=-1)),
        "metalness": _to_uint8(np.stack([metalness] * 3, axis=-1)),
    }


def generate_white_region(h: int, w: int, rng: np.random.Generator) -> dict:
    variation = _smooth_noise(h, w, 7.0, rng)
    albedo = np.zeros((h, w, 3))
    albedo[..., 0] = 0.80 + 0.08 * variation
    albedo[..., 1] = 0.81 + 0.08 * variation
    albedo[..., 2] = 0.83 + 0.08 * variation

    # A dark "lens" disc roughly centred - whichever small part samples this region
    # (camera lens barrels, feed horn tip) reads as glass/dark composite there.
    cy, cx = h * 0.42, w * 0.5
    yy, xx = np.mgrid[0:h, 0:w]
    lens = np.exp(-(((xx - cx) / (w * 0.10)) ** 2 + ((yy - cy) / (h * 0.10)) ** 2))
    albedo = albedo * (1 - lens[..., None]) + np.array([0.03, 0.03, 0.035]) * lens[..., None]

    # Faint horizontal fin/seam shading in the lower band (radiator fins live here).
    v = _row_v(h)
    fin_band = (v < 0.35).astype(np.float64)
    row_stripe = (np.sin(np.arange(h) * 0.35) > 0.6).astype(np.float64)
    fin_lines = (row_stripe * fin_band)[:, None] * np.ones((1, w))
    albedo *= (1.0 - 0.12 * fin_lines[..., None])

    height = 0.3 * variation - 0.5 * lens - 0.3 * fin_lines
    roughness = np.full((h, w), 0.42) + 0.10 * variation
    metalness = np.full((h, w), 0.12) + 0.35 * lens
    normal = _height_to_normal(height, strength=0.6)
    return {
        "albedo": _to_uint8(albedo), "normal": _to_uint8(normal),
        "roughness": _to_uint8(np.stack([roughness] * 3, axis=-1)),
        "metalness": _to_uint8(np.stack([metalness] * 3, axis=-1)),
    }


def generate_gold_region(h: int, w: int, rng: np.random.Generator) -> dict:
    wrinkle = 0.5 * _smooth_noise(h, w, 3.0, rng) + 0.5 * _smooth_noise(h, w, 9.0, rng)
    albedo = np.stack([
        0.66 + 0.22 * wrinkle,
        0.45 + 0.16 * wrinkle,
        0.10 + 0.06 * wrinkle,
    ], axis=-1)
    height = (wrinkle - 0.5) * 0.8
    roughness = 0.30 + 0.30 * wrinkle
    metalness = np.full((h, w), 0.28) + 0.12 * wrinkle
    normal = _height_to_normal(height, strength=1.1)
    return {
        "albedo": _to_uint8(albedo), "normal": _to_uint8(normal),
        "roughness": _to_uint8(np.stack([roughness] * 3, axis=-1)),
        "metalness": _to_uint8(np.stack([metalness] * 3, axis=-1)),
    }


def _uv_rect_to_pixels(rect: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    u0, v0, u1, v1 = rect
    col0, col1 = int(round(u0 * w)), int(round(u1 * w))
    row0, row1 = int(round((1.0 - v1) * h)), int(round((1.0 - v0) * h))
    return row0, row1, col0, col1


def build_chassis_atlas(rng: np.random.Generator) -> dict:
    size = CHASSIS_ATLAS_PX
    maps = {"albedo": np.zeros((size, size, 3), np.uint8),
            "normal": np.full((size, size, 3), 128, np.uint8),
            "roughness": np.full((size, size, 3), 150, np.uint8),
            "metalness": np.zeros((size, size, 3), np.uint8)}
    generators = {"hull": generate_hull_region, "solar": generate_solar_region,
                  "white": generate_white_region, "gold": generate_gold_region}
    for region_name, rect in CHASSIS_ATLAS_REGIONS.items():
        row0, row1, col0, col1 = _uv_rect_to_pixels(rect, size, size)
        rh, rw = row1 - row0, col1 - col0
        region_maps = generators[region_name](rh, rw, rng)
        for key in maps:
            maps[key][row0:row1, col0:col1, :] = region_maps[key]
    return maps


def generate_wheel_atlas(v_breakpoints: list[float], rng: np.random.Generator) -> dict:
    size = WHEEL_ATLAS_PX
    v = _row_v(size)[:, None] * np.ones((1, size))
    bp = v_breakpoints

    def band(v_lo, v_hi):
        return ((v >= v_lo) & (v < v_hi)).astype(np.float64)

    hub_band = band(0.0, bp[2]) + band(bp[9], 1.0)
    rim_band = band(bp[2], bp[3]) + band(bp[8], bp[9])
    sidewall_band = band(bp[3], bp[5]) + band(bp[6], bp[8])
    tread_band = band(bp[5], bp[6])

    metal_variation = _smooth_noise(size, size, 5.0, rng)
    rubber_variation = _smooth_noise(size, size, 6.0, rng)

    albedo = np.zeros((size, size, 3))
    metal_rgb = np.array([0.55, 0.56, 0.58]) + 0.06 * metal_variation[..., None]
    rubber_rgb = np.array([0.045, 0.045, 0.05]) + 0.02 * rubber_variation[..., None]
    albedo += hub_band[..., None] * metal_rgb
    albedo += rim_band[..., None] * (metal_rgb * 0.9)
    albedo += sidewall_band[..., None] * rubber_rgb
    albedo += tread_band[..., None] * (rubber_rgb * 0.85)

    # Spoke pattern baked into the hub cap (geometry there is a flat disc - see
    # build_wheel_hub's docstring: spokes are texture-only, real relief only at the
    # rim/hub step and the grousers).
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    hub_radius_px = int(0.42 * size)
    for k in range(6):
        angle = 2 * math.pi * k / 6
        x2, y2 = cx + hub_radius_px * math.cos(angle), cy + hub_radius_px * math.sin(angle)
        draw.line([(cx, cy), (x2, y2)], fill=255, width=max(2, size // 90))
    for k in range(6):
        angle = 2 * math.pi * (k + 0.5) / 6
        x2, y2 = cx + hub_radius_px * 0.55 * math.cos(angle), cy + hub_radius_px * 0.55 * math.sin(angle)
        draw.ellipse([x2 - 6, y2 - 6, x2 + 6, y2 + 6], fill=200)
    spokes = np.asarray(img, dtype=np.float64) / 255.0
    hub_mask_2d = hub_band  # spokes only where the profile is actually the hub-cap band
    albedo -= 0.18 * (spokes * hub_mask_2d)[..., None]

    height = 0.2 * metal_variation * (hub_band + rim_band) + 0.35 * rubber_variation * (sidewall_band + tread_band)
    height -= 0.5 * spokes * hub_mask_2d
    # Tread block striping (extra bump cue alongside the real grouser geometry).
    tread_stripe = (np.sin(np.arange(size)[None, :] * 0.25) > 0.3).astype(np.float64) * tread_band
    height -= 0.3 * tread_stripe

    roughness = 0.35 * (hub_band + rim_band) + 0.82 * (sidewall_band + tread_band)
    roughness = np.clip(roughness + 0.06 * rubber_variation, 0.0, 1.0)
    # Low, like every other "metal" in this atlas set - see generate_hull_region's
    # docstring on why high metalness goes black under this project's directional-only
    # lighting with no environment map.
    metalness = 0.28 * (hub_band + rim_band) + 0.02 * (sidewall_band + tread_band)

    normal = _height_to_normal(height, strength=0.8)
    return {
        "albedo": _to_uint8(albedo), "normal": _to_uint8(normal),
        "roughness": _to_uint8(np.stack([roughness] * 3, axis=-1)),
        "metalness": _to_uint8(np.stack([metalness] * 3, axis=-1)),
    }


# ========================================================================================
# Top-level assembly
# ========================================================================================


def _finalize(local_mesh: MeshBuilder, uv_rect: tuple[float, float, float, float]) -> MeshBuilder:
    """Remap a part's own [0,1]^2 local UV space into its atlas sub-rectangle."""
    final = MeshBuilder()
    final.merge(local_mesh, uv_rect=uv_rect)
    return final


@dataclass
class GeneratedAsset:
    name: str
    tri_count: int
    mins: tuple
    maxs: tuple


def generate_all(output_dir: Path) -> list[GeneratedAsset]:
    """Generate every rover mesh + texture into output_dir (created if needed) and
    return a manifest of what was written, for the caller to print/verify."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    assets: list[GeneratedAsset] = []

    def emit(name: str, mesh: MeshBuilder) -> None:
        mesh.write_obj(output_dir / f"{name}.obj", name)
        mins, maxs = mesh.bounds()
        assets.append(GeneratedAsset(name, mesh.tri_count(), tuple(mins.tolist()), tuple(maxs.tolist())))

    r = CHASSIS_ATLAS_REGIONS
    emit("chassis_hull", _finalize(build_chassis_hull(), r["hull"]))
    emit("solar_deck", _finalize(build_solar_deck(), r["solar"]))
    emit("mast_assembly", _finalize(build_mast_assembly(), r["white"]))
    emit("dish", _finalize(build_dish(), r["white"]))
    emit("whip_antenna", _finalize(build_whip_antenna(), r["white"]))
    emit("radiator", _finalize(build_radiator(), r["white"]))
    emit("cable_run", _finalize(build_cable_run(), r["gold"]))
    emit("equipment_box", _finalize(build_equipment_box(), r["gold"]))

    wheel_mesh, v_breakpoints = build_wheel_hub()
    emit("wheel_hub", _finalize(wheel_mesh, (0.0, 0.0, 1.0, 1.0)))

    chassis_atlas = build_chassis_atlas(rng)
    for key, arr in chassis_atlas.items():
        Image.fromarray(arr, mode="RGB").save(output_dir / f"chassis_atlas_{key}.png")

    wheel_atlas = generate_wheel_atlas(v_breakpoints, rng)
    for key, arr in wheel_atlas.items():
        Image.fromarray(arr, mode="RGB").save(output_dir / f"wheel_atlas_{key}.png")

    return assets


def main(argv: list[str]) -> int:
    output_dir = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "meshes"
    assets = generate_all(output_dir)
    total_tris = sum(a.tri_count for a in assets)
    print(f"Wrote {len(assets)} meshes + PBR texture atlases to {output_dir}")
    for a in assets:
        size = tuple(round(hi - lo, 4) for lo, hi in zip(a.mins, a.maxs))
        print(f"  {a.name:16s} {a.tri_count:5d} tris   extent (x,y,z) = {size}")
    print(f"  TOTAL (single-instance) tris: {total_tris}"
          f"  (wheel_hub is instanced x4 in the URDF: scene total ~= "
          f"{total_tris + 3 * [a for a in assets if a.name == 'wheel_hub'][0].tri_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
