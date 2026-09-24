# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Procedural starfield sky: a huge sphere around the world, carrying a generated
equirectangular starfield + Milky Way texture, rendered so it reads as black
space with pinpoint stars regardless of the scene's sun direction or ambient
level.

WHY A SPHERE, AND WHY NOT gz-sim's BUILT-IN SKY
-------------------------------------------------
Real lunar sky is black space: no atmosphere to scatter the sun, so stars are
visible in daylight and the sky itself emits no light of its own. That rules out
gz-sim's built-in ``<scene><sky>`` element, which drives Ogre2's atmospheric
scattering sky (SkyX) - an Earth-like blue/orange gradient that brightens toward
the sun and cannot be made to look like black space with pinpoint stars.

Instead this is a custom ``<mesh>`` UV sphere (see `save_sky_dome_mesh`), large
enough that the camera is always deep inside it. `cast_shadows` is off: nothing
this far away should ever throw a shadow onto the terrain, and leaving shadow
casting on a huge full-sphere mesh would be needless GPU cost for zero visual
benefit (the mesh is always entirely behind everything else, from the light's
point of view too).

HOW IT STAYS UNAFFECTED BY SCENE LIGHTING, WITHOUT ``lighting=false``
------------------------------------------------------------------------
The obvious approach - ``<material><lighting>false</lighting>`` with the star
texture as `<pbr><metal><albedo_map>` - renders as solid white on this
gz-rendering/Ogre2 build (root-caused in `sky_model_sdf`'s docstring: that
specific combination on `<mesh>` geometry silently falls back to a default
material, independent of which texture is referenced). The texture is instead
carried as `<emissive_map>` with `ambient`/`diffuse`/`specular` all zeroed and
`lighting` left at its default (true): emissive is additive and unaffected by
the (zeroed-out, so moot anyway) diffuse shading term, so the visible result is
the star texture exactly as authored, regardless of sun direction or ambient
level - the same practical outcome true unlit rendering would have given, by a
route that actually works here.

STAR MAGNITUDE DISTRIBUTION
----------------------------
Real starlight follows Pogson's magnitude scale: flux ~ 10^(-0.4 m), and the
*number* of stars brighter than magnitude m grows roughly as N(<m) ~ 10^(0.6 m)
down to the plotted limit - i.e. a few bright stars and a rapidly increasing
population of faint ones. `_sample_magnitudes` inverts that cumulative
distribution directly (rather than eyeballing a power law) so the star field this
module draws has the same "few bright, many faint" shape by construction, not by
tuning a scatter plot to look right.

Star colour follows the same idea used for real stellar colour (hotter -> bluer,
cooler -> redder): each star gets a small random colour-temperature offset from
white, independent of its brightness (real O/B and M stars span the full
brightness range).
"""

from pathlib import Path

import numpy as np
from PIL import Image
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.noise import fbm
from scipy import ndimage


def _sample_magnitudes(rng: np.random.Generator, count: int, m_min: float, m_max: float) -> np.ndarray:
    """Invert N(<m) ~ 10^(0.6*(m - m_min)) to draw `count` magnitudes in [m_min, m_max]."""
    k = 0.6 * np.log(10.0)
    u = rng.uniform(0.0, 1.0, count)
    return m_min - np.log1p(-u * (1.0 - np.exp(-k * (m_max - m_min)))) / k


def _milky_way_layer(shape: tuple, rng: np.random.Generator, intensity: float) -> np.ndarray:
    """A soft, turbulent band of light wrapped once around the sphere (a great circle), tilted at an arbitrary angle so it crosses the equirect image diagonally rather than sitting flat along it - just for visual variety, not any real celestial alignment."""
    h, w = shape
    lon = np.linspace(0.0, 2.0 * np.pi, w, endpoint=False)
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, h)
    LON, LAT = np.meshgrid(lon, lat)
    # Direction vectors on the unit sphere for every pixel.
    x = np.cos(LAT) * np.cos(LON)
    y = np.cos(LAT) * np.sin(LON)
    z = np.sin(LAT)

    tilt = np.deg2rad(63.0)
    # Unit normal of the band's great-circle plane.
    nx, ny, nz = np.sin(tilt), 0.0, np.cos(tilt)
    dist_from_plane = np.abs(x * nx + y * ny + z * nz)  # 0 on the band's great circle

    band_width = 0.16  # radians-ish (dot-product units, not literal angle)
    profile = np.exp(-((dist_from_plane / band_width) ** 2))

    turbulence = fbm(shape, rng, octaves=5, base_cell=max(6.0, w / 48.0), lacunarity=2.1, gain=0.55)
    turbulence = 0.55 + 0.45 * turbulence  # keep strictly positive, mild modulation

    return intensity * profile * turbulence


def generate_starfield_texture(cfg: TerrainConfig, rng: np.random.Generator) -> np.ndarray:
    """Return an (H, W, 3) float64 array in [0, ~1.3] - an equirectangular starfield + Milky Way, ready to gamma-encode and save."""
    h, w = cfg.sky_texture_resolution[1], cfg.sky_texture_resolution[0]
    img = np.zeros((h, w, 3), dtype=np.float64)

    img += _milky_way_layer((h, w), rng, cfg.sky_milky_way_intensity)[..., None] * np.array(
        [0.80, 0.85, 1.0]
    )

    n = cfg.sky_star_count
    # Naked-eye-and-a-bit-fainter range: a handful of "sirius-bright" outliers
    # down to a dense faint population, same shape as a real magnitude-limited
    # star count.
    mag = _sample_magnitudes(rng, n, m_min=-1.4, m_max=7.0)
    flux = 10.0 ** (-0.4 * (mag - mag.min()))
    # Gamma-compress the huge dynamic range (bright/faint flux ratio is ~10^3) so
    # faint stars stay visible as dim points instead of rounding to black, while
    # the brightest few still stand out clearly.
    disp = np.clip(flux, 0.0, None) ** 0.42
    disp /= disp.max()

    # Uniform distribution over the sphere (not the image grid) - cos(polar
    # angle) uniform in [-1, 1], not the polar angle itself, or stars would
    # visibly bunch at the poles.
    az = rng.uniform(0.0, 2.0 * np.pi, n)
    cos_pol = rng.uniform(-1.0, 1.0, n)
    pol = np.arccos(cos_pol)  # 0 .. pi

    xi = (az / (2.0 * np.pi) * w).astype(np.int64) % w
    yi = np.clip((pol / np.pi * h).astype(np.int64), 0, h - 1)

    # Colour: small random offset from white toward blue (hot) or orange/red
    # (cool), independent of brightness - real stellar colour is not correlated
    # with apparent magnitude the way a naive "brighter = whiter" rule would
    # suggest.
    temp = rng.normal(0.0, 1.0, n)
    r = 1.0 + np.clip(temp, 0.0, None) * 0.22
    g = 1.0 - np.abs(temp) * 0.03
    b = 1.0 + np.clip(-temp, 0.0, None) * 0.28
    colour = np.stack([r, g, b], axis=-1)
    colour /= colour.max(axis=-1, keepdims=True)  # keep every star's peak channel at 1.0

    star_rgb = colour * disp[:, None]
    np.add.at(img, (yi, xi), star_rgb)

    # Two tiers of soft point-spread glow, by brightness, so stars read with a
    # range of apparent sizes rather than every one being an identical single
    # pixel - real bright stars visibly bloom more than faint ones. The faintest
    # majority stay crisp single/double-pixel points either way.
    for quantile, sigma, strength in ((0.90, 0.6, 0.55), (0.98, 1.3, 0.9)):
        cut = np.quantile(disp, quantile)
        bright = disp >= cut
        if not np.any(bright):
            continue
        glow = np.zeros((h, w, 3), dtype=np.float64)
        np.add.at(glow, (yi[bright], xi[bright]), star_rgb[bright])
        glow = ndimage.gaussian_filter(glow, sigma=(sigma, sigma, 0), mode="wrap")
        img = np.maximum(img, glow * strength)

    return img


def save_sky_texture(cfg: TerrainConfig, path: Path) -> Path:
    # Independent RNG stream, seeded from but not shared with the terrain
    # generator's own rng - the sky must not consume any of the calls
    # heightmap/craters/rocks depend on for reproducibility (see the module
    # docstrings in heightmap.py / scatter.py). Deterministic per-seed regardless.
    rng = np.random.default_rng((int(cfg.seed) * 1_000_003) ^ 0xA5717A5)
    img = generate_starfield_texture(cfg, rng)
    as_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(as_u8, mode="RGB").save(path)
    return path


def save_sky_dome_mesh(cfg: TerrainConfig, path: Path, lat_segments: int = 32, lon_segments: int = 64) -> Path:
    """Write a UV-sphere OBJ with INWARD-facing winding/normals at cfg.sky_radius_m.

    THE ACTUAL BUG, ROOT-CAUSED: nothing to do with winding or culling at all.
    Two earlier passes (a primitive ``<sphere>`` with ``double_sided``, then a
    custom mesh with deliberately inward-flipped winding) both rendered as solid
    white with no visible starfield. Isolated with a minimal throwaway world (no
    terrain, no rover - just one mesh, one camera, one flat-colour material): a
    plain ``<sphere>`` PRIMITIVE with a flat red ``<material>`` rendered red, but
    the IDENTICAL material on a custom OBJ ``<mesh>`` rendered white regardless of
    winding, of `lighting`, or of whether a `<pbr>` block was present. The one
    variable that fixed it was adding vertex NORMALS (`vn`) to the OBJ and
    referencing them in every face (`f v/vt/vn`, matching terrain_mesh.py's own
    convention) - a mesh with only `v`/`vt` (no `vn`) apparently makes gz-sim's
    OBJ import fall back to a default/placeholder material, silently discarding
    the SDF-specified one. farfield.py already wrote `vn` for unrelated reasons
    (smooth shading) and never hit this; this module originally didn't, and every
    render before this fix shows it.

    Winding is still emitted in BOTH orders (twice the face count - 8192
    triangles for the default tessellation, still trivial) as a second, harmless
    line of defence for visibility from inside the sphere; `double_sided` stays
    set in the material too. Neither was the actual fix, but neither hurts.
    """
    r = cfg.sky_radius_m
    i = np.arange(lat_segments + 1)
    j = np.arange(lon_segments + 1)
    theta = i / lat_segments * np.pi  # 0 (north pole) .. pi (south pole)
    phi = j / lon_segments * 2.0 * np.pi
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")  # (lat+1, lon+1)

    sin_t = np.sin(THETA)
    x = r * sin_t * np.cos(PHI)
    y = r * sin_t * np.sin(PHI)
    z = r * np.cos(THETA)
    verts = np.stack([x, y, z], axis=-1)

    u = PHI / (2.0 * np.pi)
    v = THETA / np.pi

    rows, cols = lat_segments + 1, lon_segments + 1
    flat_v = verts.reshape(-1, 3)
    flat_uv = np.stack([u, v], axis=-1).reshape(-1, 2)

    # Inward-pointing normals (camera sits inside the sphere) - position/-radius,
    # exact for a sphere with no need for cross-product estimation.
    flat_n = -flat_v / r

    r0 = np.arange(lat_segments)[:, None] * cols
    r1 = r0 + cols
    c0 = np.arange(lon_segments)[None, :]
    c1 = c0 + 1
    a = (r0 + c0).ravel()
    b = (r0 + c1).ravel()
    c = (r1 + c1).ravel()
    d = (r1 + c0).ravel()

    # Every triangle is emitted in BOTH winding orders - see docstring: this is
    # belt-and-braces for visibility from inside the sphere, not the actual fix
    # (which was adding `vn` below).
    tris_fwd = np.concatenate([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)], axis=0)
    tris_rev = tris_fwd[:, [0, 2, 1]]
    tris = np.concatenate([tris_fwd, tris_rev], axis=0)

    lines = [
        "# Regolith lunar sky dome - UV sphere carrying the procedural starfield",
        "# texture, every triangle emitted in both winding orders. See sky.py -",
        "# critically, WITH vertex normals: a mesh missing them silently loses its",
        "# SDF-specified material on this gz-sim/Ogre2 build (see save_sky_dome_mesh).",
        "o sky_dome",
    ]
    lines += ["v {:.3f} {:.3f} {:.3f}".format(*p) for p in flat_v]
    lines += ["vt {:.5f} {:.5f}".format(*uv) for uv in flat_uv]
    lines += ["vn {:.4f} {:.4f} {:.4f}".format(*n) for n in flat_n]
    lines += [
        "f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}".format(t[0] + 1, t[1] + 1, t[2] + 1)
        for t in tris
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def sky_model_sdf(cfg: TerrainConfig, texture_path: Path, mesh_path: Path) -> str:
    """`<model>` block for the sky dome.

    THE SECOND BUG, ROOT-CAUSED: after fixing the missing-`vn` bug above, the sky
    still rendered flat white - but now with a very specific, exactly-reproducible
    fallback colour (186.8, 186.8, 190.9 mean, to several decimal places, across
    every failing variant tried). Bisected with the same minimal throwaway-world
    method: ``<pbr><metal><albedo_map>`` on a ``<mesh>`` visual renders correctly
    with DEFAULT (lit) shading, and a flat/no-texture ``<pbr>`` block renders
    correctly with ``<lighting>false</lighting>`` - but the SPECIFIC combination of
    ``<lighting>false</lighting>`` + ``<albedo_map>`` on a ``<mesh>`` always falls
    back to that same default material, for every PNG tried (including a PNG
    already proven to load fine elsewhere - Earth's own albedo texture). Unlit
    textured PBR meshes appear to not be supported on this gz-rendering/Ogre2
    build; only lit ones are.

    The fix keeps `lighting` at its default (true) and moves the starfield texture
    into `<emissive_map>` instead of `<albedo_map>`, with `ambient`/`diffuse`/
    `specular` all zeroed. Emissive is additive and, on this build, unaffected by
    the N-dot-L/shadow terms that the (zeroed-out) diffuse channel would otherwise
    carry - confirmed by rendering this exact material and checking the output is
    a dark image with visible bright points (stars), not the flat white fallback
    or a shaded/lit-looking gradient. Net effect is the same as true unlit
    rendering would have given: the sky shows the texture as authored, regardless
    of sun direction or ambient level, and `cast_shadows=false` still keeps it out
    of the shadow pass entirely either way.
    """
    return f"""    <model name="sky_dome">
      <static>true</static>
      <pose>0 0 0 0 0 0</pose>
      <link name="link">
        <!-- Visual only, deliberately: no collision geometry here, and this model is
             never written to manifest.json, so the costmap/planner - which only
             ever read the manifest and the collision geometry - cannot see it. -->
        <visual name="sky_visual">
          <cast_shadows>false</cast_shadows>
          <geometry>
            <mesh><uri>file://{mesh_path}</uri></mesh>
          </geometry>
          <material>
            <double_sided>true</double_sided>
            <ambient>0 0 0 1</ambient>
            <diffuse>0 0 0 1</diffuse>
            <specular>0 0 0 1</specular>
            <pbr>
              <metal>
                <emissive_map>file://{texture_path}</emissive_map>
                <roughness>1.0</roughness>
                <metalness>0.0</metalness>
              </metal>
            </pbr>
          </material>
        </visual>
      </link>
    </model>
"""
