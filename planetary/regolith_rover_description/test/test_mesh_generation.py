# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for mesh_gen/generate_rover_meshes.py.

Run directly: `python3 -m pytest test/` from this package's directory (this package is
ament_cmake, not ament_python, and - like regolith_bringup's own test/ directory - these
are not wired into CMakeLists.txt/colcon test; they are run the same way).

Two things matter about generated, committed mesh assets that a normal unit test
wouldn't catch:
  1. Regeneration must be BYTE-IDENTICAL (the generator is seeded and has no
     wall-clock/random-without-seed inputs) - if it ever drifts, the committed .obj/.png
     files silently stop matching what `python3 generate_rover_meshes.py` would produce,
     which defeats the point of keeping the generator in the repo at all.
  2. Every mesh's bounding box must actually fit the envelope it is meant to dress -
     this is the check that would have caught, for example, an early draft of the mast
     that added ~0.44 m above the deck instead of the intended ~0.2 m.
"""

import hashlib
import re
import sys
from pathlib import Path

import pytest

MESH_GEN_DIR = Path(__file__).resolve().parent.parent / "mesh_gen"
MESHES_DIR = Path(__file__).resolve().parent.parent / "meshes"
URDF_PATH = Path(__file__).resolve().parent.parent / "urdf" / "regolith_rover.urdf.xacro"

sys.path.insert(0, str(MESH_GEN_DIR))
import generate_rover_meshes as gen  # noqa: E402


def _xacro_property(name: str) -> float:
    """Pull a `<xacro:property name="X" value="Y"/>` value straight out of the URDF
    source, so this test is checking against the URDF's OWN numbers rather than a
    second hardcoded copy that could drift from it independently."""
    text = URDF_PATH.read_text()
    m = re.search(rf'<xacro:property name="{name}" value="([0-9.]+)"\s*/>', text)
    assert m, f"could not find xacro:property '{name}' in {URDF_PATH}"
    return float(m.group(1))


def test_generation_is_deterministic(tmp_path):
    """Two independent generation runs (fresh RNG each time, same fixed seed) must
    produce byte-identical output - every .obj and every .png."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    gen.generate_all(out_a)
    gen.generate_all(out_b)

    files_a = sorted(p.name for p in out_a.iterdir())
    files_b = sorted(p.name for p in out_b.iterdir())
    assert files_a == files_b, "the two runs produced different sets of output files"
    assert files_a, "generate_all produced no files at all"

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    mismatches = [
        name for name in files_a
        if sha256(out_a / name) != sha256(out_b / name)
    ]
    assert not mismatches, f"non-deterministic output in: {mismatches}"


def test_committed_meshes_match_the_generator(tmp_path):
    """The .obj/.png files committed under meshes/ are exactly what running the
    generator produces right now - i.e. nobody hand-edited a mesh, and nobody changed
    the generator without regenerating. Regressing this is exactly the failure mode
    the honest-reporting convention in this repo warns about: a stale shipped asset
    silently diverging from the tool that claims to produce it."""
    fresh = tmp_path / "fresh"
    gen.generate_all(fresh)

    committed = sorted(p.name for p in MESHES_DIR.glob("*"))
    freshly_generated = sorted(p.name for p in fresh.glob("*"))
    assert committed == freshly_generated, (
        "meshes/ does not contain exactly the files generate_all() produces - "
        f"committed={committed} generated={freshly_generated}"
    )

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    stale = [
        name for name in committed
        if sha256(MESHES_DIR / name) != sha256(fresh / name)
    ]
    assert not stale, (
        f"committed mesh(es)/texture(s) are stale vs. the generator: {stale} - "
        "run `python3 mesh_gen/generate_rover_meshes.py` and commit the result"
    )


def _obj_bounds(path: Path):
    import numpy as np

    verts = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
    arr = np.array(verts)
    return arr.min(axis=0), arr.max(axis=0)


@pytest.mark.parametrize("name,axis,expected_min,expected_max", [
    # Chassis hull must fill (not exceed by more than a few mm - trim rings/bosses are
    # allowed a small protrusion) the chassis box on all 3 axes.
    ("chassis_hull", 0, 0.36, 0.42),   # length (x), box is 0.40
    ("chassis_hull", 1, 0.30, 0.36),   # width (y), box is 0.34
    ("chassis_hull", 2, 0.10, 0.12),   # height (z), box is 0.11
    # Wheel hub must fill the wheel envelope: diameter 2*0.09=0.18 (grousers add a
    # little), width 0.06.
    ("wheel_hub", 0, 0.17, 0.21),
    ("wheel_hub", 1, 0.17, 0.21),
    ("wheel_hub", 2, 0.055, 0.065),
])
def test_mesh_extents_match_urdf_dimensions(tmp_path, name, axis, expected_min, expected_max):
    out = tmp_path / "extent_check"
    gen.generate_all(out)
    mins, maxs = _obj_bounds(out / f"{name}.obj")
    extent = maxs[axis] - mins[axis]
    assert expected_min <= extent <= expected_max, (
        f"{name} axis {axis} extent {extent:.4f} m outside expected "
        f"[{expected_min}, {expected_max}] m - mesh no longer matches the URDF envelope"
    )


def test_chassis_hull_extent_tracks_live_urdf_properties():
    """Belt-and-braces version of the extent check above: reads chassis_length/width/
    height straight out of the URDF (not a hardcoded copy) and checks the freshly
    generated hull mesh against THOSE numbers directly, so a future change to the
    chassis dimensions in the URDF is caught here even if the parametrized ranges above
    are never updated."""
    length = _xacro_property("chassis_length")
    width = _xacro_property("chassis_width")
    height = _xacro_property("chassis_height")

    mesh = gen.build_chassis_hull()
    mins, maxs = mesh.bounds()
    extent = maxs - mins

    # Allow a few mm of slack for the trim rings / access-panel bosses, which are
    # deliberately allowed to protrude slightly past the nominal box (see
    # build_chassis_hull's docstring) since this is a VISUAL-only mesh with no tie to
    # collision.
    slack = 0.014
    assert abs(extent[0] - length) <= slack, f"hull x-extent {extent[0]} vs chassis_length {length}"
    assert abs(extent[1] - width) <= slack, f"hull y-extent {extent[1]} vs chassis_width {width}"
    assert abs(extent[2] - height) <= slack, f"hull z-extent {extent[2]} vs chassis_height {height}"


def test_wheel_hub_radius_and_width_track_live_urdf_properties():
    wheel_radius = _xacro_property("wheel_radius")
    wheel_width = _xacro_property("wheel_width")

    mesh, _ = gen.build_wheel_hub()
    mins, maxs = mesh.bounds()
    extent = maxs - mins

    # Grousers protrude ~1 cm past the bare tire radius on purpose (see
    # build_wheel_hub's grouser_height) - so the mesh's own (x, y) footprint is
    # expected to run a bit past 2 * wheel_radius, not match it exactly.
    grouser_slack = 0.022
    assert abs(extent[0] - 2 * wheel_radius) <= grouser_slack
    assert abs(extent[1] - 2 * wheel_radius) <= grouser_slack
    assert abs(extent[2] - wheel_width) <= 0.006


def test_no_pure_black_or_pure_white_materials():
    """This world's lighting is a harsh low sun over near-black ambient (see
    generate_hull_region's docstring) - a literally (0,0,0) albedo would be
    indistinguishable from "nothing rendered" in exactly the way that made earlier
    drafts of this rover hard to read, and (1,1,1) would clip and lose all texture
    detail. Every generated albedo map should stay inside a sane, non-clipping band."""
    import numpy as np
    from PIL import Image

    assert MESHES_DIR.exists(), "meshes/ has not been generated - run the generator first"
    for path in sorted(MESHES_DIR.glob("*_albedo.png")):
        arr = np.asarray(Image.open(path), dtype=np.float64) / 255.0
        assert arr.max() < 0.995, f"{path.name} clips to pure white somewhere"
        # A handful of near-black pixels (e.g. rivet shadow cores, lens centres) is
        # fine; the map as a WHOLE must not be dominated by pure black.
        near_black_fraction = (arr.max(axis=-1) < 0.02).mean()
        assert near_black_fraction < 0.15, (
            f"{path.name} is {near_black_fraction:.0%} near-pure-black pixels"
        )


def test_triangle_budget(tmp_path):
    """Whole-rover triangle budget from the brief: <= 60k. Report generously below
    that (see the report handed to the caller for the actual measured total)."""
    assets = gen.generate_all(tmp_path / "budget_check")
    per_part = {a.name: a.tri_count for a in assets}
    wheel_tris = per_part["wheel_hub"]
    chassis_parts_tris = sum(v for k, v in per_part.items() if k != "wheel_hub")
    total_scene_tris = chassis_parts_tris + 4 * wheel_tris
    assert total_scene_tris <= 60000, f"{total_scene_tris} triangles exceeds the 60k budget"
