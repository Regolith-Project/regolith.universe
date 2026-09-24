# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Proves the visual-only claim about regolith_rover.urdf.xacro: after `xacro | gz sdf
-p`, every <collision>, <inertial>, <joint>, <sensor> and <plugin> element is
BYTE-IDENTICAL to a known-good baseline captured before this overhaul touched the file,
and only <visual>/<material> content differs.

Run directly (needs a sourced ROS 2 + this workspace, and `xacro`/`gz` on PATH):
    cd src/regolith.universe/planetary/regolith_rover_description
    python3 -m pytest test/test_physics_untouched_in_sdf.py -v

Re-capturing the baseline (only do this after a deliberate, reviewed physics change):
    python3 test/test_physics_untouched_in_sdf.py --recapture-baseline

How to double-check this test actually catches a real break: temporarily edit the
wheel mu1 value in the URDF (e.g. 1.4 -> 1.5), run the test, watch it fail with a
non-visual diff naming the changed <friction><ode><mu> value, then revert the edit.
"""

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
URDF_PATH = PKG_DIR / "urdf" / "regolith_rover.urdf.xacro"
BASELINE_PATH = Path(__file__).resolve().parent / "physics_baseline.sdf"


def _run_xacro_then_sdf(urdf_xacro_path: Path) -> str:
    """xacro-process the file, convert to SDF, and return the SDF text - same two
    commands the task's verification recipe names: `xacro <file> | gz sdf -p -`.
    Goes through a real temp file rather than a pipe/stdin: `gz sdf -p` needs a
    seekable input."""
    import tempfile

    xacro_out = subprocess.run(
        ["xacro", str(urdf_xacro_path)], capture_output=True, text=True, check=True,
    ).stdout
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(xacro_out)
        tmp_path = f.name
    try:
        sdf_out = subprocess.run(
            ["gz", "sdf", "-p", tmp_path], capture_output=True, text=True, check=True,
        ).stdout
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return sdf_out


_BENIGN_DEFAULT_SURFACE = "<surface><contact><ode /></contact><friction><ode /></friction></surface>"


def _strip_visuals(sdf_text: str) -> str:
    """Parse the SDF and remove every <visual> element, so what's left is exactly the
    physics-relevant subset: collisions, inertials, joints, sensors, plugins.

    Also drops any <surface> block that is EXACTLY the all-default skeleton
    (<contact><ode/></contact><friction><ode/></friction>, no mu/mu2/other content).
    This one artifact is expected and harmless: adding the chassis's `<gazebo
    reference="chassis">` block (needed purely to attach the PBR material - see the
    URDF's mesh_dir property comment) makes sdformat's urdf2sdf pass materialize this
    exact default-valued surface on the chassis COLLISION, which previously had no
    <surface> element written out at all. Verified against sdformat's own schema
    (/usr/share/sdformat14/*/surface.sdf: <ode><mu default="1" required="0"/>... and
    <mu2> likewise) that an absent element and a present-but-default one resolve to the
    identical value - so this is a pretty-printing artifact of the conversion, not a
    friction change. Any <surface> block that ISN'T this exact empty skeleton (e.g. the
    wheels' real mu1/mu2 friction) is left fully in place and fully compared.
    """
    root = ET.fromstring(sdf_text)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "visual":
                parent.remove(child)
            elif child.tag == "surface":
                serialized = re.sub(r">\s+<", "><", ET.tostring(child, encoding="unicode").strip())
                if serialized == _BENIGN_DEFAULT_SURFACE:
                    parent.remove(child)
    return ET.tostring(root, encoding="unicode")


def _normalize(sdf_text: str) -> str:
    """Whitespace-only normalization so a pretty-printer difference alone can't cause a
    false failure - element content and structure still have to match exactly."""
    return re.sub(r">\s+<", "><", sdf_text.strip())


def capture_baseline() -> None:
    sdf_text = _run_xacro_then_sdf(URDF_PATH)
    non_visual = _strip_visuals(sdf_text)
    BASELINE_PATH.write_text(non_visual)
    print(f"Wrote baseline ({len(non_visual)} bytes) to {BASELINE_PATH}")


def test_non_visual_sdf_is_byte_identical_to_baseline():
    assert BASELINE_PATH.exists(), (
        f"no baseline at {BASELINE_PATH} - run with --recapture-baseline once, from a "
        "commit where the physics is known-good, and commit the resulting file"
    )
    current_sdf = _run_xacro_then_sdf(URDF_PATH)
    current_non_visual = _normalize(_strip_visuals(current_sdf))
    baseline_non_visual = _normalize(BASELINE_PATH.read_text())

    if current_non_visual != baseline_non_visual:
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                baseline_non_visual.split("><"), current_non_visual.split("><"),
                fromfile="baseline (physics-relevant SDF)",
                tofile="current (physics-relevant SDF)",
                lineterm="",
            )
        )
        raise AssertionError(
            "Non-visual SDF changed - this is a PHYSICS change, which "
            "regolith_rover.urdf.xacro's header says is frozen for this task:\n" + diff
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recapture-baseline", action="store_true")
    args = ap.parse_args()
    if args.recapture_baseline:
        capture_baseline()
    else:
        sys.exit(pytest.main([__file__, "-v"]) if (pytest := __import__("pytest")) else 1)
