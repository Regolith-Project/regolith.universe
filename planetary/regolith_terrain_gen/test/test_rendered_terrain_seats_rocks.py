# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""Do the rocks float IN THE PICTURE? Asked of a real Gazebo GUI screenshot.

Every other test in this package grades geometry on disk. Three rounds of the "floating
rocks" report were closed against tests like that, and the fourth round proved why that
is not enough: the placement was provably correct - every rock bedded 3-14 cm into the
surface the files describe - and the user was still, correctly, seeing boulders in the
sky. The ground was being DRAWN somewhere other than where the data put it, by a render
path no geometry test can observe. (Ogre-Next's Terra LOD; see terrain_mesh.py.)

So this test looks at pixels, from the only render path the user ever reports on: the
Gazebo GUI window, captured with `import -window` (the GUI is an XWayland client - see
PROGRESS.md, which also retracts an earlier claim that it could not be captured).

THE CRITERION. The terrain silhouette is the upper envelope of the ground, so scanning
a column of the frame downwards, sky can give way to ground exactly once. A column that
reads sky, object, sky again, then ground has something standing clear of the horizon
with daylight underneath it. That is the report, stated in pixels, and it needs no model
of the terrain at all.

The sky is the world's flat <background_color>, rendered as one exact RGB value with no
gradient and no dithering, so the sky mask is an equality test rather than a threshold -
which is what makes this measurable rather than eyeballed.

MEASURED when it was written, seed 42, the pose below, 1200 px wide:
    <heightmap> visual (Terra LOD):  111 columns detached, widest gap 18 px
    <mesh> visual (what ships now):    0 columns detached
Boulders were visibly in the sky in the first frame and sitting on the ground in the
second.

This test needs a GPU, a display and about a minute per launch, so it skips when it
cannot run rather than failing. A skip here means the floating-rocks regression is NOT
covered by that run - it is the only check in the suite that can see one.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import pytest
from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.generate import generate_world
from regolith_terrain_gen.heightmap import build_heightmap

# A detached blob has to be at least this tall, over a sky gap at least this deep,
# before it counts. Below that the run structure is antialiasing on a boulder's own
# silhouette edge: measured on the shipped mesh world, the residue is 7 columns of
# 2-3 px gaps, against 111 columns and gaps up to 18 px for a genuinely floating field.
MIN_OBJECT_PX = 3
MIN_SKY_GAP_PX = 4

# Fractions, not pixels - the window's size is whatever the window manager gives it.
# Trims the title bar at the top and the playback toolbar at the bottom.
VIEWPORT = (0.06, 0.92, 0.02, 0.99)

_TOOLS = ("gz", "xdotool", "import")


def _requirements():
    missing = [t for t in _TOOLS if shutil.which(t) is None]
    if missing:
        return f"needs {', '.join(missing)} on PATH"
    if not os.environ.get("DISPLAY"):
        return "needs an X display (DISPLAY is unset)"
    return None


pytestmark = pytest.mark.skipif(_requirements() is not None, reason=_requirements() or "")


def _sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def _horizon_pose(cfg: TerrainConfig) -> str:
    """A pose that puts a long stretch of terrain against the sky.

    The opening GUI camera sits 7 m from the rover looking down at it, which never sees
    the horizon and so cannot see this bug at all - distance is the whole mechanism.
    This one stands near a corner and looks across the full diagonal of the world.
    """
    _raw, _visual, _craters, elevation = build_heightmap(cfg, np.random.default_rng(cfg.seed))
    x = y = -0.475 * cfg.world_size_m
    z = elevation(x, y) + 8.0
    return f"{x:.2f} {y:.2f} {z:.2f} 0 0.02 0.785"


def _capture(world_sdf: Path, pose: str, out_png: Path, timeout_s=90) -> Path:
    """Launch the GUI on this world at this pose and screenshot the window."""
    text = re.sub(
        r"<camera_pose>[^<]*</camera_pose>",
        f"<camera_pose>{pose}</camera_pose>",
        world_sdf.read_text(),
    )
    shot_world = world_sdf.parent / "horizon_probe.sdf"
    shot_world.write_text(text)

    # An orphaned window from an earlier launch has already produced one screenshot of
    # the wrong world (PROGRESS.md); start clean and require exactly one window. The
    # bracket in the pattern stops pgrep/pkill matching their own command line.
    _sh('pkill -f "gz[ ]sim"')
    time.sleep(2)
    proc = subprocess.Popen(
        ["gz", "sim", "-r", "-v", "1", str(shot_world)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + timeout_s
        windows = []
        while time.time() < deadline:
            time.sleep(1)
            windows = _sh('xdotool search --name "^Gazebo Sim$"').split()
            if windows:
                break
        assert len(windows) == 1, f"expected 1 Gazebo window, found {len(windows)}"

        # Poll until the frame settles: the scene streams in, and a screenshot taken
        # too early shows an empty or half-built world that trivially "passes".
        previous = None
        while time.time() < deadline:
            time.sleep(4)
            _sh(f"import -window {windows[0]} {out_png}")
            frame = _read_viewport(out_png)
            sky_frac = float(np.mean(_sky_mask(frame)))
            if 0.05 < sky_frac < 0.95 and previous is not None and abs(sky_frac - previous) < 0.002:
                return out_png
            previous = sky_frac
        raise AssertionError(
            f"the GUI never settled into a framed horizon within {timeout_s}s "
            f"(last sky fraction {previous})"
        )
    finally:
        proc.terminate()
        time.sleep(1)
        _sh('pkill -f "gz[ ]sim"')


def _read_viewport(png: Path) -> np.ndarray:
    from PIL import Image

    im = np.array(Image.open(png).convert("RGB")).astype(int)
    h, w = im.shape[:2]
    t, b, l, r = VIEWPORT
    return im[int(t * h) : int(b * h), int(l * w) : int(r * w)]


def _sky_mask(view: np.ndarray) -> np.ndarray:
    """Sky is the world's flat background colour - one exact RGB value, read off the
    top of the frame rather than assumed, so no threshold is involved."""
    band = view[: max(1, view.shape[0] // 10)].reshape(-1, 3)
    colours, counts = np.unique(band, axis=0, return_counts=True)
    sky = colours[counts.argmax()]
    assert counts.max() / len(band) > 0.98, (
        "the top of the frame is not flat sky - the probe camera is not framing the "
        "horizon, so this test would prove nothing"
    )
    return np.all(view == sky, axis=-1)


def detached_columns(png: Path):
    """Columns whose top-down run structure is sky, object, sky, ground."""
    sky = _sky_mask(_read_viewport(png))
    hits = []
    for c in range(sky.shape[1]):
        column = sky[:, c]
        edges = np.flatnonzero(np.diff(column)) + 1
        bounds = np.concatenate([[0], edges, [len(column)]])
        runs = [(bool(column[s]), int(e - s)) for s, e in zip(bounds[:-1], bounds[1:])]
        for k in range(len(runs) - 3):
            (is_sky, _), (o_sky, o_len), (g_sky, g_len), (n_sky, _) = runs[k : k + 4]
            if (
                is_sky
                and not o_sky
                and g_sky
                and not n_sky
                and o_len >= MIN_OBJECT_PX
                and g_len >= MIN_SKY_GAP_PX
            ):
                hits.append((c, o_len, g_len))
                break
    return hits


@pytest.mark.render
def test_no_rock_stands_in_the_sky_at_range(tmp_path):
    cfg = TerrainConfig(seed=42)
    world = generate_world(cfg, tmp_path / "world", start_paused=False)
    shot = _capture(world, _horizon_pose(cfg), tmp_path / "horizon.png")

    hits = detached_columns(shot)
    widest = max((h[2] for h in hits), default=0)
    assert not hits, (
        f"{len(hits)} image columns show a boulder standing clear of the terrain "
        f"silhouette with sky underneath it (widest gap {widest} px). The ground is "
        f"being drawn below where the geometry puts it - check that the terrain visual "
        f"is still a <mesh> and has not gone back to <heightmap>. Frame: {shot}"
    )


@pytest.mark.render
def test_the_check_can_actually_fail(tmp_path):
    """Guards the guard, by the standard this package learned the hard way: a check for
    floating rocks that has never been seen to fail is not evidence of anything."""
    cfg = TerrainConfig(seed=42)
    world = generate_world(cfg, tmp_path / "world", start_paused=False)

    # Lift every boulder half a metre, in the SDF gz is about to render.
    def raise_rock(match):
        x, y, z, rest = match.group(2).split(" ", 3)
        return f"{match.group(1)}{x} {y} {float(z) + 0.5:.3f} {rest}</pose>"

    lifted = re.sub(
        r'(<model name="rock_\d+">\s*<static>true</static>\s*<pose>)([^<]+)</pose>',
        raise_rock,
        world.read_text(),
    )
    assert lifted != world.read_text(), "no rock poses were rewritten"
    world.write_text(lifted)

    shot = _capture(world, _horizon_pose(cfg), tmp_path / "lifted.png")
    assert detached_columns(shot), (
        "raising every rock by 0.5 m did not register as floating - this detector "
        "cannot see the bug it exists for"
    )
