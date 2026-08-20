# regolith_visual_odometry

RGB-D visual odometry for the Regolith rover. Publishes **body-frame velocity**
on `/vo/odom`, which the EKF fuses as `odom1` (see
`regolith_bringup/config/ekf.yaml`).

## Why it exists

M4's acceptance was **0/3**, with the rover arriving 3.1-13.1 m from its goal.
The cause was pinned by measurement, not argument: on every seed the true error
equalled the EKF's own drift plus the stopping tolerance, to within centimetres.
The rover was arriving exactly where it believed the goal was.

About **10% of this rover's motion over boulder-strewn regolith is lateral
slide**, and nothing in the previous sensor suite could see it:

| sensor | observes | cannot observe |
|---|---|---|
| wheel odometry | `vx`, `vyaw` | `vy` - a differential-drive model assumes it is zero *by construction* |
| IMU | orientation, angular rate | position error of any kind |

So the lateral error accumulated as a random walk that nothing ever corrected.
Feeding the same build an absolute position reference (0.5 m, 1 Hz) turned 0/3
into 3/3, which is what identified the estimator - and only the estimator - as
the gap. This package is the real sensor that fills it.

## What it does and does not promise

Visual odometry is a **relative** sensor. It is *not* the oracle that produced
that 3/3: the oracle handed the filter absolute position and drove EKF
divergence to 0.00 m, which nothing onboard can do. VO makes the drift *grow
more slowly* by observing the term that was structurally invisible. Published
planetary-rover VO runs on the order of 1-2% of distance travelled; over this
acceptance's ~110 m traverses that is 1-2 m against a 1.5 m bar. Close, not
comfortable - and the real number is whatever the acceptance run measures.

## How it works

1. Shi-Tomasi corners on the keyframe, tracked into the current frame with
   Lucas-Kanade, filtered by a forward/backward consistency check (regolith at a
   low sun angle is full of near-identical shadowed pits, and a forward-only
   track slides corners onto their neighbours).
2. The keyframe's corners get metric 3D positions from its depth image.
3. `solvePnPRansac` fits the current camera pose to those landmarks.
4. The result is rotated optical → camera body → `base_link` (the mount comes
   from **TF**, not hardcoded numbers, so moving the camera in the xacro cannot
   leave this node computing motion for a camera that has moved).
5. The lever arm is subtracted: the camera sits 0.2 m ahead of `base_link`, so
   yawing translates the camera even when `base_link` does not.

**Depth is what makes it metric.** A single camera cannot observe scale at all -
monocular VO recovers the shape of a trajectory but not its size, which is
useless for correcting a metre-scale error.

### It does not run at frame rate, on purpose

Estimating between adjacent frames is worse than useless here. At 0.2 m/s a
0.1 s interval moves the camera 2 cm - about **1 px** of optical flow, against a
measured **0.74 px** of feature-tracker noise. Signal and noise are the same
size:

| baseline | 0.1 s | 0.2 s | 0.3 s | 0.5 s | 0.8 s |
|---|---|---|---|---|---|
| median flow | 1.05 px | 2.41 px | 2.93 px | 5.74 px | 10.15 px |
| recovered speed (true 0.200 m/s) | 0.056 | 0.200 | 0.192 | 0.224 | 0.201 |

So the node holds a **keyframe** and estimates only once ~0.4 s of motion has
accumulated: ~2.5 updates/s at a worst-case error of 0.051 m/s. That is ample -
this corrects a drift that grows over minutes, not a control loop.

## Measured accuracy on real frames

873 frames captured from a live run of this world, judged against
`/ground_truth/pose` offline (the estimator never sees it), plus a live
confirmation in the running sim:

| | offline, 130 pairs | live, 90 pairs |
|---|---|---|
| **`vy` error** (the fused channel) | **+0.000 ± 0.018 m/s** | **+0.001 ± 0.020 m/s** |
| `vx` error (published, not fused) | −0.060 ± 0.053 m/s | −0.067 ± 0.050 m/s |
| frame pairs usable | 71% | 63% |

The reported sigma (0.020) matches `vy`'s measured spread almost exactly, so the
filter is being told the truth about how much to trust it.

**`vx` is biased low and is therefore not fused.** How low depends on what the
rover is doing, which is worth stating rather than collapsing to one number:

| condition | `vx` bias | `vy` bias |
|---|---|---|
| straight-line driving (n=92) | −0.069 m/s (−35%) | +0.003 ± 0.016 |
| turning (n=38) | −0.036 m/s (−18%) | −0.005 ± 0.022 |
| one live cruise sample (n=51) | −0.017 m/s (−9%) | - |

Wheel odometry measures forward distance to about 1% once slip is gated, so
fusing this would inject a systematic error into the one term the existing
sensors already handle well - and a filter that believes it has travelled less
than it has drives past its goal.

The asymmetry is the interesting part: `vy` is unbiased under exactly the same
conditions that bias `vx`. The leading explanation - consistent with the data
but not separately proven - is Lucas-Kanade's translation-only motion model.
Driving forward over near ground makes texture *expand* rapidly between frames,
and a fixed-window tracker with no affine term systematically under-tracks
expansion; sliding sideways produces no scale change at all, so it does not.
That would also explain why the bias is worst in straight-line driving and why
the synthetic scenes, which are less foreshortened, show only −5%.

### Three defects this only found by being measured on real data

The first live run refused **100%** of frame pairs. All three causes were
invisible in synthetic testing:

1. **Corners were being found in the sky.** The strongest contrast in a lunar
   scene is the skyline, and sky has no range - 25 corners found, 8 with usable
   depth, against an inlier floor of 20. Fixed by masking the detector to
   pixels that have depth.
2. **The image is dim and flat** (values 10-124, sd 9.8). Corner scoring is
   relative, so a raw frame yielded 25 corners against a 400 budget. Fixed with
   CLAHE before detection.
3. **PnP sometimes returns a confident, wildly wrong pose.** 4% of estimates
   were catastrophic (up to 46 m/s for a 0.2 m/s rover) - and they are trivially
   identifiable: median reprojection RMS of **87 px** against **0.67 px** for
   the other 96%. Rejected by `max_reprojection_rms_px`. Without that gate those
   4% alone move the mean velocity error from 0.05 to 1.1 m/s.

### Synthetic accuracy

Against ray-traced scenes with an exact known answer (12 seeds, `test/`):

| | true | recovered (mean) | sd of one estimate |
|---|---|---|---|
| forward | 0.200 m/s | 0.191 | 0.017 |
| lateral, no slip | 0.000 m/s | −0.002 | 0.011 |
| lateral, 10% slip | 0.020 m/s | 0.020 | 0.026 |

**A single estimate cannot resolve one interval's slip** (0.020 m/s signal
against 0.026 m/s noise). It works because the noise is *unbiased* and the EKF
integrates thousands of them. The test asserts that averaged slip and no-slip
are distinguishable, rather than making a single-shot assertion loose enough to
pass on a dead channel.

## It never touches ground truth

Same rule as `wheel_slip_node.py`: this is a localisation input, and an
acceptance number produced by a localisation input that consulted
`/ground_truth/pose` would be meaningless. Its inputs are the two cameras and
nothing else. Contrast `absolute_reference_relay.py`, which *is* an oracle,
announces itself at four layers, and is default-off.

## Running it

On by default in `hello_moon.launch.py`:

```bash
ros2 launch regolith_bringup hello_moon.launch.py seed:=42
ros2 launch regolith_bringup hello_moon.launch.py seed:=42 visual_odometry:=false  # the 0/3 baseline
```

## Tests

```bash
cd planetary/regolith_visual_odometry && PYTHONPATH=. python3 -m pytest test -q
```

They ray-trace a textured, *rough* ground surface from two poses a known rigid
motion apart, so the expected answer is exact and no simulator is involved. The
relief matters: a perfectly flat plane makes every landmark coplanar, which is a
degenerate configuration for PnP - on a flat plane this estimator reports
0.139 m/s for a true 0.200 and invents 0.09 m/s of sideways drift out of a pure
turn. Real regolith is never that case, so a flat test plane would have been
testing a situation the rover is never in.
