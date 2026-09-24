# regolith_bringup

Launch files, EKF config, and mission scripts for the Regolith hello-world
demo (the RViz config lives in `regolith_rover_description`). This is the package the plan's package table designates as the
integration point - its launch files reference `regolith_terrain_gen`,
`regolith_rover_description`, `regolith_planner`, and `regolith_costmap` by
name, which is why it lives here in `regolith.universe/planetary/` alongside
those packages rather than in the `regolith` meta-repo.

## Launch files

- `terrain_only.launch.py` - generates a procedural lunar terrain world for
  the given `seed` (default `42`) via `regolith_terrain_gen` and opens it in
  Gazebo. `ros2 launch regolith_bringup terrain_only.launch.py seed:=42`
- `teleop_demo.launch.py` - generates terrain, spawns the
  `regolith_rover_description` rover at the actual local terrain elevation
  (never a hard-coded height - see PROGRESS.md M2 for why that matters),
  and bridges `cmd_vel`/`odom`/`imu`/`camera`/`camera_info`/`joint_states`/`tf`
  between ROS and Gazebo. `ros2 launch regolith_bringup teleop_demo.launch.py
seed:=42`, then in another terminal:
  `ros2 run teleop_twist_keyboard teleop_twist_keyboard`
- `localization_demo.launch.py` - everything `teleop_demo` does, plus fuses
  wheel odometry + IMU into an estimated pose (`robot_localization`'s
  `ekf_node`) and bridges Gazebo's ground truth separately for comparison
  (`/ground_truth/pose`, never fed into the estimator). See PROGRESS.md M3.
- `autonomous_demo.launch.py` - everything `localization_demo` does, plus
  `regolith_costmap` + `regolith_planner` + `regolith_vehicle_interface`:
  click "2D Goal Pose" in RViz and the rover plans and drives there (this
  launch file doesn't start RViz itself - open it manually, or use
  `hello_moon.launch.py` below, which does). See PROGRESS.md's "M4
  acceptance check: full 60-100 m / 3-consecutive-run result" for current
  status - the terrain-collision flip issue referenced by older notes below
  was root-caused and fixed, and the full-distance/3-consecutive-seed
  acceptance check now passes.
- `hello_moon.launch.py` - the main entry point, superseding
  `autonomous_demo.launch.py` above (which it's built on and keeps
  identical behaviour to). Also opens RViz with the rover config (disable
  with `rviz:=false`) and adds a `mission` argument:
  `ros2 launch regolith_bringup hello_moon.launch.py seed:=42` behaves
  like `autonomous_demo.launch.py` (click a goal yourself in RViz);
  `... mission:=tour` additionally runs `tour_mission.py`, a 5-waypoint
  loop, with no interaction needed. The route is **derived from the live
  `/costmap`** rather than hardcoded (see `regolith_planner/tour.py`):
  every waypoint is somewhere the planner will accept, every leg is checked
  with the same A* that will drive it, and legs are chosen to cross terrain
  the rover has to route around. Deterministic from `seed`. This is what
  `scripts/demo.sh` in the meta-repo launches. See PROGRESS.md M5 for the
  current state, including a confirmed instance of the M4 flip issue
  occurring during an unattended tour run (also since fixed - see above).
  Pass `headless:=true` to skip the Gazebo GUI window entirely (server-only,
  `-s`) for unattended/automated runs - distinct from `rviz:=false`, which
  only skips RViz.

  `mission_markers_node.py` flags the start point and the mission's goals in
  **both** windows - a green flag where the rover set off, an amber one at
  each planned waypoint, and a taller red one that follows whichever goal is
  currently live; RViz gets the same set as a labelled `MarkerArray` on
  `/mission_markers`. Disable with `markers:=false`. The flags have no
  collision and nothing in the control loop reads them, so the rover drives
  straight through them.

  Note the frames: goals are published in `odom`, the Gazebo flags stand in
  the world frame, and the two share an origin at the spawn point. A rover
  parked visibly short of a flag while reporting the goal reached is showing
  you its localisation error, not a misplaced flag.

  Pass `record_video:=true` to record the onboard camera straight to an mp4
  via gz-sim's server-side `CameraVideoRecorder` plugin - this bypasses the
  GUI/desktop-compositor entirely, which matters under WSLg (see
  PROGRESS.md M5: neither `ffmpeg -f x11grab` nor reading the raw
  `/camera/image` topic produced usable footage there). Start and stop the
  recording with:

  ```bash
  gz service -s /rover/camera/record_video --reqtype gz.msgs.VideoRecord \
    --reptype gz.msgs.Boolean --timeout 300 \
    --req 'start: true, format:"mp4", save_filename:"demo.mp4"'

  gz service -s /rover/camera/record_video --reqtype gz.msgs.VideoRecord \
    --reptype gz.msgs.Boolean --timeout 300 --req 'stop: true'
  ```

  `demo.mp4` is written to the directory `gz sim` was started from.

  ### Filming the rover (cine_camera / cine_light)

  `record_video:=true` records what the rover's *navigation* camera sees, and
  that camera is mounted at the front edge of the chassis pointing away, so
  nothing of the rover is ever in frame. Footage of a moving vehicle with no
  part of the vehicle visible reads as a still photograph - there is no
  foreground to give parallax against the terrain.

  `cine_camera:=onboard|chase|both` adds a SEPARATE 1280x720 camera whose only
  job is filming, leaving the navigation cameras' pose and intrinsics alone
  (they are load-bearing - the costmap, VO and every M3/M4 number depend on
  them, so footage requirements must never be met by moving them):

  | mode | mount | what it shows |
  |---|---|---|
  | `onboard` | mast, 0.42 m above the chassis, looking forward over the deck | first-person driving with the deck and both front wheels in the bottom of frame |
  | `chase` | boom, 1.6 m behind and 1.0 m above, tilted down | the whole rover in its terrain - what the body does, e.g. bogging down and escaping |
  | `both` | both of the above at once | one run filmed from two angles; worth the real-time-factor cost when the thing you want on film happens when it happens and cannot be re-staged |

  Each has its own recorder service - `/rover/cine/record_video` and
  `/rover/chase/record_video` - started and stopped exactly like the onboard
  one above.

  `cine_light:=true` raises the sun to 32 degrees and lifts the scene ambient
  off the floor. The shipped lighting (a 12 degree sun over a near-black
  ambient) is what a low lunar sun actually looks like and what every
  measurement in this repo was taken under, but on video it swallows the
  surface texture. It is render-only: the heightmap, the rocks, the collision
  boxes and the costmap are all generated before the light is written, so a
  clip recorded with it is driving the same world as a normal run.

  Two things to know before reading footage recorded this way:

  - **The clip is paced by wall clock, not sim time.** The recorder writes
    frames as they render, and this world runs at 0.06-0.10x real time with a
    720p camera attached, so the raw file shows the rover crawling. Speeding
    it up by 1/RTF is what makes the motion true to sim time - it is a
    correction, not an exaggeration. Measure the factor from the run rather
    than guessing it.
  - **Not every "stuck" the recovery node reports is a wedge.** Its
    `commanded_speed` includes `|ang_z| * half_track` while `gt_speed`
    measures translation only, so a rover pivoting in place to face a new leg
    reports the exact signature `commanded=0.0690, gt_speed~0` and fires an
    escape after 3 s. That asymmetry is deliberate and a symmetric fix was
    measured and rejected (PROGRESS.md - it would break detection of the real
    wedges), but it means footage of "an immobilisation" has to be chosen by
    checking `stuck_debug`: a genuine wedge has the rover commanded FORWARD
    (commanded well above 0.069) and not moving.
