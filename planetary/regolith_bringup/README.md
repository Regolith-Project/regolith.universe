# regolith_bringup

Launch files, EKF config, and mission scripts for the Regolith hello-world
demo (the RViz config lives in `regolith_rover_description`). This is the package the plan's package table designates as the
integration point — its launch files reference `regolith_terrain_gen`,
`regolith_rover_description`, `regolith_planner`, and `regolith_costmap` by
name, which is why it lives here in `regolith.universe/planetary/` alongside
those packages rather than in the `regolith` meta-repo.

## Launch files

- `terrain_only.launch.py` — generates a procedural lunar terrain world for
  the given `seed` (default `42`) via `regolith_terrain_gen` and opens it in
  Gazebo. `ros2 launch regolith_bringup terrain_only.launch.py seed:=42`
- `teleop_demo.launch.py` — generates terrain, spawns the
  `regolith_rover_description` rover at the actual local terrain elevation
  (never a hard-coded height - see PROGRESS.md M2 for why that matters),
  and bridges `cmd_vel`/`odom`/`imu`/`camera`/`camera_info`/`joint_states`/`tf`
  between ROS and Gazebo. `ros2 launch regolith_bringup teleop_demo.launch.py
  seed:=42`, then in another terminal:
  `ros2 run teleop_twist_keyboard teleop_twist_keyboard`
- `localization_demo.launch.py` — everything `teleop_demo` does, plus fuses
  wheel odometry + IMU into an estimated pose (`robot_localization`'s
  `ekf_node`) and bridges Gazebo's ground truth separately for comparison
  (`/ground_truth/pose`, never fed into the estimator). See PROGRESS.md M3.
- `autonomous_demo.launch.py` — everything `localization_demo` does, plus
  `regolith_costmap` + `regolith_planner` + `regolith_vehicle_interface`:
  click "2D Goal Pose" in RViz and the rover plans and drives there (this
  launch file doesn't start RViz itself - open it manually, or use
  `hello_moon.launch.py` below, which does). See
  PROGRESS.md M4 for the current state (pipeline works end-to-end at
  shorter range; full-distance runs hit an unresolved terrain-collision
  stability issue).
- `hello_moon.launch.py` — the main entry point, superseding
  `autonomous_demo.launch.py` above (which it's built on and keeps
  identical behavior to). Also opens RViz with the rover config (disable
  with `rviz:=false`) and adds a `mission` argument:
  `ros2 launch regolith_bringup hello_moon.launch.py seed:=42` behaves
  like `autonomous_demo.launch.py` (click a goal yourself in RViz);
  `... mission:=tour` additionally runs `tour_mission.py`, a scripted
  5-waypoint loop, with no interaction needed. This is what
  `scripts/demo.sh` in the meta-repo launches. See PROGRESS.md M5 for the
  current state, including a confirmed instance of the M4 flip issue
  occurring during an unattended tour run.

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
