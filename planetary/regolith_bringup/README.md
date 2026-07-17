# regolith_bringup

Launch files, RViz config, and mission scripts for the Regolith hello-world
demo. This is the package the plan's package table designates as the
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
  click "2D Goal Pose" in RViz and the rover plans and drives there. See
  PROGRESS.md M4 for the current state (pipeline works end-to-end at
  shorter range; full-distance runs hit an unresolved terrain-collision
  stability issue).
