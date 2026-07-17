# regolith_rover_description

URDF/Xacro for the Regolith hello-world rover: a Leo-Rover-sized 4-wheel
skid-steer chassis (0.40 x 0.34 x 0.11 m, 0.46 m track), IMU, and a
forward-tilted RGB camera. Simulated in Gazebo via the `gz-sim-diff-drive-system`
plugin in its multi-joint skid-steer configuration (two left wheel joints,
two right), plus `JointStatePublisher` and `Imu` system plugins.

Stability tuning (see PROGRESS.md M2): wide track and a low chassis for a
low center of gravity, μ=1.4 wheel friction, and conservative velocity limits
(0.4 m/s linear, 0.3 rad/s angular) — a narrower/faster earlier revision
flipped when turning and driving forward at once over rough terrain.

## Files

- `urdf/regolith_rover.urdf.xacro` — the rover description; processed by
  `regolith_bringup`'s launch files via the `xacro` Python API (in-process,
  not shelled out).
- `rviz/rover.rviz` — RViz config with RobotModel, TF, Camera, and Odometry
  displays.

## Gazebo topics

The rover's Gazebo-side topics are unscoped (not `/model/<name>/...`):
`cmd_vel`, `odometry`, `pose` (bridged to `/tf`), `imu`, `camera`,
`camera_info`, and `/world/<world>/model/<name>/joint_state` (bridged to
`/joint_states`). See `regolith_bringup`'s `teleop_demo.launch.py` for the
full bridge configuration.
