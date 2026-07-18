# regolith_vehicle_interface

Translates the planner's `nav_msgs/Path` into skid-steer `cmd_vel`: a minimal
pure pursuit follower (see `docs/architecture.md`'s reuse log for why
`autoware_pure_pursuit` wasn't reused - it's built for Ackermann steering).

Modest speed profile: slows for high-cost costmap cells, and stops
translating entirely to rotate in place first whenever the heading error to
the lookahead point exceeds 30° (driving forward while turning sharply on
rough terrain was found to destabilize the rover - see PROGRESS.md M4).
Minimal recovery per the plan ("do not build elaborate FDIR"): if the rover
strays too far from the path or stalls, it stops and re-triggers planning
from wherever it currently is. If the raw IMU shows the rover has actually
flipped (roll or pitch beyond `flipped_attitude_deg`, default 60° - a known
terrain-collision failure mode, see PROGRESS.md M4), it halts and logs an
error instead of silently replanning forever; it resumes automatically if
the attitude ever returns to normal.
