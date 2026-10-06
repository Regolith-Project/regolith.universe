# regolith_costmap

2.5D traversability costmap built from the known terrain heightmap (not
onboard perception - see docs/architecture.md's "Hello-World PoC Scope").
Computes slope and roughness from the heightmap, marks cells lethal above a
slope threshold or inside a rock footprint (from `manifest.json`), inflates
by the rover radius, and publishes a transient-local `nav_msgs/OccupancyGrid`
on `/costmap`.

## Usage

Launched by `regolith_bringup`'s `autonomous_demo.launch.py` and
`hello_moon.launch.py`, which pass `manifest_path` (from
`regolith_terrain_gen`'s output) as a parameter.

## Onboard hazard layer (`hazcam_node`, off by default)

`hello_moon.launch.py hazcam:=true` (needs `nav_cameras:=true`) starts
`hazcam_node`, which turns the forward depth camera into obstacle points in
`odom` on `/hazcam/obstacles` (`sensor_msgs/PointCloud2`). A cell of a
rover-local, gravity-levelled 0.1 m elevation grid is an obstacle when its top
stands 0.15 m or more above the lowest point within 0.2 m, out to 6 m: a local
step, not a height above an assumed ground plane, so attitude error barely
matters. `costmap_node` (parameter `hazcam`) marks a cell lethal, with the static
layer's rover-radius inflation, once two frames have reported it; marks are
permanent, never within `hazcam_goal_clearance_m` of the active goal, and a mark
on the path still ahead re-sends the goal (at most once per
`hazcam_replan_min_interval_s`). Geometry and thresholds are explained in
`regolith_costmap/hazcam.py`; tests render depth images analytically
(`test/test_hazcam_geometry.py`, `test/test_hazcam_costmap.py`).
