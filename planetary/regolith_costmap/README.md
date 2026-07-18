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
