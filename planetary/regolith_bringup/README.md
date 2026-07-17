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
