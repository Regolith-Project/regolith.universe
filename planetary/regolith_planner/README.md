# regolith_planner

Cost-aware A* (not shortest-path - traversal cost scales with cell cost, so
the search prefers low-risk routing) over `regolith_costmap`'s
`/costmap`, from the EKF-estimated current pose (`/odometry/filtered`) to an
RViz "2D Goal Pose" click (`/goal_pose`). Publishes a lightly-smoothed
`nav_msgs/Path` on `/planned_path`.

See `astar.py` for the search/smoothing implementation (pure Python,
dependency-free beyond numpy) and `planner_node.py` for the ROS wiring.
