cmake_minimum_required(VERSION 3.8)
project(regolith_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/sensor_covariance_relay.py
  scripts/tour_mission.py
  scripts/flip_recovery_node.py
  scripts/wheel_slip_node.py
  scripts/absolute_reference_relay.py
  scripts/mission_markers_node.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
