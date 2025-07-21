cmake_minimum_required(VERSION 3.0.2)
project(semantic_navigator_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  cv_bridge
  std_msgs
)

catkin_package()

install(PROGRAMS
  scripts/navigator_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
) 