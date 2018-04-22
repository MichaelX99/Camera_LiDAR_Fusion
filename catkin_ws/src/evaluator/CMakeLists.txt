cmake_minimum_required(VERSION 2.8.3)
project(evaluator)

add_compile_options(-std=c++11)
set (CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  detection_fusion
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs detection_fusion
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
