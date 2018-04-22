cmake_minimum_required(VERSION 2.8.3)
project(detection_fusion)

add_compile_options(-std=c++11)
set (CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  pcl_conversions
  pcl_ros
  message_generation
)

add_message_files(
  FILES
  Sensor_Data.msg
  BBox2D.msg
  Image_Detections.msg
  Detections_Cloud.msg
  Detection3D.msg
  Fusion.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS rospy roscpp std_msgs sensor_msgs cv_bridge image_transport pcl_conversions pcl_ros message_generation
)

find_package(CUDA 9.0 REQUIRED)
find_package(OpenCV)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

cuda_add_executable(${PROJECT_NAME}
  src/Tools.cpp
  src/Cuda_Ops.cu
  src/Fuser.cpp
  src/Found_NMS.cpp
  src/lidar_detector_node.cpp
)

cuda_add_cublas_to_target(${PROJECT_NAME})

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
