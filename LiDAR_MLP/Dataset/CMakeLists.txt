cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(statistical_removal)

set (CMAKE_CXX_STANDARD 11)

find_package(PCL 1.2 REQUIRED)

include_directories(
  ${PCL_INCLUDE_DIRS}
  include/
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (detector
  src/main.cpp
  src/Classifier.cpp
  src/Projector.cpp
  src/Labeler.cpp
  src/Writer.cpp
  src/Feature_Extractors.cpp
  src/Utils.cpp
)

target_link_libraries (
  detector
  ${PCL_LIBRARIES}
)
