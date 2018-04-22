#include <iostream>

#include "ros/ros.h"

#include "Fuser.h"

int main(int argc, char **argv)
{
  // Set up ROS
  ros::init(argc, argv, "fusion_node");

  // Construct the fusion node
  Fuser fusion_node;

  // Enter the spin
  ros::spin();

  return 0;
}
