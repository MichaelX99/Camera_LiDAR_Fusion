#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "detection_fusion/Sensor_Data.h"
#include "std_msgs/Header.h"

#include "ros/ros.h"

#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
namespace fs = boost::filesystem;
std::vector<std::string> glob(std::string input_path)
{
  std::vector<std::string> output;

  fs::path full_path( fs::initial_path<fs::path>() );

  full_path = fs::system_complete( fs::path( input_path ) );

  unsigned long file_count = 0;
  unsigned long dir_count = 0;
  unsigned long other_count = 0;
  unsigned long err_count = 0;
  std::string path;

  if ( fs::is_directory( full_path ) )
  {
    fs::directory_iterator end_iter;
    for ( fs::directory_iterator dir_itr( full_path );
          dir_itr != end_iter;
          ++dir_itr )
    {
      try
      {
        if ( fs::is_directory( dir_itr->status() ) )
        {
          ++dir_count;
        }
        else if ( fs::is_regular_file( dir_itr->status() ) )
        {
          ++file_count;
          path = dir_itr->path().filename().string();
          output.push_back(path);
        }
        else
        {
          ++other_count;
        }

      }
      catch ( const std::exception & ex )
      {
        ++err_count;
      }
    }
  }
  return output;
}







int main(int argc, char **argv)
{
  ros::init(argc, argv, "publisher");

  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<detection_fusion::Sensor_Data>("incoming_data", 1);
  int pub_rate;
  nh.getParam("rate", pub_rate);
  ros::Duration rate(pub_rate);



  std::string base_img_path = "/home/mikep/DataSets/KITTI/Images/Train/left_img/";
  std::string base_lidar_path = "/home/mikep/DataSets/KITTI/Point_Cloud/pcds/";
  std::vector<std::string> img_paths = glob(base_img_path);
  std::vector<std::string> lidar_paths;
  std::string temp_img_path, temp_lidar_path;
  size_t r_ind;
  std::string to_find = ".png";

  for (int i = 0; i < img_paths.size(); i++)
  {
    temp_img_path = base_img_path + img_paths[i];
    temp_lidar_path = img_paths[i];

    r_ind = temp_lidar_path.find(to_find);
    temp_lidar_path.replace(r_ind, to_find.length(), ".pcd");
    temp_lidar_path = base_lidar_path + temp_lidar_path;

    lidar_paths.push_back(temp_lidar_path);
    img_paths[i] = temp_img_path;
  }




  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);




  int count = 0;
  while (ros::ok() && count < img_paths.size())
  {
    detection_fusion::Sensor_Data msg;

    std_msgs::Header header;
    header.seq = count;
    header.stamp = ros::Time::now();
    msg.header = header;
    msg.image.header = header;
    msg.filename = img_paths[count];


    cv::Mat image = cv::imread(img_paths[count], CV_LOAD_IMAGE_COLOR);
    cv_bridge::CvImage img_bridge;
    sensor_msgs::Image img_msg;

    img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, image);
    img_bridge.toImageMsg(img_msg);
    msg.image = img_msg;

    pcl::io::loadPCDFile<pcl::PointXYZ> (lidar_paths[count], *cloud);
    sensor_msgs::PointCloud2 lidar_msg;
    pcl::toROSMsg(*cloud, lidar_msg);
    msg.pcloud = lidar_msg;

    pub.publish(msg);
    //std::cout << img_paths[count] << "\n" << lidar_paths[count] << "\n";

    ros::spinOnce();

    if (pub_rate != 0)
    {
      rate.sleep();
    }
    ++count;

  }

  return 0;
}
