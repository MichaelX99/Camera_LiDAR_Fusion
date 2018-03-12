#include <Projector.h>

#include "Eigen/Dense"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <pcl/common/transforms.h>

bool Projector::_is_Float(std::string someString)
{
  using boost::lexical_cast;
  using boost::bad_lexical_cast;

  try
  {
    boost::lexical_cast<float>(someString);
  }
  catch (bad_lexical_cast &)
  {
    return false;
  }

  return true;
}

std::vector<std::vector<float> > Projector::_Parse(std::string calib_fpath)
{
  std::vector<std::vector<float> > float_contents;
  std::vector<float> temp;

  std::string line;
  std::ifstream myfile(calib_fpath.c_str());
  float f_temp;
  size_t ind, t_ind;
  if (myfile.is_open())
  {
    while ( getline (myfile,line, ' '))
    {
      if (line.find("e"))
      {
        if (_is_Float(line))
        {
          //ind = line.find("e");
          f_temp = std::stof(line);
          temp.push_back(f_temp);
        }
        else
        {
          ind = line.find("e");
          t_ind = ind + 4;
          std::string sub = line.substr(0,t_ind);
          if(_is_Float(sub))
          {
            f_temp = std::stof(sub);
            temp.push_back(f_temp);
          }
          float_contents.push_back(temp);
          temp.clear();
        }
      }
    }
    myfile.close();
  }

  return float_contents;
}

Eigen::MatrixXd Projector::_Fill_P2(std::vector<float> input)
{
  assert(input.size() == 12);

  Eigen::MatrixXd P2(4,4);
  P2(0,0) = input[0];
  P2(0,1) = input[1];
  P2(0,2) = input[2];
  P2(0,3) = input[3];

  P2(1,0) = input[4];
  P2(1,1) = input[5];
  P2(1,2) = input[6];
  P2(1,3) = input[7];

  P2(2,0) = input[8];
  P2(2,1) = input[9];
  P2(2,2) = input[10];
  P2(2,3) = input[11];

  P2(3,0) = 0;
  P2(3,1) = 0;
  P2(3,2) = 0;
  P2(3,3) = 1;

  return P2;
}

Eigen::MatrixXd Projector::_Fill_R0(std::vector<float> input)
{
  assert(input.size() == 9);

  Eigen::MatrixXd R0_rect(4,4);
  R0_rect(0,0) = input[0];
  R0_rect(0,1) = input[1];
  R0_rect(0,2) = input[2];
  R0_rect(0,3) = 0;

  R0_rect(1,0) = input[3];
  R0_rect(1,1) = input[4];
  R0_rect(1,2) = input[5];
  R0_rect(1,3) = 0;

  R0_rect(2,0) = input[6];
  R0_rect(2,1) = input[7];
  R0_rect(2,2) = input[8];
  R0_rect(2,3) = 0;

  R0_rect(3,0) = 0;
  R0_rect(3,1) = 0;
  R0_rect(3,2) = 0;
  R0_rect(3,3) = 1;

  return R0_rect;
}

Eigen::MatrixXd Projector::_Fill_Tr(std::vector<float> input)
{
  assert(input.size() == 12);

  //Eigen::Matrix4d Tr_velo_to_cam;
  Eigen::MatrixXd Tr_velo_to_cam(4,4);
  Tr_velo_to_cam(0,0) = input[0];
  Tr_velo_to_cam(0,1) = input[1];
  Tr_velo_to_cam(0,2) = input[2];
  Tr_velo_to_cam(0,3) = input[3];

  Tr_velo_to_cam(1,0) = input[4];
  Tr_velo_to_cam(1,1) = input[5];
  Tr_velo_to_cam(1,2) = input[6];
  Tr_velo_to_cam(1,3) = input[7];

  Tr_velo_to_cam(2,0) = input[8];
  Tr_velo_to_cam(2,1) = input[9];
  Tr_velo_to_cam(2,2) = input[10];
  Tr_velo_to_cam(2,3) = input[11];

  Tr_velo_to_cam(3,0) = 0;
  Tr_velo_to_cam(3,1) = 0;
  Tr_velo_to_cam(3,2) = 0;
  Tr_velo_to_cam(3,3) = 1;

  return Tr_velo_to_cam;
}

Projector::Projector() { }

pcl::PointCloud<pcl::PointXYZ>::Ptr Projector::project(std::string calib_fpath, std::string pcd_fpath)
{
  std::vector<std::vector<float> > values = _Parse(calib_fpath);


  Eigen::MatrixXd P2 = _Fill_P2(values[3]);

  Eigen::MatrixXd R0_rect = _Fill_R0(values[5]);

  Eigen::MatrixXd Tr_velo_to_cam = _Fill_Tr(values[6]);

  Eigen::MatrixXd mult = P2 * R0_rect * Tr_velo_to_cam;

  pcl::PCDReader reader;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  reader.read (pcd_fpath, *cloud);

  for (size_t i = 0; i < cloud->points.size(); i++)
  {
    Eigen::VectorXd temp(4);
    temp(0) = cloud->points[i].x;
    temp(1) = cloud->points[i].y;
    temp(2) = cloud->points[i].z;
    temp(3) = 1;

    temp = Tr_velo_to_cam * temp;

    cloud->points[i].x = temp(0);
    cloud->points[i].y = temp(1);
    cloud->points[i].z = temp(2);
  }

  return cloud;
}
