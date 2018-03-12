#ifndef WRITER
#define WRITER

#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>

#include <vector>
#include <algorithm>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <iostream>
#include <fstream>

struct Labeled_Cluster {
  //std::vector<pcl::PointsXYZRGB> cluster_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud;
  const int NA = 0;
  const int CAR = 1;
  const int PEDESTRIAN = 2;
  const int CYCLIST = 3;
  int label;
  float z_dist;
  float l;
};

//void cluster_write();
std::vector<Labeled_Cluster> cluster_write();

#endif
