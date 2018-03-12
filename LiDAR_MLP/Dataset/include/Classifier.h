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

struct Cluster_Return {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud;
  std::vector<pcl::PointIndices> cluster_indices;
};


class Classifier
{
public:
  Classifier();
  Cluster_Return cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr input);


private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr _ROI_mask(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
  pcl::PointCloud<pcl::PointXYZ>::Ptr _Segment_Plane(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
  Cluster_Return _Cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr _Cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
};
