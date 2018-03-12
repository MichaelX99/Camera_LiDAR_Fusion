#include "Classifier.h"

Classifier::Classifier() { }

Cluster_Return Classifier::cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
//pcl::PointCloud<pcl::PointXYZRGB>::Ptr Classifier::cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
  // Apply a mask to only look at roughly the camera field of view portion of the cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr masked = _ROI_mask(input);

  // Segment out the ground plane
  //pcl::PointCloud<pcl::PointXYZ>::Ptr f_cloud = _Segment_Plane(input);
  pcl::PointCloud<pcl::PointXYZ>::Ptr f_cloud = _Segment_Plane(masked);

  // Cluster the points and return a color coded cloud of clusters
  Cluster_Return _display = _Cluster(f_cloud);
  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr _display = _Cluster(f_cloud);

  return _display;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Classifier::_ROI_mask(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr output (new pcl::PointCloud<pcl::PointXYZ>);
  float x, y, z, theta;
  const float theta_thresh = 3.14 / 4;
  const float theta_middle = 0;
  for (size_t i = 0; i < input->points.size(); i++)
  {
    x = input->points[i].x;
    y = input->points[i].y;
    z = input->points[i].z;
    //theta = atan2(y, x);        //IF IN VELO COORDINATES
    theta = atan2(x, z);          //IF IN CAMERA COORDINATES
    if ((theta >= theta_middle - theta_thresh) && (theta <= theta_middle + theta_thresh))
    {
      output->points.push_back(input->points[i]);
    }
  }
  output->width = output->points.size ();
  output->height = 1;
  output->is_dense = true;

  return output;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Classifier::_Segment_Plane(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr output (new pcl::PointCloud<pcl::PointXYZ>);

  // Do the plane fit
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.2);

  seg.setInputCloud (input);
  seg.segment (*inliers, *coefficients);

  if (inliers->indices.size () == 0)
  {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
  }


  // Get all points not in the plane
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud (input);
  //extract.setInputCloud (cloud);
  extract.setIndices (inliers);
  extract.setNegative (true);
  extract.filter (*output);

  return output;
}

Cluster_Return Classifier::_Cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
//pcl::PointCloud<pcl::PointXYZRGB>::Ptr Classifier::_Cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output (new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (input);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.25); // 25cm

  ec.setMinClusterSize (5);
  //ec.setMinClusterSize (50);
  ec.setMinClusterSize (50);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (input);
  ec.extract (cluster_indices);

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->points.push_back (input->points[*pit]); //*
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;
    clusters.push_back(cloud_cluster);
  }

  output->width = input->width;
  output->height = input->height;
  output->points.resize(input->width * input->height);

  srand (1);
  int r, g, b;

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    r = rand() % 256;
    g = rand() % 256;
    b = rand() % 256;
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    {
      output->points[*pit].x = input->points[*pit].x;
      output->points[*pit].y = input->points[*pit].y;
      output->points[*pit].z = input->points[*pit].z;

      output->points[*pit].r = r;
      output->points[*pit].g = g;
      output->points[*pit].b = b;
    }
  }

  Cluster_Return test_output;
  test_output.cluster_cloud = output;
  test_output.cluster_indices = cluster_indices;

  return test_output;
  //return output;
}
