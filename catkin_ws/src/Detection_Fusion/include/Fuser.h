#ifndef __FUSER__
#define __FUSER__

#include <iostream>
#include <fstream>
#include <cmath>

#include "ros/ros.h"

#include "Eigen/Dense"
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>

#include "Cuda_Ops.h"
#include "Tools.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "detection_fusion/Detections_Cloud.h"
#include "detection_fusion/Detection3D.h"
#include "detection_fusion/Fusion.h"

class Fuser
{
public:
  Fuser();
  ~Fuser();
  void incoming_data_callback(const detection_fusion::Detections_Cloud::ConstPtr& msg);

private:
  // Image space distance for combining detections and cluster centroid's
  //const float _radial_thresh = 50;
  const float _radial_thresh = 75;

  double _timer = 0.0;
  double _count = 0.0;

  cv_bridge::CvImagePtr _cv_ptr;

  struct Sample
  {
    std::vector<pcl::PointXYZ> cluster;
    detection_fusion::BBox2D detection;
    std::vector<float> feature;
    classification_output output;
  };

  cublasHandle_t _cublas_handle;

  std::vector<Sample> _samples;

  ros::NodeHandle _nh;
  Eigen::MatrixXd _Tr_velo_to_cam, _P, _R;

  // Host memory
  float *_h_layer_kernel, *_h_class_kernel, *_h_length_kernel, *_h_z_kernel, *_h_rotation_kernel;
  float *_h_layer_bias, *_h_class_bias, *_h_length_bias, *_h_z_bias, *_h_rotation_bias;

  // Device (GPU) memory
  float *_d_layer_kernel, *_d_class_kernel, *_d_length_kernel, *_d_z_kernel, *_d_rotation_kernel;
  float *_d_layer_bias, *_d_class_bias, *_d_length_bias, *_d_z_bias, *_d_rotation_bias;

  // Size of MLP
  const int _input_size = 15;
  const int _layer1_size = 150;
  const int _class_size = 4;
  const int _z_size = 1;
  const int _length_size = 1;
  const int _rotation_size = 1;

  // Scale of sigmoid output to meters
  const float _scale = 50.0;
  const float _rotation_scale = 3.141592653589793238463;

  float *_h_input, *_d_input, *_d_layer1_ouput, *_d_class_output, *_d_z_output, *_d_length_output, *_d_rotation_output;

  ros::Subscriber _incoming_subscriber = _nh.subscribe("/detections_and_cloud", 1, &Fuser::incoming_data_callback, this);
  ros::Publisher _fusion_pub = _nh.advertise<detection_fusion::Fusion>("fusion_output", 1);
  ros::Publisher _cloud_pub = _nh.advertise<sensor_msgs::PointCloud2>("cloud", 1);
  ros::Publisher _image_pub = _nh.advertise<sensor_msgs::Image>("/fusion/detections", 1);

  pcl::PointCloud<pcl::PointXYZ>::Ptr _mask_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
  pcl::PointCloud<pcl::PointXYZ>::Ptr _segment_ground_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
  pcl::PointCloud<pcl::PointXYZ>::Ptr _transform_coords(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
  std::vector<std::vector<pcl::PointXYZ> > _cluster_incoming_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input);

  std::vector<Sample> _filter_clusters(std::vector<std::vector<pcl::PointXYZ> > clusters, detection_fusion::Image_Detections detections);

  std::vector<Sample> _extract_features(std::vector<Sample> input);

  std::vector<Sample> _classify(std::vector<Sample> input);

  std::vector<detection_fusion::Detection3D> _combine_detections(std::vector<Sample> input);

  std::vector<detection_fusion::Detection3D> _nms(std::vector<detection_fusion::Detection3D> input);

  void Publish_Cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
  void _publish_image(std::vector<detection_fusion::Detection3D> detections);

  void Project_Cluster(float x_mean, float y_mean, float z_mean, int &u_centroid, int &v_centroid);
  void debug_pub(std::vector<Sample> samples);



  void Load_Kernel(std::string input, const int in, const int out, float* &h_kernel, float* &d_kernel);
  void Load_Bias(std::string input, const int out, float* &h_bias, float* &d_bias);
  void Load_Layer(std::string input, const int in, const int out, float* &h_kernel, float* &d_kernel, float* &h_bias, float* &d_bias);
};


#endif
