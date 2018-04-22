#include "Fuser.h"

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include <boost/foreach.hpp>

#include "Found_NMS.h"

#include <time.h>

void Fuser::Load_Kernel(std::string input, const int in, const int out, float* &h_kernel, float* &d_kernel)
{
  std::string input_path;
  _nh.getParam(input, input_path);
  std::vector<std::vector<float> > vector_kernel = Populate_Weights(input_path);
  h_kernel = Columate_Matrix(vector_kernel);
  d_kernel = cublas_Set_Matrix(h_kernel, in, out);
}

void Fuser::Load_Bias(std::string input, const int out, float* &h_bias, float* &d_bias)
{
  std::string input_path;
  _nh.getParam(input, input_path);
  std::vector<float> vector_bias = Populate_Biases(input_path);
  h_bias = Vectorize(vector_bias);
  d_bias = cublas_Set_Vector(h_bias, out);
}

void Fuser::Load_Layer(std::string input, const int in, const int out, float* &h_kernel, float* &d_kernel, float* &h_bias, float* &d_bias)
{
  std::string bias_input, kernel_input;

  bias_input = input + "_bias";
  kernel_input = input + "_weight";

  Load_Kernel(kernel_input, in, out, h_kernel, d_kernel);
  Load_Bias(bias_input, out, h_bias, d_bias);
}

Fuser::Fuser()
{
/*
  Input(s): N/A
  Output(s): N/A
  Function: Construct the Fuser object
*/
  std::string input_path;

  // Populate the transform matrices to be used
  _nh.getParam("transforms", input_path);
  _Tr_velo_to_cam = Populate_Transform(input_path, "Tr_velo_to_cam: ");
  _P = Populate_Transform(input_path, "P2: ");
  _R = Populate_Transform(input_path, "R0_rect: ");

  // Populate the class layer in GPU memory
  Load_Layer("class", _layer1_size, _class_size, _h_class_kernel, _d_class_kernel, _h_class_bias, _d_class_bias);

  // Populate the length layer in GPU memory
  Load_Layer("length", _layer1_size, _length_size, _h_length_kernel, _d_length_kernel, _h_length_bias, _d_length_bias);

  // Populate the z layer in GPU memory
  Load_Layer("z", _layer1_size, _z_size, _h_z_kernel, _d_z_kernel, _h_z_bias, _d_z_bias);

  // Populate the hidden layer
  Load_Layer("layer", _input_size, _layer1_size, _h_layer_kernel, _d_layer_kernel, _h_layer_bias, _d_layer_bias);

  // Populate the rotation layer
  Load_Layer("rotation", _layer1_size, _rotation_size, _h_rotation_kernel, _d_rotation_kernel, _h_rotation_bias, _d_rotation_bias);

  // Create the CUBLAS handle used in the CUBLAS API
  cublasCreate(&_cublas_handle);

  // Preallocate space for the outputs of the MLP
  _d_input = Allocate_MLP(_input_size);
  _d_layer1_ouput = Allocate_MLP(_layer1_size);
  _d_class_output = Allocate_MLP(_class_size);
  _d_z_output = Allocate_MLP(_z_size);
  _d_length_output = Allocate_MLP(_length_size);
  _d_rotation_output = Allocate_MLP(_rotation_size);

  _h_input = (float*)malloc(_input_size * sizeof(float));
}

Fuser::~Fuser()
{
/*
  Input(s): N/A
  Output(s): N/A
  Function: Destroy the Fuser object
*/
  // Destroy the CUBLAS handle
  cublasDestroy(_cublas_handle);

  // Destroy GPU memory
  destroy_memory(_d_class_kernel);
  destroy_memory(_d_length_kernel);
  destroy_memory(_d_z_kernel);
  destroy_memory(_d_layer_kernel);
  destroy_memory(_d_class_bias);
  destroy_memory(_d_length_bias);
  destroy_memory(_d_z_bias);
  destroy_memory(_d_layer_bias);
  destroy_memory(_d_input);
  destroy_memory(_d_layer1_ouput);
  destroy_memory(_d_class_output);
  destroy_memory(_d_z_output);
  destroy_memory(_d_length_output);
  destroy_memory(_d_rotation_output);

  free(_h_input);
}

void Fuser::Publish_Cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
/*
  Input(s): Point Cloud colored according to clusters
  Output(s): N/A
  Function: Publish a point cloud in ROS
*/
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(*cloud, output);

  output.header.frame_id = "map";

  _cloud_pub.publish(output);
}

void Fuser::debug_pub(std::vector<Fuser::Sample> samples)
{
/*
  Input(s): Vector of potential samples found in the world
  Output(s): N/A
  Function: Combine all clusters from each sample in a single Point Cloud and publish it using ROS
*/
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  int r, g, b;
  int N = 0;
  for (int i = 0; i < samples.size(); i++)
  {
    r = rand() % 255;
    g = rand() % 255;
    b = rand() % 255;

    for (int j = 0; j < samples[i].cluster.size(); j++)
    {
      pcl::PointXYZRGB point;
      point.x = samples[i].cluster[j].x;
      point.y = samples[i].cluster[j].y;
      point.z = samples[i].cluster[j].z;
      point.r = r;
      point.g = g;
      point.b = b;
      cloud->points.push_back(point);
      N++;
    }
  }
  cloud->height = 1;
  cloud->width = N;

  Publish_Cloud(cloud);
}

void Fuser::_publish_image(std::vector<detection_fusion::Detection3D> detections)
{
/*
  Input(s): Vector of detections from the output of the fusion algorithm
  Output(s): N/A
  Function: Publish an annotated image of the final fused detections from the algorithm
*/
  cv::Point p1, p2;
  cv::Scalar Blue(255, 0, 0);
  cv::Scalar Green(0, 255, 0);
  cv::Scalar Red(0, 0, 255);

  int x1, x2, y1, y2;

  for (int i = 0; i < detections.size(); i++)
  {
    x1 = detections[i].bbox.x1;
    x2 = detections[i].bbox.x2;
    y1 = detections[i].bbox.y1;
    y2 = detections[i].bbox.y2;

    p1.x = x1;
    p1.y = y1;

    p2.x = x2;
    p2.y = y2;

    if (detections[i].bbox.obj_class == 1) { cv::rectangle(_cv_ptr->image, p1, p2, Blue, 3); } //Car
    else if (detections[i].bbox.obj_class == 2) { cv::rectangle(_cv_ptr->image, p1, p2, Green, 3); } //Pedestrian
    else if (detections[i].bbox.obj_class == 3) { cv::rectangle(_cv_ptr->image, p1, p2, Red, 3); } //Cyclist
  }

  cv_bridge::CvImage img_bridge;
  img_bridge = cv_bridge::CvImage(_cv_ptr->header, _cv_ptr->encoding, _cv_ptr->image);

  sensor_msgs::Image ros_image;
  img_bridge.toImageMsg(ros_image);

  _image_pub.publish(ros_image);
}

void Fuser::incoming_data_callback(const detection_fusion::Detections_Cloud::ConstPtr& msg)
{
/*
  Input(s): Message recieved from the image only CNN ROS node
  Output(s): N/A
  Function: Perform the point cloud processing on the data collected from the world
*/
  clock_t tStart = clock();

  // To get the detections from the msg data
  detection_fusion::Image_Detections detections = msg->detections;

  std::string filename = msg->filename;

  // Get the image from the msg
  try
  {
    _cv_ptr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (detections.detections.size() != 0)
  {
    // To populate a point cloud with the msg data
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(msg->pcloud,pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2,*cloud);

    double t1 = (double)(clock() - tStart)/CLOCKS_PER_SEC;

    // Only reason about points that are roughly in the same space as the image
    cloud = _mask_cloud(cloud);
    double t2 = ((double)(clock() - tStart)/CLOCKS_PER_SEC);// - t1;

    // Segment out the ground plane
    cloud = _segment_ground_plane(cloud);
    double t3 = ((double)(clock() - tStart)/CLOCKS_PER_SEC);// - t2;

    // Transform the point cloud from LiDAR coordinates to camera coordinates
    cloud = _transform_coords(cloud);
    double t4 = ((double)(clock() - tStart)/CLOCKS_PER_SEC);// - t3;

    // Seperate the cloud into clusters
    std::vector<std::vector<pcl::PointXYZ> > clusters = _cluster_incoming_cloud(cloud);
    double t5 = ((double)(clock() - tStart)/CLOCKS_PER_SEC);//- t4;

    // Combine clusters and image based detections
    std::vector<Sample> samples = _filter_clusters(clusters, detections);
    double t6 = ((double)(clock() - tStart)/CLOCKS_PER_SEC);// - t5;

    // If any clusters and image detections matched up
    if (samples.size() != 0)
    {
      // Perform feature extraction on the clusters
      samples = _extract_features(samples);
      double t7 = ((double)(clock() - tStart)/CLOCKS_PER_SEC);// - t6;

      // Run the features through the MLP
      samples = _classify(samples);
      double t8 = ((double)(clock() - tStart)/CLOCKS_PER_SEC);// - t7;

      // Adjust the confidence of the image detections based upon the output of the LiDAR MLP
      std::vector<detection_fusion::Detection3D> combined_detections = _combine_detections(samples);
      double t9 = ((double)(clock() - tStart)/CLOCKS_PER_SEC);// - t8;

      // Perform  2D NMS on the combined detections
      std::vector<detection_fusion::Detection3D> fusion_output = _nms(combined_detections);
      double t10 = ((double)(clock() - tStart)/CLOCKS_PER_SEC);// - t9;
      double t11 = ((double)(clock() - tStart)/CLOCKS_PER_SEC);

      // Publish the clustered cloud
      debug_pub(samples);

      // Publish the final annotated image
      _publish_image(fusion_output);

      // Publish the final output msg of the fusion algorithm of 3D detections
      if (fusion_output.size() != 0)
      {
        detection_fusion::Fusion output_msg;
        output_msg.detections = fusion_output;
        output_msg.filename = filename;
        output_msg.image_detections = detections;

        _fusion_pub.publish(output_msg);
      }
      //printf("t1=%.2f, t2=%.2f, t3=%.2f, t4=%.2f, t5=%.2f, t6=%.2f, t7=%.2f, t8=%.2f, t9=%.2f, t10=%.2f, t11=%.2f\n", t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11);
      //printf("t1=%.4f, t2=%.4f, t3=%.4f, t4=%.4f, t5=%.4f, t6=%.4f, t7=%.4f, t8=%.4f, t9=%.4f, t10=%.4f, t11=%.3f\n\n\n", t1, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7, t9-t8, t10-t9, t11);
    }
  }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Fuser::_mask_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
/*
  Input(s): Point cloud
  Output(s): Point cloud
  Function: Apply a mask to the point cloud that removes points roughly not in the cameras view
*/
  pcl::PointCloud<pcl::PointXYZ>::Ptr output (new pcl::PointCloud<pcl::PointXYZ>);
  float x, y, z, theta;
  const float theta_thresh = 3.14 / 4;
  const float theta_middle = 0;
  for (size_t i = 0; i < input->points.size(); i++)
  {
    x = input->points[i].x;
    y = input->points[i].y;
    z = input->points[i].z;
    theta = atan2(y, x);        //IF IN VELO COORDINATES
    //theta = atan2(x, z);          //IF IN CAMERA COORDINATES
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

pcl::PointCloud<pcl::PointXYZ>::Ptr Fuser::_segment_ground_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
/*
  Input(s): Point cloud
  Output(s): Point cloud
  Function: Segment out the ground plane based on a planar RANSAC model

  http://pointclouds.org/documentation/tutorials/planar_segmentation.php
*/
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
  extract.setIndices (inliers);
  extract.setNegative (true);
  extract.filter (*output);

  return output;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Fuser::_transform_coords(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
/*
  Input(s): Point cloud
  Output(s): Point cloud
  Function: Transform the point cloud from LiDAR coordinate frame to camera coordinate frame
*/
  Eigen::VectorXd temp(4);

  for (size_t i = 0; i < input->points.size(); i++)
  {
    temp(0) = input->points[i].x;
    temp(1) = input->points[i].y;
    temp(2) = input->points[i].z;
    temp(3) = 1;

    temp = _Tr_velo_to_cam * temp;

    input->points[i].x = temp(0);
    input->points[i].y = temp(1);
    input->points[i].z = temp(2);
  }

  return input;
}

std::vector<std::vector<pcl::PointXYZ> > Fuser::_cluster_incoming_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input)
{
/*
  Input(s): Point cloud
  Output(s): Vector of clusters which are vectors of points
  Function: Seperate the cloud into clusters based upon Euclidean distance

  http://www.pointclouds.org/documentation/tutorials/cluster_extraction.php
*/
  std::vector<std::vector<pcl::PointXYZ> > output;

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (input);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.25); // 25cm
  ec.setMinClusterSize (5);
  //ec.setMinClusterSize (25);
  //ec.setMaxClusterSize (2500);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (input);
  ec.extract (cluster_indices);

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    std::vector<pcl::PointXYZ> temp_cluster;
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    {
      temp_cluster.push_back(input->points[*pit]);
    }
    output.push_back(temp_cluster);
  }

  return output;
}

void Fuser::Project_Cluster(float x_mean, float y_mean, float z_mean, int &u_centroid, int &v_centroid)
{
/*
  Input(s): The x, y, and z, values of the cluster centroid, the pixel space locations for the centroid
  Output(s): N/A
  Function: Find where the cluster would appear in image space
*/
  Eigen::VectorXd input(4);
  Eigen::VectorXd output(3);

  // This is for the shift between camera 0 and camera 2
  input(0) = x_mean - .06;
  input(1) = y_mean + .0003;
  input(2) = z_mean - .0025;
  input(3) = 1.;

  output = _P * input;
  //output = _P * _R * input;
  u_centroid = (int)(output(0) / output(2));
  v_centroid = (int)(output(1) / output(2));
}

std::vector<Fuser::Sample> Fuser::_filter_clusters(std::vector<std::vector<pcl::PointXYZ> > clusters, detection_fusion::Image_Detections detections)
{
/*
  Input(s): Point clusters, image detections
  Output(s): Combined detections of clusters that are close to the image detections
  Function: To match up possible image detections with point clusters based upon image space distance
*/
  std::vector<Sample> output;
  float x_mean, y_mean, z_mean;
  int u_center, v_center;
  float x_center, y_center;
  float dist;

  int u_centroid, v_centroid;

  // Compute the centroid of the cluster
  for (int i = 0; i < clusters.size(); i++)
  {
    x_mean = y_mean = z_mean = 0;

    for (int j = 0; j < clusters[i].size(); j++)
    {
      x_mean += clusters[i][j].x;
      y_mean += clusters[i][j].y;
      z_mean += clusters[i][j].z;
    }
    x_mean /= clusters[i].size();
    y_mean /= clusters[i].size();
    z_mean /= clusters[i].size();

    // Find the pixel location of the cluster's centroid
    Project_Cluster(x_mean, y_mean, z_mean, u_centroid, v_centroid);

    for (int j = 0; j < detections.detections.size(); j++)
    {
      u_center = (detections.detections[j].x2 + detections.detections[j].x1) / 2;
      v_center = (detections.detections[j].y2 + detections.detections[j].y1) / 2;

      // Determine the distance of the cluster centroid the the bounding box centroid
      dist = sqrt(pow(u_center - u_centroid, 2) + pow(v_center - v_centroid, 2));
      if (dist < _radial_thresh)
      {
        Sample iter;
        iter.cluster = clusters[i];
        iter.detection = detections.detections[j];
        output.push_back(iter);
      }
    }
  }

  return output;
}

std::vector<Fuser::Sample> Fuser::_extract_features(std::vector<Fuser::Sample> input)
{
/*
  Input(s): Combined clusters and detections
  Output(s): Combined clusters and detections
  Function: Extract features from each of the clusters
*/
  float x_mean, y_mean, z_mean;
  float x_sqr, y_sqr, z_sqr;
  float x_std, y_std, z_std;
  float x_range, y_range, z_range;
  float x_y, y_x, x_z, z_x, y_z, z_y;

  float x_min, x_max, y_min, y_max, z_min, z_max;

  int N;

  std::vector<float> cluster_features;
  for (int i = 0; i < input.size(); i++)
  {
    N = input[i].cluster.size();
    x_mean = y_mean = z_mean = 0;
    x_sqr = y_sqr = z_sqr = 0;
    x_std = y_std = z_std = 0;
    x_range = y_range = z_range = 0;
    x_y = y_x = x_z = z_x = y_z = z_y = 0;
    x_min = y_min = z_min = 1000000;
    x_max = y_max = z_max = -1000000;

    for (int j = 0; j < N; j++)
    {
      // X Points
      x_mean += input[i].cluster[j].x;
      x_sqr += pow(input[i].cluster[j].x, 2);
      if (input[i].cluster[j].x < x_min)
      {
        x_min = input[i].cluster[j].x;
      }
      else if (input[i].cluster[j].x > x_max)
      {
        x_max = input[i].cluster[j].x;
      }

      // Y Points
      y_mean += input[i].cluster[j].y;
      y_sqr += pow(input[i].cluster[j].y, 2);
      if (input[i].cluster[j].y < y_min)
      {
        y_min = input[i].cluster[j].y;
      }
      else if (input[i].cluster[j].y > y_max)
      {
        y_max = input[i].cluster[j].y;
      }

      // Z Points
      z_mean += input[i].cluster[j].z;
      z_sqr += pow(input[i].cluster[j].z, 2);
      if (input[i].cluster[j].z < z_min)
      {
        z_min = input[i].cluster[j].z;
      }
      else if (input[i].cluster[j].z > z_max)
      {
        z_max = input[i].cluster[j].z;
      }
    }
    x_mean /= N;
    y_mean /= N;
    z_mean /= N;

    x_sqr /= N;
    y_sqr /= N;
    z_sqr /= N;

    x_std = x_sqr - pow(x_mean, 2);
    y_std = y_sqr - pow(y_mean, 2);
    z_std = z_sqr - pow(z_mean, 2);

    x_range = x_max - x_min;
    y_range = y_max - y_min;
    z_range = z_max - z_min;

    x_y = x_range / y_range;
    y_x = y_range / x_range;
    x_z = x_range / z_range;
    z_x = z_range / x_range;
    y_z = y_range / z_range;
    z_y = z_range / y_range;

    cluster_features.push_back(x_mean);
    cluster_features.push_back(y_mean);
    cluster_features.push_back(z_mean);

    cluster_features.push_back(x_std);
    cluster_features.push_back(y_std);
    cluster_features.push_back(z_std);

    cluster_features.push_back(x_range);
    cluster_features.push_back(y_range);
    cluster_features.push_back(z_range);

    cluster_features.push_back(x_y);
    cluster_features.push_back(y_x);
    cluster_features.push_back(x_z);
    cluster_features.push_back(z_x);
    cluster_features.push_back(y_z);
    cluster_features.push_back(z_y);

    input[i].feature = cluster_features;
    cluster_features.clear();
  }
  return input;
}

std::vector<Fuser::Sample> Fuser::_classify(std::vector<Fuser::Sample> input)
{
/*
  Input(s): Combined clusters and detections
  Output(s): Combined clusters and detections
  Function: Push each cluster feature vector through the MLP and populate the output
*/
  std::vector<float> temp;
  for (int i = 0; i < _input_size; i++)
  {
    temp.push_back((float)i);
  }

  for (int i = 0; i < input.size(); i++)
  {
    cublas_Set_Input(_h_input, input[i].feature, _d_input, _input_size);

    blas_layer(_d_input, _d_layer_kernel, _d_layer_bias, _d_layer1_ouput, _layer1_size, _input_size, _cublas_handle);
    RELU(_d_layer1_ouput, _layer1_size);

    blas_layer(_d_layer1_ouput, _d_class_kernel, _d_class_bias, _d_class_output, _class_size, _layer1_size, _cublas_handle);
    Softmax(_d_class_output, _class_size);

    blas_layer(_d_layer1_ouput, _d_length_kernel, _d_length_bias, _d_length_output, _length_size, _layer1_size, _cublas_handle);
    Sigmoid(_d_length_output, _length_size);

    blas_layer(_d_layer1_ouput, _d_z_kernel, _d_z_bias, _d_z_output, _z_size, _layer1_size, _cublas_handle);
    Sigmoid(_d_z_output, _z_size);

    blas_layer(_d_layer1_ouput, _d_rotation_kernel, _d_rotation_bias, _d_rotation_output, _rotation_size, _layer1_size, _cublas_handle);

    classification_output iter_output = populate_output(_d_class_output, _d_length_output, _d_z_output, _d_rotation_output);
    input[i].output = iter_output;
  }

  return input;
}

std::vector<detection_fusion::Detection3D> Fuser::_combine_detections(std::vector<Fuser::Sample> input)
{
/*
  Input(s): Combined clusters and detections
  Output(s): 3D detections
  Function: Filter out detections and clusters that do not agree and adjust confidence of those that do
*/
  std::vector<detection_fusion::Detection3D> output;

  detection_fusion::BBox2D camera_detection;
  classification_output lidar_detection;
  std::vector<float> feature;

  int camera_class, lidar_class;
  float camera_prob, lidar_prob;

  detection_fusion::Detection3D iter;

  for (int i = 0; i < input.size(); i++)
  {
    camera_detection = input[i].detection;
    lidar_detection = input[i].output;
    feature = input[i].feature;

    camera_class = camera_detection.obj_class;
    lidar_class = lidar_detection.obj_class;

    camera_prob = camera_detection.probability;
    lidar_prob = lidar_detection.probability;

    if (camera_class == lidar_class)
    {
      iter.bbox = camera_detection;
      iter.x = feature[0];
      iter.y = feature[1];
      iter.z = lidar_detection.z * _scale;
      iter.w = feature[6];
      iter.h = feature[7];
      iter.l = lidar_detection.l * _scale;
      iter.a = lidar_detection.a * _rotation_scale;


      iter.bbox.probability += .5;
      if (iter.bbox.probability > 1.) iter.bbox.probability = 1.0;

      output.push_back(iter);
    }
    else
    { continue; }
  }

  return output;
}

std::vector<detection_fusion::Detection3D> Fuser::_nms(std::vector<detection_fusion::Detection3D> temp_input)
{
/*
  Input(s): 3D detections
  Output(s): Final 3D detections
  Function: Convert box forms to run the NMS algorithm on the detections
*/
  std::vector<detection_fusion::Detection3D> output;
  for (int i = 0; i < temp_input.size(); i ++)
  {
    if (temp_input[i].bbox.probability > .6)
    {
      output.push_back(temp_input[i]);
    }
  }

  std::vector<std::vector<float> > temp_boxes;
  std::vector<float> temp;
  float x1, x2, y1, y2;
  for (int i = 0; i < output.size(); i++)
  {
    x1 = (float)output[i].bbox.x1;
    x2 = (float)output[i].bbox.x2;
    y1 = (float)output[i].bbox.y1;
    y2 = (float)output[i].bbox.y2;

    temp.push_back(x1);
    temp.push_back(y1);
    temp.push_back(x2);
    temp.push_back(y2);
    temp_boxes.push_back(temp);
    temp.clear();
  }
  std::vector<cv::Rect> reducedRectangle = nms(temp_boxes, .3);

  int nms_x1, nms_x2, nms_y1, nms_y2, box_x1, box_x2, box_y1, box_y2;
  std::vector<detection_fusion::Detection3D> real_output;
  for (int i = 0; i < reducedRectangle.size(); i++)
  {
    for (int j = 0; j < output.size(); j++)
    {
      nms_x1 = (int)reducedRectangle[i].x;
      nms_x2 = (int)reducedRectangle[i].width + nms_x1;
      nms_y1 = (int)reducedRectangle[i].y;
      nms_y2 = (int)reducedRectangle[i].height + nms_y1;
      box_x1 = output[j].bbox.x1;
      box_x2 = output[j].bbox.x2;
      box_y1 = output[j].bbox.y1;
      box_y2 = output[j].bbox.y2;
      if (box_x1 == nms_x1 && box_y1 == nms_y1 && box_x2 == nms_x2 && box_y2 == nms_y2)
      {
        real_output.push_back(output[j]);
      }
    }
  }

  return real_output;
}
