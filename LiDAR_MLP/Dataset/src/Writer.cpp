#include "Classifier.h"
#include "Projector.h"
#include "Labeler.h"
#include "Writer.h"

std::vector<Labeled_Cluster> cluster_write()
//void cluster_write()
{
  std::string calib_fpath, pcd_fpath, label_fpath, data_dir, base_path, num;

  // Form the list of all the points cloud numbers
  std::string all[7481];
  std::string in_temp, out_temp;
  //const int max = 10;
  const int max = 7480;
  for (int i = 0; i <= max; i++)
  {
    in_temp = std::to_string(i);
    out_temp = in_temp;
    for (int j = 0; j < 6 - in_temp.size(); j++)
    {
      out_temp = "0" + out_temp;
    }
    all[i] = out_temp;
  }

  // Form objects
  Projector my_projector;
  Classifier my_classifier;
  Labeler label;

  Cluster_Return returned;
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster;
  std::vector<std::vector<float> > labels;
  std::string output_fpath;
  int car_count=0, ped_count=0, cyc_count=0;
  long int total_count=0, actual_count=0;

  std::ofstream label_file;
  std::string label_path;

  base_path = "/home/mikep/DataSets/KITTI/";

  //pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");

  std::vector<Labeled_Cluster> output;
  int label_count, na_count;
  for (int i = 0; i <= max; i++)
  {
    label_count = 0;
    na_count = 0;
    calib_fpath = base_path + "Images/Train/calibration/" + all[i] + ".txt";
    pcd_fpath = base_path + "Point_Cloud/pcds/" + all[i] + ".pcd";
    label_fpath = base_path + "Images/Labels/left_img/" + all[i] + ".txt";

    // Read in the transformed lidar point cloud in camera coordinate frame
    cloud = my_projector.project(calib_fpath.c_str(), pcd_fpath.c_str());

    // Cluster the points in the cloud
    returned = my_classifier.cluster(cloud);
    cloud_cluster = returned.cluster_cloud;
    cluster_indices = returned.cluster_indices;

    /*viewer.showCloud (cloud_cluster);
    while (!viewer.wasStopped ())
    {
    }*/

    // Get the labels from the label file
    labels = label.Extract_Labels(label_fpath.c_str());

    // Extract the individual labels that I care about
    std::vector<float> label_x, label_y, label_z, label_w, label_h, label_l, label_c;
    for (size_t j = 0; j < labels.size(); j++)
    {
      label_x.push_back(labels[j][11]);
      label_y.push_back(labels[j][12]);
      label_z.push_back(labels[j][13]);

      label_h.push_back(labels[j][8]);
      label_w.push_back(labels[j][9]);
      label_l.push_back(labels[j][10]);

      label_c.push_back(labels[j][0]);
      total_count++;
    }

    float x,y,z;
    bool x_check, y_check, z_check;
    int found_label;
    float found_z, found_l;
    float max_z = -100000000;
    float min_z = 100000000;
    std::vector<int> cluster_labels;
    std::vector<float> cluster_z, cluster_l;
    // Loop over every different cluster
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
      found_label = DONTCARE_LABEL;
      found_z = 0;
      found_l = 0;
      // Loop through every label
      for (int j = 0; j < label_x.size(); j++)
      {
        float count = 0.0, pt = 0.0;
        // Loop through every point in the cluster
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
          pt = pt + 1;
          x_check = false;
          y_check = false;
          z_check = false;
          x = cloud_cluster->points[*pit].x;
          y = cloud_cluster->points[*pit].y;
          z = cloud_cluster->points[*pit].z;
          if (max_z < z)
          {
            max_z = z;
          }
          x_check = (x > label_x[j] + label_w[j]) || (x < label_x[j] - label_w[j]);
          if (min_z > z)
          {
            min_z = z;
          }
          y_check = (y > label_y[j] + label_h[j]) || (y < label_y[j] - label_h[j]);
          z_check = (z > label_z[j] + label_l[j]) || (z < label_z[j] - label_l[j]);
          if (x_check || y_check || z_check)
          {
            count = count + 1;
          }
        }
        if (count/pt < 0.05)
        {
          found_label = label_c[j];
          found_z = label_z[j];
          found_l = label_l[j];
          label_count++;
        }
        else
        {
          found_z = (max_z - min_z)/2 + min_z;
          found_l = max_z - min_z;
        }
      }
      cluster_labels.push_back(found_label);
      cluster_l.push_back(found_l);
      cluster_z.push_back(found_z);
      total_count++;
    }
    /*i = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
      if (cluster_labels[i] != DONTCARE_LABEL)
      {
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
          temp_cloud->points.push_back(cloud_cluster->points[*pit]);
        }
      }
      i++;
    }
    viewer.showCloud (temp_cloud);
    while (!viewer.wasStopped ())
    {
    }*/

    srand (time(NULL));
    int ind;
    size_t k = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
      ind = rand() % 100;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
      std::vector<pcl::PointXYZRGB> data;
      Labeled_Cluster temp;
      if (cluster_labels[k] != DONTCARE_LABEL)
      {
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
          temp_cloud->points.push_back(cloud_cluster->points[*pit]);
        }
        //temp.cluster_cloud = &temp_cloud.points;
        temp.cluster_cloud = temp_cloud;
        if (cluster_labels[k] == CAR_LABEL)
        {
          temp.label = temp.CAR;
        }
        else if (cluster_labels[k] == PEDESTRIAN_LABEL)
        {
          temp.label = temp.PEDESTRIAN;
        }
        else if (cluster_labels[k] == CYCLIST_LABEL)
        {
          temp.label = temp.CYCLIST;
        }
        temp.z_dist = cluster_z[k];
        temp.l = cluster_l[k];
        output.push_back(temp);
      }
      else if ((na_count <= label_count) && (ind < 4))
      {
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
          temp_cloud->points.push_back(cloud_cluster->points[*pit]);
        }
        temp.cluster_cloud = temp_cloud;
        temp.label = temp.NA;
        temp.z_dist = cluster_z[k];
        temp.l = cluster_l[k];
        output.push_back(temp);
        na_count++;
      }
      k++;
    }
    float temp_counter = (float)i;
    std::cout << temp_counter/max << " Percent complete\n";
  }


  float car_count1 = 0;
  float ped_count1 = 0;
  float cyc_count1 = 0;
  float na_count1 = 0;
  float total1 = 0;
  for (int i = 0; i < output.size(); i++)
  {
    if (output[i].label == output[i].NA)
    {
      na_count1++;
    }
    else if (output[i].label == output[i].CAR)
    {
      car_count1++;
    }
    else if (output[i].label == output[i].PEDESTRIAN)
    {
      ped_count1++;
    }
    else if (output[i].label == output[i].CYCLIST)
    {
      cyc_count1++;
    }
    total1++;
  }
  std::cout << "Car Percent was = " << car_count1/total1 << std::endl;
  std::cout << "Predestrian Percent was = " << ped_count1/total1 << std::endl;
  std::cout << "Cyclist Percent was = " << cyc_count1/total1 << std::endl;
  std::cout << "NA Percent was = " << na_count1/total1 << std::endl;
  std::cout << "Total number of clusters are = " << total1 << std::endl;



  return output;

  /*if (!boost::filesystem::is_directory(base_path+"Clusters/"))
  {
    boost::filesystem::create_directory(base_path+"Clusters/");
  }

  for (int i = 0; i <= max; i++)
  {
    calib_fpath = base_path + "Images/Train/calibration/" + all[i] + ".txt";
    pcd_fpath = base_path + "Point_Cloud/pcds/" + all[i] + ".pcd";
    label_fpath = base_path + "Images/Labels/left_img/" + all[i] + ".txt";
    data_dir = base_path + "Clusters/" + all[i];

    // Read in the transformed lidar point cloud in camera coordinate frame
    cloud = my_projector.project(calib_fpath.c_str(), pcd_fpath.c_str());

    // Cluster the points in the cloud
    returned = my_classifier.cluster(cloud);
    cloud_cluster = returned.cluster_cloud;
    //cloud_cluster = my_classifier.cluster(cloud);

    // Get the labels from the label file
    labels = label.Extract_Labels(label_fpath.c_str());

    // Extract the individual labels that I care about
    std::vector<float> label_x, label_y, label_z, label_w, label_h, label_l, label_c;
    for (size_t j = 0; j < labels.size(); j++)
    {
      label_x.push_back(labels[j][11]);
      label_y.push_back(labels[j][12]);
      label_z.push_back(labels[j][13]);

      label_h.push_back(labels[j][8]);
      label_w.push_back(labels[j][9]);
      label_l.push_back(labels[j][10]);

      label_c.push_back(labels[j][0]);
      total_count++;
    }

    // Loop through every label and then every point to see how many points lie in that label to push them to their own cloud
    float x, y, z;
    bool x_check, y_check, z_check;
    for (size_t j = 0; j < label_x.size(); j++)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr labeled_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

      for (size_t k = 0; k < cloud_cluster->points.size(); k++)
      {
        x_check = false;
        y_check = false;
        z_check = false;
        x = cloud_cluster->points[k].x;
        y = cloud_cluster->points[k].y;
        z = cloud_cluster->points[k].z;


        x_check = (x < label_x[j] + label_w[j]) && (x > label_x[j] - label_w[j]);
        y_check = (y < label_y[j] + label_h[j]) && (y > label_y[j] - label_h[j]);
        z_check = (z < label_z[j] + label_l[j]) && (z > label_z[j] - label_l[j]);
        if (x_check && y_check && z_check)
        {
          labeled_cloud->points.push_back(cloud_cluster->points[k]);
        }
      }
      labeled_cloud->width = labeled_cloud->points.size();
      labeled_cloud->height = 1;
      labeled_cloud->is_dense = true;

      if (labeled_cloud->points.size() != 0)
      {
        if (!boost::filesystem::is_directory(data_dir))
        {
          boost::filesystem::create_directory(data_dir);
        }
        5;

        if (label_c[j] == CAR_LABEL)
        {
          output_fpath = data_dir + "/Car" + std::to_string(car_count) + ".pcd";
          label_path = data_dir + "/Car" + std::to_string(car_count) + "_Label.txt";
          car_count++;
        }
        else if (label_c[j] == PEDESTRIAN_LABEL)
        {
          output_fpath = data_dir + "/Pedestrian" + std::to_string(ped_count) + ".pcd";
          label_path = data_dir + "/Pedestrian" + std::to_string(ped_count) + "_Label.txt";
          ped_count++;
        }
        else if (label_c[j] == CYCLIST_LABEL)
        {
          output_fpath = data_dir + "/Cyclist" + std::to_string(cyc_count) + ".pcd";
          label_path = data_dir + "/Cyclist" + std::to_string(cyc_count) + "_Label.txt";
          cyc_count++;
        }

        pcl::io::savePCDFileASCII (output_fpath.c_str(), *labeled_cloud);
        label_file.open(label_path);
        label_file << label_c[j] << std::endl;
        label_file << label_z[j] << std::endl;
        label_file.close();
        actual_count++;
      }
    }
    std::cout << "Finished #" << i << '\n';
    car_count = 0;
    ped_count = 0;
    cyc_count = 0;
  }
  std::cout << "\n\n\n" << "Wrote " << actual_count << " out of " << total_count << " examples\n";*/
}
