#include "Utils.h"
#include "Feature_Extractors.h"

#define BOOST_FILESYSTEM_NO_DEPRECATED

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <iostream>
#include <fstream>

//http://www.boost.org/doc/libs/1_37_0/libs/filesystem/example/simple_ls.cpp
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
          if (path.find(".pcd") != std::string::npos) {
            output.push_back(path);
          }
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

std::vector<std::vector<std::string> > find_cluster_files()
{
  const int max = 7481;
  std::string all[max];
  std::string in_temp, out_temp;
  for (int i = 0; i < max; i++)
  {
    in_temp = std::to_string(i);
    out_temp = in_temp;
    for (int j = 0; j < 6 - in_temp.size(); j++)
    {
      out_temp = "0" + out_temp;
    }
    all[i] = out_temp;
  }

  const std::string global_path = "/home/mikep/DataSets/KITTI/Clusters/";
  std::string folders[max];
  std::string temp_ped, temp_car, temp_cyc;
  std::vector<std::vector<std::string> > clusters;
  std::vector<std::string> temp_paths;
  for (int i = 0; i < max; i++)
  {
    folders[i] = global_path + all[i];
    temp_paths = glob(folders[i]);
    for (int j = 0; j < temp_paths.size(); j++)
    {
      temp_paths[j] = folders[i] + "/" + temp_paths[j];
    }
    clusters.push_back(temp_paths);
  }

  return clusters;
}

std::vector<std::vector<std::vector<std::vector<float> > > > populate_vectors(std::vector<std::vector<std::string> > cluster_paths)
{
  // Cloud / Cluster / X&Y&Z / Points
  std::vector<std::vector<std::vector<std::vector<float> > > > output;

  pcl::PCDReader reader;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<float> temp_x, temp_y, temp_z;
  std::vector<std::vector<float> > temp_cluster;
  std::vector<std::vector<std::vector<float> > > temp_cloud;
  for (int i = 0; i < cluster_paths.size(); i++)
  {
    for (int j = 0; j < cluster_paths[i].size(); j++)
    {
      reader.read(cluster_paths[i][j], *cloud);
      for (size_t k = 0; k < cloud->points.size(); k++)
      {
        temp_x.push_back(cloud->points[k].x);
        temp_y.push_back(cloud->points[k].y);
        temp_z.push_back(cloud->points[k].z);
      }
      temp_cluster.push_back(temp_x);
      temp_cluster.push_back(temp_y);
      temp_cluster.push_back(temp_z);
      temp_cloud.push_back(temp_cluster);
      temp_x.clear();
      temp_y.clear();
      temp_z.clear();
      temp_cluster.clear();
    }
    output.push_back(temp_cloud);
    temp_cloud.clear();
  }
  return output;
}

std::vector<std::vector<std::vector<float> > > extract_cluster_features(std::vector<std::vector<std::string> > cluster_paths)
{
  std::vector<std::vector<std::vector<std::vector<float> > > > all_points = populate_vectors(cluster_paths);

  std::vector<std::vector<std::vector<float> > > x_points, y_points, z_points;
  std::vector<std::vector<float> > x_cloud, y_cloud, z_cloud;
  // For every cloud
  for (int i = 0; i < all_points.size(); i++)
  {
    // For every cluster
    for (int j = 0; j < all_points[i].size(); j++)
    {
        x_cloud.push_back(all_points[i][j][0]);
        y_cloud.push_back(all_points[i][j][1]);
        z_cloud.push_back(all_points[i][j][2]);
    }
    x_points.push_back(x_cloud);
    y_points.push_back(y_cloud);
    z_points.push_back(z_cloud);
    x_cloud.clear();
    y_cloud.clear();
    z_cloud.clear();
  }
  std::cout << "Done isolating point vectors\n";

  float temp_mean;
  std::vector<std::vector<float> > x_mean, y_mean, z_mean;
  std::vector<float> cluster_x, cluster_y, cluster_z;
  for (int i = 0; i < x_points.size(); i++)
  {
    for (int j = 0; j < x_points[i].size(); j++)
    {
      temp_mean = extract_mean(x_points[i][j]);
      cluster_x.push_back(temp_mean);

      temp_mean = extract_mean(y_points[i][j]);
      cluster_y.push_back(temp_mean);

      temp_mean = extract_mean(z_points[i][j]);
      cluster_z.push_back(temp_mean);
    }
    x_mean.push_back(cluster_x);
    y_mean.push_back(cluster_y);
    z_mean.push_back(cluster_z);
    cluster_x.clear();
    cluster_y.clear();
    cluster_z.clear();
  }
  std::cout << "Done extracting means\n";

  std::vector<std::vector<float> > num_points;
  std::vector<float> temp_num;
  for (int i = 0; i < x_points.size(); i++)
  {
    for (int j = 0; j < x_points[i].size(); j++)
    {
      temp_num.push_back(x_points[i][j].size());
    }
    num_points.push_back(temp_num);
    temp_num.clear();
  }
  std::cout << "Done extracting number of points\n";

  float temp_std_dev;
  std::vector<std::vector<float> > x_std_dev, y_std_dev, z_std_dev;
  for (int i = 0; i < x_points.size(); i++)
  {
    for (int j = 0; j < x_points[i].size(); j++)
    {
      temp_std_dev = extract_std_dev(x_points[i][j], x_mean[i][j]);
      cluster_x.push_back(temp_std_dev);

      temp_std_dev = extract_std_dev(y_points[i][j], y_mean[i][j]);
      cluster_y.push_back(temp_std_dev);

      temp_std_dev = extract_std_dev(z_points[i][j], z_mean[i][j]);
      cluster_z.push_back(temp_std_dev);
    }
    x_std_dev.push_back(cluster_x);
    y_std_dev.push_back(cluster_y);
    z_std_dev.push_back(cluster_z);
    cluster_x.clear();
    cluster_y.clear();
    cluster_z.clear();
  }
  std::cout << "Done extracting standard deviations\n";

  float temp_range;
  std::vector<std::vector<float> > x_range, y_range, z_range;
  for (int i = 0; i < x_points.size(); i++)
  {
    for (int j = 0; j < x_points[i].size(); j++)
    {
      temp_range = extract_range(x_points[i][j]);
      cluster_x.push_back(temp_range);

      temp_range = extract_range(y_points[i][j]);
      cluster_y.push_back(temp_range);

      temp_range = extract_range(z_points[i][j]);
      cluster_z.push_back(temp_range);
    }
    x_range.push_back(cluster_x);
    y_range.push_back(cluster_y);
    z_range.push_back(cluster_z);
    cluster_x.clear();
    cluster_y.clear();
    cluster_z.clear();
  }
  std::cout << "Done extracting ranges\n";

  float temp_ratio;
  std::vector<std::vector<float> > x_y_ratio, y_x_ratio, x_z_ratio, z_x_ratio, y_z_ratio, z_y_ratio;
  std::vector<float> x_y_cluster, y_x_cluster, x_z_cluster, z_x_cluster, y_z_cluster, z_y_cluster;
  for (int i = 0; i < x_points.size(); i++)
  {
    for (int j = 0; j < x_points[i].size(); j++)
    {
      temp_ratio = extract_ratio(x_range[i][j], y_range[i][j]);
      x_y_cluster.push_back(temp_ratio);

      temp_ratio = extract_ratio(y_range[i][j], x_range[i][j]);
      y_x_cluster.push_back(temp_ratio);

      temp_ratio = extract_ratio(x_range[i][j], z_range[i][j]);
      x_z_cluster.push_back(temp_ratio);

      temp_ratio = extract_ratio(z_range[i][j], x_range[i][j]);
      z_x_cluster.push_back(temp_ratio);

      temp_ratio = extract_ratio(y_range[i][j], z_range[i][j]);
      y_z_cluster.push_back(temp_ratio);

      temp_ratio = extract_ratio(z_range[i][j], y_range[i][j]);
      z_y_cluster.push_back(temp_ratio);
    }
    x_y_ratio.push_back(x_y_cluster);
    y_x_ratio.push_back(y_x_cluster);
    x_z_ratio.push_back(x_z_cluster);
    z_x_ratio.push_back(z_x_cluster);
    y_z_ratio.push_back(y_z_cluster);
    z_y_ratio.push_back(z_y_cluster);
    x_y_cluster.clear();
    y_x_cluster.clear();
    x_z_cluster.clear();
    z_x_cluster.clear();
    y_z_cluster.clear();
    z_y_cluster.clear();
  }
  std::cout << "Done extracting ratios\n";



  // Combine features
  std::vector<std::vector<std::vector<float> > > output;
  std::vector<float> cluster_feat;
  std::vector<std::vector<float> > cloud_feat;
  for (int i = 0; i < x_points.size(); i++)
  {
    for (int j = 0; j < x_points[i].size(); j++)
    {
      cluster_feat.push_back(x_mean[i][j]);
      cluster_feat.push_back(y_mean[i][j]);
      cluster_feat.push_back(z_mean[i][j]);

      cluster_feat.push_back(num_points[i][j]);

      cluster_feat.push_back(x_std_dev[i][j]);
      cluster_feat.push_back(y_std_dev[i][j]);
      cluster_feat.push_back(z_std_dev[i][j]);

      cluster_feat.push_back(x_range[i][j]);
      cluster_feat.push_back(y_range[i][j]);
      cluster_feat.push_back(z_range[i][j]);

      cluster_feat.push_back(x_y_ratio[i][j]);
      cluster_feat.push_back(y_x_ratio[i][j]);

      cluster_feat.push_back(x_z_ratio[i][j]);
      cluster_feat.push_back(z_x_ratio[i][j]);

      cluster_feat.push_back(y_z_ratio[i][j]);
      cluster_feat.push_back(z_y_ratio[i][j]);

      cloud_feat.push_back(cluster_feat);
      cluster_feat.clear();
    }
    output.push_back(cloud_feat);
    cloud_feat.clear();
  }
  std::cout << "Done extracting features\n";

  return output;
}


void save_features(std::vector<std::vector<std::vector<float> > > cluster_features, std::vector<std::vector<std::string> > cluster_paths)
{
  std::ofstream myfile;
  std::string path;
  size_t f;
  for (int i = 0; i < cluster_features.size(); i++)
  {
    for (int j = 0; j < cluster_features[i].size(); j++)
    {
      path = cluster_paths[i][j];
      f = path.find(".pcd");
      path.replace(f, std::string(".pcd").length(), ".txt");
      myfile.open(path);
      for (int k = 0; k < cluster_features[i][j].size(); k++)
      {
        myfile << cluster_features[i][j][k] << std::endl;
      }
      myfile.close();
    }
  }
}





















void temp_extract(std::vector<Labeled_Cluster> clusters)
{
  std::vector<float> x_points, y_points, z_points, features;
  float x_mean, y_mean, z_mean;
  float x_std, y_std, z_std;
  float x_range, y_range, z_range;
  float x_y, y_x, x_z, z_x, y_z, z_y;
  //float num_points;

  std::ofstream myfile;
  const std::string path = "/home/mikep/DataSets/KITTI/Clusters/";

  if (!boost::filesystem::is_directory(path))
  {
    boost::filesystem::create_directory(path);
  }

  for (int i = 0; i < clusters.size(); i++)
  {
    // Get points from cluster
    for (int j = 0; j < clusters[i].cluster_cloud->points.size(); j++)
    {
      x_points.push_back(clusters[i].cluster_cloud->points[j].x);
      y_points.push_back(clusters[i].cluster_cloud->points[j].y);
      z_points.push_back(clusters[i].cluster_cloud->points[j].z);
    }

    x_mean = extract_mean(x_points);
    y_mean = extract_mean(y_points);
    z_mean = extract_mean(z_points);

    //num_points = x_points.size();

    x_std = extract_std_dev(x_points, x_mean);
    y_std = extract_std_dev(y_points, y_mean);
    z_std = extract_std_dev(z_points, z_mean);

    x_range = extract_range(x_points);
    y_range = extract_range(y_points);
    z_range = extract_range(z_points);

    x_y = extract_ratio(x_range, y_range);
    y_x = extract_ratio(y_range, x_range);
    x_z = extract_ratio(x_range, z_range);
    z_x = extract_ratio(z_range, x_range);
    y_z = extract_ratio(y_range, z_range);
    z_y = extract_ratio(z_range, y_range);


    features.push_back(x_mean);
    features.push_back(y_mean);
    features.push_back(z_mean);
    //features.push_back(num_points);
    features.push_back(x_std);
    features.push_back(y_std);
    features.push_back(z_std);
    features.push_back(x_range);
    features.push_back(y_range);
    features.push_back(z_range);
    features.push_back(x_y);
    features.push_back(y_x);
    features.push_back(x_z);
    features.push_back(z_x);
    features.push_back(y_z);
    features.push_back(z_y);



    myfile.open(path + std::to_string(i) + ".txt");
    myfile << clusters[i].label << std::endl;
    myfile << clusters[i].z_dist << std::endl;
    myfile << clusters[i].l << std::endl << std::endl << std::endl;
    for (int j = 0; j < features.size(); j++)
    {
      myfile << features[j] << std::endl;
    }
    myfile.close();

    x_points.clear();
    y_points.clear();
    z_points.clear();
    features.clear();
  }




}
