#include <vector>
#include <string>
#include <iostream>

#include "Writer.h"
#include "Utils.h"

int main (int argc, char** argv)
{
  // Write out extracted clusters to disk
  std::vector<Labeled_Cluster> clusters = cluster_write();
  std::cout << "Done extracting clusters\n";

  //std::vector<std::vector<std::string> > cluster_paths = find_cluster_files();
  //std::cout << "Done finding files\n";

  temp_extract(clusters);

  //std::vector<std::vector<std::vector<float> > > cluster_features = extract_cluster_features(cluster_paths);

  //save_features(cluster_features, cluster_paths);

  return 0;
}
