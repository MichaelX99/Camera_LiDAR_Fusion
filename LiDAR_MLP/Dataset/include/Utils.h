#include <vector>
#include <string>

#include "Writer.h"

std::vector<std::vector<std::string> > find_cluster_files();
std::vector<std::vector<std::vector<float> > > extract_cluster_features(std::vector<std::vector<std::string> > cluster_paths);
void save_features(std::vector<std::vector<std::vector<float> > > cluster_features, std::vector<std::vector<std::string> > cluster_paths);

void temp_extract(std::vector<Labeled_Cluster> clusters);
