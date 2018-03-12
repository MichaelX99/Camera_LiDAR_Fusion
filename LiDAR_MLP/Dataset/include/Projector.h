#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

class Projector
{
public:
  Projector();
  pcl::PointCloud<pcl::PointXYZ>::Ptr project(std::string calib_fpath, std::string pcd_fpath);

private:
  bool _is_Float(std::string);
  std::vector<std::vector<float> > _Parse(std::string calib_fpath);
  Eigen::MatrixXd _Fill_P2(std::vector<float> input);
  Eigen::MatrixXd _Fill_R0(std::vector<float> input);
  Eigen::MatrixXd _Fill_Tr(std::vector<float> input);
};
