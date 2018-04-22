/*
  The following code was not written by me but taken from Martin Kersner.  The github for this code can be found @

  https://github.com/martinkersner/non-maximum-suppression-cpp
*/
#ifndef _FOUND_NMS_
#define _FOUND_NMS_

#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

enum PointInRectangle {XMIN, YMIN, XMAX, YMAX};

std::vector<cv::Rect> nms(const std::vector<std::vector<float>> &,
                          const float &);

std::vector<float> GetPointFromRect(const std::vector<std::vector<float>> &,
                                    const PointInRectangle &);

std::vector<float> ComputeArea(const std::vector<float> &,
                               const std::vector<float> &,
                               const std::vector<float> &,
                               const std::vector<float> &);

template <typename T>
std::vector<int> argsort(const std::vector<T> & v);

std::vector<float> Maximum(const float &,
                           const std::vector<float> &);

std::vector<float> Minimum(const float &,
                           const std::vector<float> &);

std::vector<float> CopyByIndexes(const std::vector<float> &,
                                 const std::vector<int> &);

std::vector<int> RemoveLast(const std::vector<int> &);

std::vector<float> Subtract(const std::vector<float> &,
                            const std::vector<float> &);

std::vector<float> Multiply(const std::vector<float> &,
                            const std::vector<float> &);

std::vector<float> Divide(const std::vector<float> &,
                          const std::vector<float> &);

std::vector<int> WhereLarger(const std::vector<float> &,
                             const float &);

std::vector<int> RemoveByIndexes(const std::vector<int> &,
                                 const std::vector<int> &);

std::vector<cv::Rect> BoxesToRectangles(const std::vector<std::vector<float>> &);

template <typename T>
std::vector<T> FilterVector(const std::vector<T> &,
                            const std::vector<int> &);

#endif // _NMS_
