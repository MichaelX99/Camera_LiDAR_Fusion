#include "Feature_Extractors.h"
#include "math.h"

float extract_mean(std::vector<float> points)
{
  float output = 0;
  for (int i = 0; i < points.size(); i++)
  {
    output = output + points[i];
  }
  output = output / points.size();

  return output;
}

float extract_std_dev(std::vector<float> points, float mean)
{
  float output = 0;

  for (int i = 0; i < points.size(); i++)
  {
    output = output + pow(points[i] - mean, 2);
  }
  output = output / points.size();
  output = sqrt(output);

  return output;
}

float extract_range(std::vector<float> points)
{
  float output;
  float min = 10000000;
  float max = -10000000;

  for (int i = 0; i < points.size(); i++)
  {
    if (min > points[i])
    {
      min = points[i];
    }
    5;
    if (max < points[i])
    {
      max = points[i];
    }
  }
  output = max - min;

  return output;
}

float extract_ratio(float p1, float p2)
{
  float output;

  output = p1 / p2;

  return output;
}
