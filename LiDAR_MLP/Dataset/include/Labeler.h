#include <iostream>
#include <string>
#include <vector>

#define CAR_LABEL 0.
#define PEDESTRIAN_LABEL 1.
#define CYCLIST_LABEL 2.
#define DONTCARE_LABEL 3.

class Labeler
{
public:
  Labeler();
  std::vector<std::vector<float> > Extract_Labels(std::string fpath);

private:
  bool _is_Float(std::string someString);
};
