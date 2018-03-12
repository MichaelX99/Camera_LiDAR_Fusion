#include "Labeler.h"
#include <cstdlib>
#include <fstream>
#include <boost/lexical_cast.hpp>

Labeler::Labeler() { }

bool Labeler::_is_Float(std::string someString)
{
  using boost::lexical_cast;
  using boost::bad_lexical_cast;

  try
  {
    boost::lexical_cast<float>(someString);
  }
  catch (bad_lexical_cast &)
  {
    return false;
  }

  return true;
}

std::vector<std::vector<float> > Labeler::Extract_Labels(std::string fpath)
{
  std::ifstream myfile(fpath.c_str());
  std::string line;
  std::vector<std::string> string_label;
  std::vector<std::string> in_label;
  if (myfile.is_open())
  {
    while ( getline (myfile,line, '\n'))
    {
      in_label.push_back(line);
    }
    myfile.close();
  }

  for (int i = 0; i < in_label.size(); i++)
  {
    std::istringstream iss(in_label[i].c_str());
    std::string s;
    while ( getline( iss, s, ' ' ) )
    {
      string_label.push_back(s);
    }
  }


  std::vector<std::vector<float> > temp_output, output;
  std::vector<float> temp;
  for (size_t i = 0; i < string_label.size(); i++)
  {
    if (_is_Float(string_label[i]))
    {
      temp.push_back(std::stof(string_label[i]));
    }
    else
    {

      if (temp.size() != 0)
      {
        temp_output.push_back(temp);
        temp.clear();
        if ((string_label[i].compare("Car") == 0) || (string_label[i].compare("Van") == 0) || (string_label[i].compare("Truck") == 0))
        {
          temp.push_back(CAR_LABEL);
        }
        else if (string_label[i].compare("Pedestrian") == 0)
        {
          temp.push_back(PEDESTRIAN_LABEL);
        }
        else if (string_label[i].compare("Cyclist") == 0)
        {
          temp.push_back(CYCLIST_LABEL);
        }
        else
        {
          temp.push_back(DONTCARE_LABEL);
        }
      }
      else
      {
        if ((string_label[i].compare("Car") == 0) || (string_label[i].compare("Van") == 0) || (string_label[i].compare("Truck") == 0))
        {
          temp.push_back(CAR_LABEL);
        }
        else if (string_label[i].compare("Pedestrian") == 0)
        {
          temp.push_back(PEDESTRIAN_LABEL);
        }
        else if (string_label[i].compare("Cyclist") == 0)
        {
          temp.push_back(CYCLIST_LABEL);
        }
        else
        {
          temp.push_back(DONTCARE_LABEL);
        }
      }
    }
  }
  if (temp.size() == 15)
  {
    temp_output.push_back(temp);
  }

  for (size_t i = 0; i < temp_output.size(); i++)
  {
    if (temp_output[i][0] != DONTCARE_LABEL)
    {
      output.push_back(temp_output[i]);
    }
  }

  return output;
}
