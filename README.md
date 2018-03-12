# Camera_LiDAR_Fusion
The implementation for my Masters Thesis

# Things to Do
1- Install Tensorflow Object Detection API @ https://github.com/tensorflow/models/tree/master/research/object_detection

2- Download KITTI Dataset and place within KITTI_Dataset @ http://www.cvlibs.net/datasets/kitti/

3- Convert KITTI left images into a TFRecord File using Image_Detection_CNN/TFRecord_Conversion/record.py

4- Download the SSD MobileNets COCO trained network to Image_Detection_CNN/Transfer_Learning @ https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

5- Fine tune the network using the Object Detection API's config file stored in Transfer_Learning

6- Optionally evaluate the fine tuned networks in Image_Detection_CNN/Evaluation

7- Convert KITTI .bin point clouds to .pcd point clouds using https://github.com/jaejunlee0538/kitti2pcl

8- Form the LiDAR training and evaluation dataset using LiDAR_MLP/Dataset

9- Traing the LiDAR MLP using LiDAR_MLP/Classification/train.py

10- Build the ROS system in catkin_ws

11- Source catkin_ws/devel/setup.bash

12- Test out the fusion algorithm, roslaunch publisher publisher.launch

13- Visualize using RVIZ
