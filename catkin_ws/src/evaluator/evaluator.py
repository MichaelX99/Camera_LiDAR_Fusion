#!/usr/bin/env python
import rospy
import numpy as np
from detection_fusion.msg import *
import os
import shutil

class Evaluator(object):
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('evaluator')

        # Subscriber to the synchronized incoming data
        self.subscriber = rospy.Subscriber('/fusion_output', Fusion, self.callback)

        home_dir = rospy.get_param("home_dir")
        home_dir = home_dir + "/for_eval"
        if not os.path.exists(home_dir):
            os.mkdir(home_dir)
        self.fusion_dir = home_dir + "/fusion/"
        if not os.path.exists(self.fusion_dir):
            os.mkdir(self.fusion_dir)
        self.cnn_dir = home_dir + "/cnn/"
        if not os.path.exists(self.cnn_dir):
            os.mkdir(self.cnn_dir)

        self.label_path = "/home/mikep/DataSets/KITTI/Images/Labels/left_img/"

        self.count = 0

        rospy.spin()

    def callback(self, msg):
        print(self.count)
        img_number = msg.filename[-10:]

        #fusion_path = self.fusion_dir + img_number.replace(".png", "")
        fusion_path = self.fusion_dir + "{0:06d}".format(self.count)
        if not os.path.exists(fusion_path):
            os.mkdir(fusion_path)

        #cnn_path = self.cnn_dir + img_number.replace(".png", "")
        cnn_path = self.cnn_dir + "{0:06d}".format(self.count)
        if not os.path.exists(cnn_path):
            os.mkdir(cnn_path)

        message_label_path = self.label_path + img_number.replace(".png", ".txt")

        fusion_label_destiation = fusion_path + "/label.txt"
        shutil.copyfile(message_label_path, fusion_label_destiation)

        cnn_label_destiation = cnn_path + "/label.txt"
        shutil.copyfile(message_label_path, cnn_label_destiation)


        fusion_detections = msg.detections
        cnn_detections = msg.image_detections.detections

        self.process_fusion(fusion_detections, fusion_path)
        self.process_detection(cnn_detections, cnn_path)

        self.count += 1

    def process_detection(self, cnn_detections, path):
        if len(cnn_detections) != 0:
            cnn_path = path + "/cnn.txt"
            f = open(cnn_path, 'w')

            for cnn in cnn_detections:
                p = cnn.probability
                if p > .6:

                    c = cnn.obj_class
                    if c == 1:
                        #vehicle
                        f.write("Car ")
                    elif c == 2:
                        #pedestrian
                        f.write("Pedestrian ")
                    elif c == 3:
                        #cyclist
                        f.write("Cyclist ")

                    f.write("-10 -10 -10 ")

                    x1 = cnn.x1
                    f.write("{0:.2f} ".format(x1))
                    y1 = cnn.y1
                    f.write("{0:.2f} ".format(y1))
                    x2 = cnn.x2
                    f.write("{0:.2f} ".format(x2))
                    y2 = cnn.y2
                    f.write("{0:.2f} ".format(y2))

                    f.write("-10 -10 -10 -10 -10 -10 -10 ")

                    f.write("{0:.2f}\n".format(p))

            f.close()


    def process_fusion(self, fusion_detections, path):
        if len(fusion_detections) != 0:
            fusion_path = path + "/fusion.txt"
            f = open(fusion_path, 'w')

            for fuse in fusion_detections:
                c = fuse.bbox.obj_class
                if c == 1:
                    #vehicle
                    f.write("Car ")
                elif c == 2:
                    #pedestrian
                    f.write("Pedestrian ")
                elif c == 3:
                    #cyclist
                    f.write("Cyclist ")

                f.write("-10 -10 ")

                a = fuse.a
                f.write("{0:.2f} ".format(a))

                x1 = fuse.bbox.x1
                f.write("{0:.2f} ".format(x1))
                y1 = fuse.bbox.y1
                f.write("{0:.2f} ".format(y1))
                x2 = fuse.bbox.x2
                f.write("{0:.2f} ".format(x2))
                y2 = fuse.bbox.y2
                f.write("{0:.2f} ".format(y2))

                h = fuse.h
                f.write("{0:.2f} ".format(h))
                w = fuse.w
                f.write("{0:.2f} ".format(w))
                l = fuse.l
                f.write("{0:.2f} ".format(l))

                x = fuse.x
                f.write("{0:.2f} ".format(x))
                y = fuse.y
                f.write("{0:.2f} ".format(y))
                z = fuse.z
                f.write("{0:.2f} ".format(z))

                f.write("-10 ")

                p = fuse.bbox.probability
                f.write("{0:.2f}\n".format(p))

            f.close()

if __name__ == "__main__":
    try:
        Evaluator()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start evaluator node.')
