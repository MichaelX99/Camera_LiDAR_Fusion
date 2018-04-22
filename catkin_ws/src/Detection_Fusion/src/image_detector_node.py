#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge
import numpy as np
import tensorflow as tf
import cv2
from detection_fusion.msg import *

class ImageDetector(object):
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('image_detector')

        # Get the model path from the ROS parameter server
        model_path = rospy.get_param("image_model")

        # Load the graph into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            det_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                det_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(det_graph_def, name='')

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # To not take up all the memory of the gpu in order to leave some for the point cloud classifier
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Define the session to evaluate incoming images
        self.sess = tf.Session(graph=self.detection_graph, config=config)

        # Bridge from sensor_msgs.image to CV Mat
        self.bridge = CvBridge()

        # Subscriber to the synchronized incoming data
        self.subscriber = rospy.Subscriber('/incoming_data', Sensor_Data, self.detect_image)

        # Publisher for the image only detections
        self.detection_publisher = rospy.Publisher('/image_detections', Image_Detections, queue_size=1)

        # Publisher to the next fusion node
        self.fusion_publisher = rospy.Publisher('/detections_and_cloud', Detections_Cloud, queue_size=1)

        self.img_pub = rospy.Publisher('/debug_img', Image, queue_size=1)

        rospy.spin()

    def detect_image(self, msg):
        filename = msg.filename
        image = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        og_image = np.copy(image)

        height = image.shape[0]
        width = image.shape[1]

        boxes, scores, classes, num = self.sess.run([self.detection_boxes,
                                                     self.detection_scores,
                                                     self.detection_classes,
                                                     self.num_detections],
                                                    feed_dict={self.image_tensor: [image]})
        num = num[0]

        detections = []
        to_fuse = Detections_Cloud()
        for i in range(num):
            BBox = BBox2D()
            BBox.obj_class = classes[0][i]
            BBox.probability = scores[0][i]
            BBox.x1 = int(boxes[0][i][1] * width)
            BBox.x2 = int(boxes[0][i][3] * width)
            BBox.y1 = int(boxes[0][i][0] * height)
            BBox.y2 = int(boxes[0][i][2] * height)
            detections.append(BBox)

            if BBox.probability > .6:
                image = cv2.rectangle(image, (BBox.x1,BBox.y1), (BBox.x2,BBox.y2), (125,125,125), 3)

        self.img_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

        detection_msg = Image_Detections()
        detection_msg.detections = detections

        to_fuse.pcloud = msg.pcloud
        to_fuse.detections = detection_msg
        to_fuse.image = self.bridge.cv2_to_imgmsg(og_image, "bgr8")
        to_fuse.filename = filename

        self.detection_publisher.publish(detections)
        self.fusion_publisher.publish(to_fuse)

if __name__ == "__main__":
    try:
        ImageDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start image detector node.')
