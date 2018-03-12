import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from glob import glob
from scipy.misc import imsave

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("/home/mikep/hdd/models/research")
sys.path.append("/home/mikep/hdd/models/research/object_detection")

from utils import label_map_util
from utils import visualization_utils as vis_util

NUM_CLASSES = 3

SSD = 1
RFCN = 2
FRCN = 3
nms_SSD = 4
nms_RFCN = 5
nms_FRCN = 6

class Box_Drawer(object):
    def __init__(self, path_to_graph, path_to_labels, path_to_images, kind):
        self.path_to_graph = path_to_graph
        self.path_to_labels = path_to_labels
        self.path_to_images = path_to_images
        self.kind = kind

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          self.od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(self.path_to_graph, 'rb') as fid:
            serialized_graph = fid.read()
            self.od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(self.od_graph_def, name='')

        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        self.sess = tf.Session(graph=self.detection_graph)

    def load_image_into_numpy_array(self, image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

    def draw(self):
        # Definite input and output Tensors for detection_graph
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        count = 0

        out_path = "images"
        if self.kind == SSD:
            out_path = "SSD_"+out_path
        elif self.kind == RFCN:
            out_path = "RFCN_"+out_path
        elif self.kind == FRCN:
            out_path = "FRCN_"+out_path
        elif self.kind == nms_SSD:
            out_path = "nms_SSD_"+out_path
        elif self.kind == nms_RFCN:
            out_path = "nms_RFCN_"+out_path
        elif self.kind == nms_FRCN:
            out_path = "nms_FRCN_"+out_path
        var = 5

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_path = out_path + "/"

        for image_path in self.path_to_images:
          image = Image.open(image_path)
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          image_np = self.load_image_into_numpy_array(image)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
          (boxes, scores, classes, num) = self.sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              self.category_index,
              use_normalized_coordinates=True,
              line_thickness=8)

          fpath = out_path + str(count) + ".png"
          imsave(fpath, image_np)
          count = count + 1


if __name__ == "__main__":
    test_image_paths = glob("/home/mikep/DataSets/KITTI/Images/Validation/Images/*.png")
    path_to_labels = "/home/mikep/DataSets/KITTI/Images/TFRecord/kitti_label_map.pbtxt"

    ssd_path = "/home/mikep/Documents/Thesis/Code/Detection_Codes/kitti_ssd/output_inference_graph/frozen_inference_graph.pb"
    ssd_drawer = Box_Drawer(ssd_path, path_to_labels, test_image_paths, SSD)

    rfcn_path = "/home/mikep/Documents/Thesis/Code/Detection_Codes/kitti_rfcn/output_inference_graph/frozen_inference_graph.pb"
    rfcn_drawer = Box_Drawer(rfcn_path, path_to_labels, test_image_paths, RFCN)

    frcn_path = "/home/mikep/Documents/Thesis/Code/Detection_Codes/kitti_faster_rcnn/output_inference_graph/frozen_inference_graph.pb"
    frcn_drawer = Box_Drawer(frcn_path, path_to_labels, test_image_paths, FRCN)

    nms_ssd_path = "/home/mikep/Documents/Thesis/Code/Detection_Codes/kitti_ssd/nms_output_inference_graph/frozen_inference_graph.pb"
    nms_ssd_drawer = Box_Drawer(nms_ssd_path, path_to_labels, test_image_paths, nms_SSD)

    nms_rfcn_path = "/home/mikep/Documents/Thesis/Code/Detection_Codes/kitti_rfcn/nms_output_inference_graph/frozen_inference_graph.pb"
    nms_rfcn_drawer = Box_Drawer(nms_rfcn_path, path_to_labels, test_image_paths, nms_RFCN)

    nms_frcn_path = "/home/mikep/Documents/Thesis/Code/Detection_Codes/kitti_faster_rcnn/nms_output_inference_graph/frozen_inference_graph.pb"
    nms_frcn_drawer = Box_Drawer(nms_frcn_path, path_to_labels, test_image_paths, nms_FRCN)

    print("Starting SSD")
    ssd_drawer.draw()

    print("Starting RFCN")
    rfcn_drawer.draw()

    print("Starting FRCN")
    frcn_drawer.draw()

    print("Starting nms SSD")
    nms_ssd_drawer.draw()

    print("Starting nms RFCN")
    nms_rfcn_drawer.draw()

    print("Starting nms FRCN")
    nms_frcn_drawer.draw()
