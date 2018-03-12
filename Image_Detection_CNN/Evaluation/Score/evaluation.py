import tensorflow as tf
from glob import glob
from scipy.misc import imread
from time import time
import numpy as np
import cv2
import pickle
import os.path
import copy
from scipy.misc import imsave
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("/home/mikep/hdd/models/research")
sys.path.append("/home/mikep/hdd/models/research/object_detection")

from utils import label_map_util
from utils import visualization_utils as vis_util

CAR = 1
PEDESTRIAN = 2
CYCLIST = 3

WIDTH = 1240
HEIGHT = 375

NMS_THRESH = 0.5

SSD = 1
RFCN = 2
FRCN = 3
nms_SSD = 4
nms_RFCN = 5
nms_FRCN = 6

"""
Categories are:
[0] = Confident, Correct Object
[1] = Confident, Not Correct Object
[2] = Not Confident, Correct Object
[3] = Not Confident, Not Correct Object
[4] = Confident Miss
[5] = Not Confident Miss
"""

class BBOX(object):
    """
    Bounding Box object to store an individual bounding box
    """
    def __init__(self, x1, x2, y1, y2, label, truncated=None, occluded=None, confidence=1, width=WIDTH, height=HEIGHT, path=None):
        self.n_x1 = x1
        self.n_x2 = x2
        self.n_y1 = y1
        self.n_y2 = y2
        self.width = width
        self.height = height
        self.label = label
        self.confidence = confidence
        self.unnormalize()
        self.compute_centroid()
        self.category = None
        if truncated != None: self.truncated = truncated
        self.label_type = None
        if occluded != None: self.occluded = occluded
        if path != None: self.path = path

    def compute_centroid(self):
        """
        Computes the centroid of a bounding box
        """
        x = self.x2 - self.x1 + 1
        y = self.y2 - self.y1 + 1
        self.centroid = [x, y]

    def unnormalize(self):
        """
        Gets pixel coordinates not size ratios
        """
        self.x1 = self.n_x1 * self.width
        self.x2 = self.n_x2 * self.width
        self.y1 = self.n_y1 * self.height
        self.y2 = self.n_y2 * self.height

    def set_category(self, cat):
        if cat == 0:
            self.category = 0
        elif cat == 1:
            self.category = 1
        elif cat == 2:
            self.category = 2
        elif cat == 3:
            self.category = 3
        elif cat == 4:
            self.category = 4
        elif cat == 5:
            self.category = 5

class Evaluator(object):
    """
    Evaluation Object that will analyze the performance of a frozen detection model with respect to its test set
    """
    def __init__(self, path, kind):
        # Read in the graph stored in the frozen model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            det_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                serialized_graph = fid.read()
                det_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(det_graph_def, name='')

        # Load Image Information
        path_to_labels = "/home/mikep/DataSets/KITTI/Images/TFRecord/kitti_label_map.pbtxt"
        self.label_map = label_map_util.load_labelmap(path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=3, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Define the seesion we will be running in
        self.sess = tf.Session(graph=self.detection_graph)

        # The image size
        self.height = HEIGHT
        self.width = WIDTH

        # Store the total inference time of the validation set
        self.t = 0
        self.count = 0

        self.lower_bound = .25
        self.upper_bound = .75
        self.interested = 0
        self.counts = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

        # what model are we evaluating
        self.kind = kind

    def get_labels(self, path, width, height):
        """
        Function to retrieve the annotations in the label files
        """
        f = open(path)
        output = []
        for line in f:
            temp = line.split()
            type = temp[0]
            truncated = float(temp[1])
            occluded = int(temp[2])
            alpha = float(temp[3])
            xmin = float(temp[4]) / width
            ymax = float(temp[7]) / height
            xmax = float(temp[6]) / width
            ymin = float(temp[5]) / height
            v_height = float(temp[8])
            v_width = float(temp[9])
            v_length = float(temp[10])
            x = float(temp[11])
            y = float(temp[12])
            z = float(temp[13])
            rotation_y = float(temp[14])

            if (type == 'Car') or (type == 'Van') or (type == 'Truck'):
                type = 'Car'
                label = CAR
            elif (type == 'Pedestrian') or (type == 'Person_sitting'):
                type = 'Pedestrian'
                label = PEDESTRIAN
            elif (type == 'Cyclist'):
                label = CYCLIST
            else:
                type = 'DontCare'
                continue


            bbox_obj = BBOX(xmin, xmax, ymin, ymax, label, truncated=truncated, occluded=occluded, path=path)
            output.append(bbox_obj)

        return output


    def network_inference(self, img, img_path):
        """
        Perform inference of the detection network to retrieve a list of bounding box objects
        """
        t1 = time()
        detections, scores, classes, number = self.sess.run([self.detection_boxes,
                                                             self.detection_scores,
                                                             self.detection_classes,
                                                             self.num_detections], feed_dict={self.image_tensor: [img]})
        t2 = time()
        self.t = self.t + (t2-t1)
        self.count = self.count + 1

        output = []
        for i in range(number):
            y1 = detections[0][i][0]
            x1 = detections[0][i][1]
            y2 = detections[0][i][2]
            x2 = detections[0][i][3]
            confidence = scores[0][i]
            if (x1 == 0) and (y1 == 0) and (x2 == 0) and (y2 == 0):
                continue
            if (confidence < self.upper_bound) and (confidence > self.lower_bound):
                self.interested = self.interested + 1
            label = classes[0][i]
            iter_bbox = BBOX(x1, x2, y1, y2, label, confidence=confidence, path=img_path)
            output.append(iter_bbox)
            if (confidence < .1):
                self.counts[0] = self.counts[0] + 1
            elif (confidence < .2) and (confidence >= .1):
                self.counts[1] = self.counts[1] + 1
            elif (confidence < .3) and (confidence >= .2):
                self.counts[2] = self.counts[2] + 1
            elif (confidence < .4) and (confidence >= .3):
                self.counts[3] = self.counts[3] + 1
            elif (confidence < .5) and (confidence >= .4):
                self.counts[4] = self.counts[4] + 1
            elif (confidence < .6) and (confidence >= .5):
                self.counts[5] = self.counts[5] + 1
            elif (confidence < .7) and (confidence >= .6):
                self.counts[6] = self.counts[6] + 1
            elif (confidence < .8) and (confidence >= .7):
                self.counts[7] = self.counts[7] + 1
            elif (confidence < .9) and (confidence >= .8):
                self.counts[8] = self.counts[8] + 1
            elif (confidence >= .9):
                self.counts[9] = self.counts[9] + 1

        return output

    def retrieve_bboxes(self, label_paths, img_paths):
        """
        Retrieve the list of inferences from the network and labels
        """
        fname = "saved"
        if self.kind == SSD:
            fname = fname + "_SSD.p"
        elif self.kind == RFCN:
            fname = fname + "_RFCN.p"
        elif self.kind == FRCN:
            fname = fname + "_FRCN.p"
        elif self.kind == nms_SSD:
            fname = fname + "_nms_SSD.p"
        elif self.kind == nms_RFCN:
            fname = fname + "_nms_RFCN.p"
        elif self.kind == nms_FRCN:
            fname = fname + "_nms_FRCN.p"
        var = 5
        if not os.path.isfile(fname):
            label_bboxes = []
            detection_bboxes = []

            for img_path, label_path in zip(img_paths, label_paths):
                img = imread(img_path)
                height = img.shape[1]
                width = img.shape[0]
                label_bbox = self.get_labels(label_path, height, width)
                detections = self.network_inference(img, img_path)
                label_bboxes.append(label_bbox)
                detection_bboxes.append(detections)

            try:
                with open(fname, 'wb') as pfile:
                    pickle.dump(
                        {   'label_bboxes': label_bboxes,
                            'detection_bboxes': detection_bboxes,
                            'self.t': self.t,
                            'self.count': self.count,
                            'self.counts': self.counts,
                            'self.interested': self.interested
                        },
                        pfile, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', fname, ':', e)
                raise
        else:
            with open(fname, mode='rb') as f:
                data = pickle.load(f)

            label_bboxes = data['label_bboxes']
            detection_bboxes = data['detection_bboxes']
            self.t = data['self.t']
            self.count = data['self.count']
            self.counts = data['self.counts']
            self.interested = data['self.interested']


        return label_bboxes, detection_bboxes

    def compute_union(self, label, detection, itersection):
        """Computes the Union of 2 bounding boxes
        """
        label_area = (label.x2 - label.x1 + 1) * (label.y2 - label.y1 + 1)
        detection_area = (detection.x2 - detection.x1 + 1) * (detection.y2 - detection.y1 + 1)

        union = label_area + detection_area - itersection

        return union

    def compute_intersection(self, label, detection):
        """
        Computes the Intersection of 2 bounding boxes
        """
        x1 = max(label.x1, detection.x1)
        x2 = max(label.x2, detection.x2)
        y1 = min(label.y1, detection.y1)
        y2 = min(label.y2, detection.y2)

        itersection = (x2 - x1 + 1) * (y2 - y1 + 1)
        return itersection

    def compute_iou(self, label, detection):
        """
        Computes the IOU for a single label and detection
        """
        intersection = self.compute_intersection(label, detection)
        union = self.compute_union(label, detection, intersection)
        return intersection/union

    def overlap(self, label, detection):
        """
        Determines if the 2 boxes overlap
        """
        if (detection.x2 < label.x1): return False # detection is left of label
        elif (detection.x1 > label.x2): return False # detection is right of label
        elif (detection.y2 < label.y1): return False # detection is above label
        elif (detection.y1 > label.y2): return False # detection is below label
        return True

    def categorize_detection(self, label, detection, correctness, iou):
        """
        Puts a single detection into a category
        [0]: confident, correct object detection
        [1]: confident, not correct object detection
        [2]: not confident, correct object detection
        [3]: not confident, not correct object detection
        """
        if (detection.confidence >= NMS_THRESH) and (correctness == True): detection.set_category(0)
        elif (detection.confidence >= NMS_THRESH) and (correctness == False): detection.set_category(1)
        elif (detection.confidence < NMS_THRESH) and (correctness == True): detection.set_category(2)
        elif (detection.confidence < NMS_THRESH) and (correctness == False): detection.set_category(3)

    def rank_label(self, label):
        """
        Ranks a label according to the KITTI standards
        [0]: Easy
        [1]: Moderate
        [2]: Hard
        [3]: Don't Care
        """
        occlusion = label.occluded
        truncation = label.truncated
        h = label.y2 - label.y1
        if (h > 40) and (occlusion == 0) and (truncation < .15):
            return 0
        elif (h > 25) and (occlusion == 1) and (truncation < .30):
            return 1
        elif (h > 25) and (occlusion == 2) and (truncation < .50):
            return 2
        else:
            return 3

    def categorize_image(self, labels, detections):
        """
        Put detections from a single image into one of six categories.
        [0]: confident, correct object detection
        [1]: confident, not correct object detection
        [2]: not confident, correct object detection
        [3]: not confident, not correct object detection
        [4]: confident nothing
        [5]: not confident nothing
        """
        for iter_det in detections:
            overlapping_labels = []
            for iter_label in labels:
                # does the detection overlap with any label?
                if self.overlap(iter_label, iter_det):
                    iou = self.compute_iou(iter_label, iter_det) # Compute the IOU
                    label_difficulty = self.rank_label(iter_label) # Determine the label difficulty
                    if iter_label.label == iter_det.label: correctness = True # Is the detection correct
                    else: correctness = False
                    temp = [iter_label, iou, correctness, label_difficulty]
                    overlapping_labels.append(temp)

            # Was there a label that it matched up with
            if len(overlapping_labels) == 0:
                if iter_det.confidence > NMS_THRESH: iter_det.set_category(4)
                else: iter_det.set_category(5)
                continue

            # If there are more than one overlapping labels
            elif len(overlapping_labels) != 1:
                max_iou = -10000000
                best_overlap = None
                for overlap in overlapping_labels:
                    if overlap[1] > max_iou:
                        max_iou = overlap[1]
                        best_overlap = overlap
                overlapping_labels = []
                overlapping_labels = [best_overlap]

            #print(overlapping_labels)
            temp_label = overlapping_labels[0][0]
            temp_iou = overlapping_labels[0][1]
            temp_correctness = overlapping_labels[0][2]
            temp_difficulty = overlapping_labels[0][3]
            self.categorize_detection(temp_label, iter_det, temp_correctness, temp_iou)
            iter_det.difficulty = temp_difficulty

    def save_img(self, image_detections, counter, out_path):
        img = imread(image_detections[0].path)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        non = False
        boxes = []
        classes = []
        scores = []
        for iter_det in image_detections:
            x1 = iter_det.x1
            y1 = iter_det.y1
            x2 = iter_det.x2
            y2 = iter_det.y2
            iter_class = iter_det.label
            iter_score = iter_det.confidence
            boxes.append([y1,x1,y2,x2])
            classes.append(iter_class)
            scores.append(iter_score)

            if iter_score > .15:
                if iter_class == CAR:
                    color = (255,0,0)
                elif iter_class == PEDESTRIAN:
                    color = (0,255,0)
                elif iter_class == CYCLIST:
                    color = (0,0,255)
                cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, thickness=2)
                cv2.putText(img=img, text=str(iter_score)[:4], org=(int(x1),int(y1)-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.6, color=color)


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        """
        boxes = [boxes]
        classes = [classes]
        scores = [scores]

        vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=False,
            line_thickness=8)
        """

        fpath = out_path + str(counter) + ".png"
        imsave(fpath, img)

    def print_info(self, all_detections, nms = False):
        if self.kind == SSD:
            out_path = "SSD"
        elif self.kind == RFCN:
            out_path = "RFCN"
        elif self.kind == FRCN:
            out_path = "FRCN"
        elif self.kind == nms_SSD:
            out_path = "nms_SSD"
        elif self.kind == nms_RFCN:
            out_path = "nms_RFCN"
        elif self.kind == nms_FRCN:
            out_path = "nms_FRCN"

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_path = out_path + "/"

        # Determine how many detections are in categories 1, 2, 4
        total = 0.
        cat0 = 0.
        cat1 = 0.
        cat2 = 0.
        cat3 = 0.
        cat4 = 0.
        cat5 = 0.
        counter = 0
        for image_detections in all_detections:
            flag = False
            for iter_det in image_detections:
                count_flag = False
                if not nms:
                    count_flag = True
                elif iter_det.confidence != 0:
                    count_flag = True
                var = 5
                if count_flag:
                    if iter_det.category == 0: cat0=cat0+1
                    elif iter_det.category == 1:
                        cat1=cat1+1
                        flag = True
                    elif iter_det.category == 2:
                        cat2=cat2+1
                        flag = True
                    elif iter_det.category == 3: cat3=cat3+1
                    elif iter_det.category == 4:
                        cat4=cat4+1
                        flag = True
                    elif iter_det.category == 5: cat5=cat5+1
                    total = total + 1
            if flag and nms==True:
                self.save_img(image_detections, counter, out_path)
                counter = counter + 1

        if not nms:
            lower = 0.0
            upper = 0.1
            for num in self.counts:
                print(str(num/total) + " detections between " + str(lower) + " and " +str(upper) + " confidence")
                lower = upper
                upper = upper + 0.1
        print("There were " + str(cat0/total) + " detections of confidently correct objects")
        print("There were " + str(cat1/total) + " detections of confidently incorrect objects")
        print("There were " + str(cat2/total) + " detections of unconfidently correct objects")
        print("There were " + str(cat3/total) + " detections of unconfidently incorrect objects")
        print("There were " + str(cat4/total) + " detections of confident misses")
        print("There were " + str(cat5/total) + " detections of unconfident misses")
        print("Out of " + str(total) + " detections")
        if not nms:
            print("There were " + str(self.interested/total) + " detections within the confidence bounds\n")

    def sort(self, detections):
        """
        Sorts bounding boxes from highest confidence to lowest
        """
        # Make a deep copy
        temp_detections = copy.deepcopy(detections)
        sorted_detections = []
        # While there are detections still left
        while len(temp_detections) != 0:
            max_confidence = -10000000
            max_ind = 0
            # Determine which bounding box that is left has the largest confidence
            for i in range(len(temp_detections)):
                iter_det = temp_detections[i]
                if max_confidence < iter_det.confidence:
                    max_confidence = iter_det.confidence
                    max_ind = i
            # Add the largest confidence bounding box to our sorted list and delete it from the iterating list
            sorted_detections.append(temp_detections[max_ind])
            del temp_detections[max_ind]

        return sorted_detections

    def nms(self, detections):
        """
        Perform NMS on the bounding boxes in order to find which boxes are going to be deleted
        """
        sorted_detections = self.sort(detections)
        output_detections = []
        for i in range(len(sorted_detections)):
            for j in range(i+1, len(sorted_detections)):
                if self.overlap(sorted_detections[i], sorted_detections[j]):
                    temp_iou = self.compute_iou(sorted_detections[i], sorted_detections[j])
                    if temp_iou > NMS_THRESH:
                        sorted_detections[j].confidence = 0
        for iter_det in sorted_detections:
            if iter_det.confidence != 0:
                output_detections.append(iter_det)

        return sorted_detections#output_detections


    def evaluate(self, path):
        """
        Perform the network evaluation
        """

        # Find all the evaluation image paths
        img_paths = glob(path+"/Images/*")
        img_paths.sort()

        # Find all the evaluation label paths
        label_paths = glob(path+"/Labels/*")
        label_paths.sort()

        #img_paths = img_paths[:1]
        #label_paths = label_paths[:1]

        assert len(img_paths) == len(label_paths)
        N = len(img_paths)

        if self.kind == SSD:
            print("\nFor the SSD model")
        elif self.kind == RFCN:
            print("\nFor the RFCN model")
        elif self.kind == FRCN:
            print("\nFor the FRCN model")
        elif self.kind == nms_SSD:
            print("\nFor the nms_SSD model")
        elif self.kind == nms_RFCN:
            print("\nFor the nms_RFCN model")
        elif self.kind == nms_FRCN:
            print("\nFor the nms_FRCN model")

        # Get all the labels and all the network predicitions
        all_labels, all_detections = self.retrieve_bboxes(label_paths, img_paths)
        print("Average Inference time = " + str(self.t/self.count))

        nms_detections = []
        for image_detections in all_detections:
            iter_nms = self.nms(image_detections)
            nms_detections.append(iter_nms)

        # Categorize Detections in the 6 categories
        for image_labels, image_detections, nms in zip(all_labels, all_detections, nms_detections):
            self.categorize_image(image_labels, image_detections)
            self.categorize_image(image_labels, nms)


        self.print_info(all_detections)
        self.print_info(nms_detections, nms=True)




if __name__ == "__main__":
    validation_path = "/home/mikep/DataSets/KITTI/Images/Validation"

    ssd_path = "/home/mikep/Documents/Thesis/Code/Detection_Codes/kitti_ssd/output_inference_graph/frozen_inference_graph.pb"
    ssd_eval = Evaluator(ssd_path, SSD)

    rfcn_path = "/home/mikep/Documents/Thesis/Code/Detection_Codes/kitti_rfcn/output_inference_graph/frozen_inference_graph.pb"
    rfcn_eval = Evaluator(rfcn_path, RFCN)

    frcn_path = "/home/mikep/Documents/Thesis/Code/Detection_Codes/kitti_faster_rcnn/output_inference_graph/frozen_inference_graph.pb"
    frcn_eval = Evaluator(frcn_path, FRCN)

    nms_ssd_path = "/home/mikep/Documents/Thesis/Code/Detection_Codes/kitti_ssd/nms_output_inference_graph/frozen_inference_graph.pb"
    nms_ssd_eval = Evaluator(nms_ssd_path, nms_SSD)

    nms_rfcn_path = "/home/mikep/Documents/Thesis/Code/Detection_Codes/kitti_rfcn/nms_output_inference_graph/frozen_inference_graph.pb"
    nms_rfcn_eval = Evaluator(nms_rfcn_path, nms_RFCN)

    nms_frcn_path = "/home/mikep/Documents/Thesis/Code/Detection_Codes/kitti_faster_rcnn/nms_output_inference_graph/frozen_inference_graph.pb"
    nms_frcn_eval = Evaluator(nms_frcn_path, nms_FRCN)

    ssd_eval.evaluate(validation_path)
    rfcn_eval.evaluate(validation_path)
    frcn_eval.evaluate(validation_path)
    """
    nms_ssd_eval.evaluate(validation_path)
    nms_rfcn_eval.evaluate(validation_path)
    nms_frcn_eval.evaluate(validation_path)
    """
