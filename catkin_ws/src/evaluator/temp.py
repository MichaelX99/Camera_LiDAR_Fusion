from glob import glob
import pickle
import os
import copy
import numpy as np

CAR = 1
PEDESTRIAN = 2
CYCLIST = 3

WIDTH = 1240
HEIGHT = 375

SSD = 1
RFCN = 2
FRCNN = 3

NMS_THRESH = 0.5

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


class Evaluator:
    def __init__(self, base_path):
        self.base_path = base_path
        self.pickle_fpath = base_path + "pickle.p"

        self.confusion_matrix = [[0.,0.,0.],
                                 [0.,0.,0.],
                                 [0.,0.,0.]]

        if not os.path.isfile(self.pickle_fpath):
            print("created")
            cnn_folders = glob(base_path + "cnn/*")
            cnn_folders = sorted(cnn_folders)
            fusion_folders = glob(base_path + "fusion/*")
            fusion_folders = sorted(fusion_folders)

            self.cnn_outputs = []
            self.fusion_outputs = []
            self.labels = []

            for path in cnn_folders:
                temp_file = path + "/cnn.txt"
                self.cnn_outputs.append(temp_file)
                temp_file = path + "/label.txt"
                self.labels.append(temp_file)

            for path in fusion_folders:
                temp_file = path + "/fusion.txt"
                self.fusion_outputs.append(temp_file)

            self.gt_boxes = []
            self.cnn_boxes = []
            self.fusion_boxes = []
            for i in range(len(self.labels)):
                gt_box = self.get_labels(self.labels[i], WIDTH, HEIGHT)
                cnn_box = self.get_labels(self.cnn_outputs[i], WIDTH, HEIGHT, True)
                fusion_box = self.get_labels(self.fusion_outputs[i], WIDTH, HEIGHT, True)

                self.gt_boxes.append(gt_box)
                self.cnn_boxes.append(cnn_box)
                self.fusion_boxes.append(fusion_box)


            try:
                with open(self.pickle_fpath, 'wb') as pfile:
                    pickle.dump(
                        {   'gt_boxes': self.gt_boxes,
                            'cnn_boxes': self.cnn_boxes,
                            'fusion_boxes': self.fusion_boxes
                        },
                        pfile, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', fname, ':', e)
                raise

        else:
            print("loaded")
            with open(self.pickle_fpath, mode='rb') as f:
                data = pickle.load(f)

            self.gt_boxes = data['gt_boxes']
            self.cnn_boxes = data['cnn_boxes']
            self.fusion_boxes = data['fusion_boxes']

    def get_labels(self, path, width, height, detections=False):
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
            if detections:
                confidence = float(temp[15])

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

            if detections:
                bbox_obj = BBOX(xmin, xmax, ymin, ymax, label, confidence=confidence, truncated=truncated, occluded=occluded, path=path)
            else:
                bbox_obj = BBOX(xmin, xmax, ymin, ymax, label, truncated=truncated, occluded=occluded, path=path)
            output.append(bbox_obj)

        f.close()

        return output

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

            #self.confusion(temp_label, iter_det)
            self.confusion_matrix[iter_det.label-1][temp_label.label-1] += 1

    def confusion(self):
        for i, row in enumerate(self.confusion_matrix):
            s = 0.
            for elem in row: s+= elem
            for j, _ in enumerate(row): self.confusion_matrix[i][j] /= s

        self.confusion_matrix = np.asarray(self.confusion_matrix)

    def print_info(self, kind, all_detections, all_labels, nms = False):
        if kind == SSD:
            out_path = "SSD"
        elif kind == RFCN:
            out_path = "RFCN"
        elif kind == FRCNN:
            out_path = "FRCNN"


        num_labels = 0.
        for f in all_labels:
            for t in f:
                num_labels += 1

        # Determine how many detections are in categories 1, 2, 4
        total = 0.
        cat0 = 0.
        cat1 = 0.
        cat2 = 0.
        cat3 = 0.
        cat4 = 0.
        cat5 = 0.
        for image_detections in all_detections:
            for iter_det in image_detections:
                count_flag = True
                if not nms:
                    count_flag = True
                elif iter_det.confidence != 0:
                    count_flag = True
                var = 5
                if count_flag:
                    if iter_det.category == 0: cat0=cat0+1
                    elif iter_det.category == 1: cat1=cat1+1
                    elif iter_det.category == 2: cat2=cat2+1
                    elif iter_det.category == 3: cat3=cat3+1
                    elif iter_det.category == 4: cat4=cat4+1
                    elif iter_det.category == 5: cat5=cat5+1
                    total = total + 1

        print("There were " + str(cat0/total) + " detections of confidently correct objects")
        print("There were " + str(cat1/total) + " detections of confidently incorrect objects")
        print("There were " + str(cat2/total) + " detections of unconfidently correct objects")
        print("There were " + str(cat3/total) + " detections of unconfidently incorrect objects")
        print("There were " + str(cat4/total) + " detections of confident misses")
        print("There were " + str(cat5/total) + " detections of unconfident misses")
        print("Out of " + str(total) + " detections")

        top = cat0 - cat1 - cat3 - cat4
        print("Adjusted Accuracy = " + str(top/num_labels))

        self.confusion()
        print np.array_str(self.confusion_matrix, precision=4, suppress_small=True)
        print("---------------------\n")

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

        return output_detections
        #return sorted_detections


    def evaluate(self, kind):
        """
        Perform the network evaluation
        """

        #N = len(self.gt_boxes)

        if kind == SSD:
            print("\nFor the SSD model")
        elif kind == RFCN:
            print("\nFor the RFCN model")
        elif kind == FRCNN:
            print("\nFor the FRCNN model")

        cnn_nms_detections = []
        for image_detections in self.cnn_boxes:
            iter_nms = self.nms(image_detections)
            cnn_nms_detections.append(iter_nms)

        #Categorize Detections in the 6 categories
        for image_labels, cnn_detections in zip(self.gt_boxes, cnn_nms_detections):
            self.categorize_image(image_labels, cnn_detections)

        self.print_info(SSD, cnn_nms_detections, self.gt_boxes)

        if kind == SSD:
            print("\nFor the MFDS + SSD model")
        elif kind == RFCN:
            print("\nFor the MFDS + RFCN model")
        elif kind == FRCNN:
            print("\nFor the MFDS + FRCNN model")

        fusion_nms_detections = []
        for image_detections in self.fusion_boxes:
            iter_nms = self.nms(image_detections)
            fusion_nms_detections.append(iter_nms)

        """
        for image_labels, fusion_detections in zip(self.gt_boxes, self.fusion_boxes):
            self.categorize_image(image_labels, fusion_detections)

        self.print_info(SSD, self.fusion_boxes, self.gt_boxes)
        """

        for image_labels, fusion_detections in zip(self.gt_boxes, fusion_nms_detections):
            self.categorize_image(image_labels, fusion_detections)

        self.print_info(SSD, fusion_nms_detections, self.gt_boxes)


if __name__ == "__main__":
    ssd_eval = Evaluator("/home/mikep/Documents/Thesis/Code/Final/src/evaluator/ssd_evaluated/for_eval/")
    ssd_eval.evaluate(SSD)

    rfcn_eval = Evaluator("/home/mikep/Documents/Thesis/Code/Final/src/evaluator/rfcn_evaluated/for_eval/")
    rfcn_eval.evaluate(RFCN)

    frcnn_eval = Evaluator("/home/mikep/Documents/Thesis/Code/Final/src/evaluator/frcnn_evaluated/for_eval/")
    frcnn_eval.evaluate(FRCNN)
