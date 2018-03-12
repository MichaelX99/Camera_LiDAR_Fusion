import tensorflow as tf
from classifier import Classifier
import sys

sys.path.append("/home/mikep/Documents/Thesis/Code/LIDAR_Classification/Evaluation")
from evaluate import *

if __name__ == "__main__":
    base_path = "/home/mikep/DataSets/KITTI/Clusters/"
    lidar_classifier = Classifier(base_path)


    best_acc = -1000000000.

    for _ in range(10):
        iter_acc = lidar_classifier.train()
        if iter_acc > best_acc:
            best_acc = iter_acc
            lidar_classifier.save()
        lidar_classifier.sess.run(lidar_classifier.init)
        print("\n")
    tf.reset_default_graph()


    evaluate(lidar_classifier.test_clusters)
