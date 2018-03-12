import tensorflow as tf
from glob import glob
import sys
import numpy as np

sys.path.append("/home/mikep/Documents/Thesis/Code/LIDAR_Classification")

from Cluster import Cluster

def build_vars(input_shape, output_shape, wd=.0001):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        init = tf.truncated_normal_initializer(stddev=.01, dtype=dtype)
        kernel_shape = [input_shape, output_shape]
        bias_shape = [output_shape]
        kernel = tf.get_variable('kernel', kernel_shape, initializer=init, dtype=dtype)
        bias = tf.get_variable('bias', bias_shape, initializer=init, dtype=dtype)


        kernel_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name='kernel_loss')
        bias_decay = tf.multiply(tf.nn.l2_loss(bias), wd, name='bias_loss')

        tf.add_to_collection('losses', kernel_decay)
        tf.add_to_collection('losses', bias_decay)
    return kernel, bias

def composite_function(input, name, output_shape, relu=True):
    input_shape = input.get_shape().as_list()[1]

    with tf.variable_scope(name) as in_scope:
        matrix, bias = build_vars(input_shape, output_shape)

        output = tf.matmul(input, matrix, name='matmul') + bias

        if relu:
            output = tf.nn.relu(output, name='act')
        else:
            output = tf.nn.sigmoid(output, name='act')

    return output

def graph(input):
    layer1 = composite_function(input, 'layer1', output_shape=150)

    class_output = composite_function(layer1,  'class_output', output_shape=4)

    length_output = composite_function(layer1,  'length_output', output_shape=1, relu=False)

    z_output = composite_function(layer1,  'z_output', output_shape=1, relu=False)

    return layer1, tf.nn.softmax(class_output), length_output, z_output

def evaluate(clusters):
    input_tensor = tf.placeholder(tf.float32, shape=[None,15])

    layer1, class_output, length_output, z_output = graph(input_tensor)

    ckpt = tf.train.get_checkpoint_state("/home/mikep/Documents/Thesis/Code/LIDAR_Classification/")

    saver = tf.train.Saver()

    sess = tf.Session()

    saver.restore(sess, ckpt.model_checkpoint_path)

    """
    temp_cluster = []
    for i in range(15):
        temp_cluster.append(float(i))

    predicted_layer1, predicted_class_batch, predicted_length_batch, predicted_z_batch = sess.run([layer1, class_output, length_output, z_output], feed_dict={input_tensor: [temp_cluster]})
    print(predicted_layer1[0])
    print(predicted_class_batch[0])
    print(predicted_length_batch[0][0])
    print(predicted_z_batch[0][0])
    tensors = tf.trainable_variables()
    names_and_vals = []
    for tensor in tensors:
        tensor_name = tensor.name[:-2]
        tensor_value = tensor.eval(session=sess)
        temp_name_and_val = [tensor_name, tensor_value]
        names_and_vals.append(temp_name_and_val)

    for name_and_val in names_and_vals:
        name = name_and_val[0]
        val = name_and_val[1]

    layer1 = names_and_vals[0][1]
    layer1_b = names_and_vals[1][1]

    layer1 = np.array(layer1).transpose()
    print(layer1_b)
    #print(np.shape(layer1))

    output = np.dot(layer1, np.array(temp_cluster)) + layer1_b
    #output = np.dot(np.array(temp_cluster).transpose(), layer1) + layer1_b
    for i in range(len(output)):
        output[i] = max(0.0, output[i])
    #print(np.allclose(output, predicted_layer1))

    """
    batch_size = 100

    correct_nothing = 0.0
    correct_vehicle = 0.0
    correct_pedestrian = 0.0
    correct_cyclist = 0.0

    vehicle_z_error = 0.0
    pedestrian_z_error = 0.0
    cyclist_z_error = 0.0

    vehicle_l_error = 0.0
    pedestrian_l_error = 0.0
    cyclist_l_error = 0.0

    num_nothing = 0
    num_vehicle = 0
    num_pedestrian = 0
    num_cyclist = 0

    num = 0
    correct = 0.0
    z = 0.0
    l = 0.0

    for i in range(0,len(clusters),batch_size):

        cluster_batch = clusters[i:i+batch_size]
        batch_features = []
        batch_class = []
        batch_l = []
        batch_z = []
        for cluster in cluster_batch:
            batch_features.append(cluster.features)
            batch_class.append(cluster.label)
            batch_l.append(cluster.length)
            batch_z.append(cluster.z)


        predicted_class_batch, predicted_length_batch, predicted_z_batch = sess.run([class_output, length_output, z_output], feed_dict={input_tensor: batch_features})
        #print("batch_size = %d, len(batch) = %d", batch_size, len(predicted_class))


        for j in range(len(cluster_batch)):
            temp_class = predicted_class_batch[j]
            correct_class = batch_class[j]
            temp_z = predicted_z_batch[j][0]
            correct_z = batch_z[j]
            temp_l = predicted_length_batch[j][0]
            correct_l = batch_l[j]

            predicted_class = np.argmax(temp_class)

            z += (temp_z - correct_z)**2
            l += (temp_l - correct_l)**2
            if predicted_class == correct_class:
                correct += 1
            num += 1


            if correct_class == 0:
                num_nothing += 1
            elif correct_class == 1:
                num_vehicle += 1
            elif correct_class == 2:
                num_pedestrian += 1
            elif correct_class == 3:
                num_cyclist += 1
            x=5
            if predicted_class == 0:
                if correct_class == 0:
                    correct_nothing += 1
            elif predicted_class == 1:
                vehicle_l_error+= (temp_l - correct_l)**2
                vehicle_z_error+= (temp_z - correct_z)**2
                if correct_class == 1:
                    correct_vehicle += 1
            elif predicted_class == 2:
                pedestrian_l_error+= (temp_l - correct_l)**2
                pedestrian_z_error+= (temp_z - correct_z)**2
                if correct_class == 2:
                    correct_pedestrian += 1
            elif predicted_class == 3:
                cyclist_l_error+= (temp_l - correct_l)**2
                cyclist_z_error+= (temp_z - correct_z)**2
                if correct_class == 3:
                    correct_cyclist += 1


    correct_nothing /= num_nothing
    correct_vehicle /= num_vehicle
    correct_pedestrian /= num_pedestrian
    correct_cyclist /= num_cyclist

    vehicle_l_error /= num_vehicle / clusters[0].MAX_LENGTH
    pedestrian_l_error /= num_pedestrian / clusters[0].MAX_LENGTH
    cyclist_l_error /= num_cyclist / clusters[0].MAX_LENGTH

    vehicle_z_error /= num_vehicle / clusters[0].MAX_Z
    pedestrian_z_error /= num_pedestrian / clusters[0].MAX_Z
    cyclist_z_error /= num_cyclist / clusters[0].MAX_Z

    correct /= num
    z /= num
    l /= num
    print("Correct = {:.4f}, l = {:.4f}, z = {:.4f}".format(correct, l, z))

    print("Nothing = {:.4f}".format(correct_nothing))
    print("Vehicle = {:.4f}, l = {:.4f}, z = {:.4f}".format(correct_vehicle, vehicle_l_error, vehicle_z_error))
    print("Pedestrian = {:.4f}, l = {:.4f}, z = {:.4f}".format(correct_pedestrian, pedestrian_l_error, pedestrian_z_error))
    print("Cyclist = {:.4f}, l = {:.4f}, z = {:.4f}".format(correct_cyclist, cyclist_l_error, cyclist_z_error))
