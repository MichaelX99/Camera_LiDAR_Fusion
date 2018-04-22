import scipy.io as sio
import glob
import tensorflow as tf
from sklearn.utils import shuffle
#import cv2
import numpy as np
import os

#import tflearn

IMAGENET = '/home/mikep/DataSets/ImageNet2012/'
#IMAGENET = '/home/mikep/DataSets/CIFAR10/'
TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999
N_GPUS = 3
SPLIT_BATCH_SIZE = 64
BATCH_SIZE = SPLIT_BATCH_SIZE * N_GPUS
IMAGE_SIZE = 224
#IMAGE_SIZE = 32
NUM_CLASSES = 1000
#NUM_CLASSES = 10
#DECAY = 5e-4
DECAY = 1e-4
TRAIN_SHARDS = 128
#TRAIN_SHARDS = 16
VALIDATION_SHARDS = 24
#VALIDATION_SHARDS = 8
NUM_THREADS = 4
MAX_EPOCHS = 150
#MAX_STEPS = MAX_EPOCHS * (50000 / (BATCH_SIZE))
MAX_STEPS = MAX_EPOCHS * (1281167 / (BATCH_SIZE))

def count_params():
    total = 0;
    for var in tf.trainable_variables():
        var_shape = var.get_shape().as_list()
        temp = 1
        for dim in var_shape:
            temp *= dim
        total += temp
    return total


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]

    Returns:
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    #labels = tf.cast(labels, tf.int32)
    labels = tf.cast(labels, tf.int64)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    l2_loss = l2 * DECAY
    tf.add_to_collection('losses', l2_loss)

    softmax = tf.nn.softmax(logits)

    preds = tf.argmax(softmax, axis=1)
    #preds = tf.cast(preds, tf.int32)
    correct = tf.equal(preds, labels)
    tf.add_to_collection('accuracy', tf.reduce_mean(tf.cast(correct, tf.float32)))

    tf.add_to_collection('cross_entropies', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    #return tf.add_n(tf.get_collection('losses'), name='total_loss')
    return cross_entropy_mean + l2_loss

def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def variable_with_weight_decay(name, shape, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

    Returns:
    Variable Tensor
    """
    dtype = tf.float32
    var = variable_on_cpu(
                        name,
                        shape,
                        tf.contrib.layers.variance_scaling_initializer())
    return var

def convolution(conv, shape, decay, stride, in_scope, dropout, is_training):
    with tf.variable_scope(in_scope) as scope:
        kernel = variable_with_weight_decay('weights',
                                            shape=shape,
                                            wd=decay)

        conv = tf.contrib.layers.batch_norm(conv, scale=True, is_training=is_training, center=True, fused=True)
        conv = tf.nn.relu(conv, name=scope.name)

        conv = tf.nn.conv2d(conv, kernel, [1, stride, stride, 1], padding='SAME')

        #conv = tf.nn.dropout(conv, dropout)

        #print(in_scope + ": " + str(conv.get_shape().as_list()))

    return conv



def transition(input, scope, d_out, dropout, is_training):
    d_in = input.get_shape().as_list()[3]

    transition = tf.nn.avg_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    transition = convolution(transition, [1,1,d_in,d_out], DECAY, 1, scope, dropout, is_training)


    return transition

def inception(input, scope, dropout, is_training, red3, red5, x1, x3, x5, pool_proj):
    d_in = input.get_shape().as_list()[3]

    with tf.name_scope(scope):
        conv_red3 = convolution(input, [1,1,d_in,red3], DECAY, 1, scope+'_red3', dropout, is_training)
        conv_3x3 = convolution(conv_red3, [3,3,red3,red3], DECAY, 1, scope+'_3x3', dropout, is_training)
        conv_expand3 = convolution(conv_3x3, [1,1,red3,x3], DECAY, 1, scope+'_expand3', dropout, is_training)

        conv_red5 = convolution(input, [1,1,d_in,red5], DECAY, 1, scope+'_red5', dropout, is_training)
        conv3_1 = convolution(conv_red5, [3,3,red5,red5], DECAY, 1, scope+'_conv3_1', dropout, is_training)
        conv3_2 = convolution(conv3_1, [3,3,red5,red5], DECAY, 1, scope+'_conv3_2', dropout, is_training)
        conv_expand5 = convolution(conv3_2, [1,1,red5,x5], DECAY, 1, scope+'_expand5', dropout, is_training)

        output = tf.add_n([conv_expand3, conv_expand5])

        return output

def dense_inception(input, scope, dropout, is_training, red3, red5, x1, x3, x5, pool_proj):
    d_in = input.get_shape().as_list()[3]
    N = len(red3)

    with tf.name_scope(scope):
        conv = inception(input, scope+'_inception0', dropout, is_training, red3[0], red5[0], x1[0], x3[0], x5[0], pool_proj[0])
        conv = tf.add_n([input, conv])

        for i in range(N - 1):
            j = i + 1
            d_in = conv.get_shape().as_list()[3]

            l = inception(conv, scope+'_inception'+str(j), dropout, is_training, red3[j], red5[j], x1[j], x3[j], x5[j], pool_proj[j])

            conv = tf.add_n([conv, l])

        return conv

def softmax_classifier(input, scope, dropout, is_training):
    with tf.variable_scope(scope) as scope:
        k = input.get_shape().as_list()[1]
        d = input.get_shape().as_list()[3]

        conv = convolution(input, [1,1,d,NUM_CLASSES], DECAY, 1, scope.name+'_conv', dropout, is_training)

        conv = tf.contrib.layers.batch_norm(conv, scale=True, is_training=is_training, center=True, fused=True)

        activated = tf.nn.relu(conv, name=scope.name)

        #avg_pool = tf.nn.avg_pool(activated, ksize=[1,k,k,1], strides=[1,k,k,1], padding='VALID')

        #dense = tf.contrib.layers.flatten(avg_pool)
        dense = tf.reduce_mean(activated, axis=[1,2])

        d = dense.get_shape().as_list()[1]

    with tf.variable_scope("fc") as scope:
        kernel = variable_on_cpu('weights', shape=[NUM_CLASSES,NUM_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
        output = tf.matmul(dense, kernel)

    with tf.device('/cpu:0'):
        dtype = tf.float32
        biases = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[NUM_CLASSES]), dtype=dtype)

    with tf.variable_scope("fc") as scope:
        output = output + biases

        return output
