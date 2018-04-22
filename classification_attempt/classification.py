import tensorflow as tf
import helper

def inference(images, dropout, is_training):
    L = 20
    N = (L - 6) // 4

    d4 = 1024
    d4_red = d4 / 4

    d3 = d4 / 2
    d3_red = d3 / 4

    d2 = d3 / 2
    d2_red = d2 / 4

    d1 = d2 / 2
    d1_red = d1 / 4

    # conv1
    conv1 = helper.convolution(images, [3,3,3,d1_red], helper.DECAY, 1, 'conv1', dropout, is_training)
    conv1 = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    conv2 = helper.convolution(conv1, [3,3,d1_red,d1], helper.DECAY, 1, 'conv2', dropout, is_training)
    conv2 = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    x1 = N * [d1]
    red3 = N * [d1_red]
    x3 = N * [d1]
    red5 = N * [d1_red]
    x5 = N * [d1]
    pool_proj = N * [d1]

    block1 = helper.dense_inception(conv2, 'block1', dropout, is_training, red3, red5, x1, x3, x5, pool_proj)
    transition1 = helper.transition(block1, 'transition1', d2, dropout, is_training)

    x1 = N * [d2]
    red3 = N * [d2_red]
    x3 = N * [d2]
    red5 = N * [d2_red]
    x5 = N * [d2]
    pool_proj = N * [d2]

    block2 = helper.dense_inception(transition1, 'block2', dropout, is_training, red3, red5, x1, x3, x5, pool_proj)
    transition2 = helper.transition(block2, 'transition2', d3, dropout, is_training)

    x1 = N * [d3]
    red3 = N * [d3_red]
    x3 = N * [d3]
    red5 = N * [d3_red]
    x5 = N * [d3]
    pool_proj = N * [d3]

    block3 = helper.dense_inception(transition2, 'block3', dropout, is_training, red3, red5, x1, x3, x5, pool_proj)
    transition3 = helper.transition(block3, 'transition3', d4, dropout, is_training)

    x1 = N * [d4]
    red3 = N * [d4_red]
    x3 = N * [d4]
    red5 = N * [d4_red]
    x5 = N * [d4]
    pool_proj = N * [d4]

    block4 = helper.dense_inception(transition3, 'block4', dropout, is_training, red3, red5, x1, x3, x5, pool_proj)

    output = helper.softmax_classifier(block4, 'output', dropout, is_training)

    return output
