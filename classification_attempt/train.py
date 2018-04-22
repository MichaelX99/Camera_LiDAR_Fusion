from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

import tensorflow as tf
import numpy as np
import re
import time
from datetime import datetime
import copy
#import os
import math
import helper
import classification

from imagenet_data import ImagenetData, CIFARData
import image_processing

SAVE_POINT = 'model/'

RESTORE = True

from tensorflow.python.client import timeline

def tower_loss(scope, images, labels, dropout, is_training):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  logits = classification.inference(images, dropout, is_training)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = helper.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def evaluate(images, labels, sess, dropout, is_training, train):
    #Evaluate model on Dataset for a number of steps.
    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.

    # Build a Graph that computes the logits predictions from the
    # inference model.
    """
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
      [images, labels], capacity=2 * helper.N_GPUS)

    print(images.get_shape().as_list())
    with tf.device('/gpu:0'):
        image_batch0, label_batch0 = batch_queue.dequeue()
        tf.get_variable_scope().reuse_variables()
        logits0 = classification.inference(image_batch0, dropout, is_training)
        #logits = classification.inference(images, dropout, is_training)
        softmax0 = tf.nn.softmax(logits0)
    top_1_op0 = tf.nn.in_top_k(softmax0, label_batch0, 1)
    top_5_op0 = tf.nn.in_top_k(softmax0, label_batch0, 5)

    with tf.device('/gpu:1'):
        image_batch1, label_batch1 = batch_queue.dequeue()
        tf.get_variable_scope().reuse_variables()
        logits1 = classification.inference(image_batch1, dropout, is_training)
        #logits = classification.inference(images, dropout, is_training)
        softmax1 = tf.nn.softmax(logits1)
    top_1_op1 = tf.nn.in_top_k(softmax1, label_batch1, 1)
    top_5_op1 = tf.nn.in_top_k(softmax1, label_batch1, 5)

    with tf.device('/gpu:2'):
        image_batch2, label_batch2 = batch_queue.dequeue()
        tf.get_variable_scope().reuse_variables()
        logits2 = classification.inference(image_batch2, dropout, is_training)
        #logits = classification.inference(images, dropout, is_training)
        softmax2 = tf.nn.softmax(logits2)

    top_1_op2 = tf.nn.in_top_k(softmax2, label_batch2, 1)
    top_5_op2 = tf.nn.in_top_k(softmax2, label_batch2, 5)
    """
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
      [images, labels], capacity=2 * helper.N_GPUS)

    total_softmax = []
    total_labels = []
    for i in range(helper.N_GPUS):
        with tf.device('/gpu:%d' % i):
            image_batch, label_batch = batch_queue.dequeue()
            tf.get_variable_scope().reuse_variables()
            logits = classification.inference(image_batch, dropout, is_training)
            #tf.get_variable_scope().reuse_variables()
            softmax = tf.nn.softmax(logits)
            total_softmax.append(softmax)
            total_labels.append(labels)

    stacked_softmax = tf.squeeze(tf.concat([tf.expand_dims(tensor, 0) for tensor in total_softmax], axis=1))
    stacked_labels = tf.squeeze(tf.concat([tf.expand_dims(tensor,0) for tensor in total_labels], axis=1))
    top_1_op = tf.nn.in_top_k(stacked_softmax, stacked_labels, 1)
    top_5_op = tf.nn.in_top_k(stacked_softmax, stacked_labels, 5)


    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    graph_def = tf.get_default_graph().as_graph_def()

    ckpt = tf.train.get_checkpoint_state(SAVE_POINT)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)


        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
        print('No checkpoint file found')
        return

    tf.train.start_queue_runners(sess=sess)
    num_iter = int(math.ceil(50000 / helper.BATCH_SIZE))
    #num_iter = int(math.ceil(10000 / helper.SPLIT_BATCH_SIZE))
    print('starting evaluation on (%s).' % ('validation'))
    count_top_1 = 0.0
    count_top_5 = 0.0
    #total_sample_count = num_iter * helper.SPLIT_BATCH_SIZE
    total_sample_count = num_iter * helper.BATCH_SIZE

    start_time = time.time()
    for step in range(num_iter):
      """
      top_10, top_50, top_11, top_51, top_12, top_52 = sess.run([top_1_op0, top_5_op0, top_1_op1, top_5_op1, top_1_op2, top_5_op2], feed_dict={dropout: 1.0, is_training: False})
      count_top_1 += np.sum(top_10)
      count_top_5 += np.sum(top_50)
      count_top_1 += np.sum(top_11)
      count_top_5 += np.sum(top_51)
      count_top_1 += np.sum(top_12)
      count_top_5 += np.sum(top_52)
      """
      top_1, top_5 = sess.run([top_1_op, top_5_op], feed_dict={dropout: 1.0, is_training: False})
      count_top_1 += np.sum(top_1)
      count_top_5 += np.sum(top_5)
      if step % 20 == 0:
        duration = time.time() - start_time
        sec_per_batch = duration / 20.0
        examples_per_sec = helper.BATCH_SIZE / sec_per_batch
        #examples_per_sec = helper.SPLIT_BATCH_SIZE / sec_per_batch
        print('[%d batches out of %d] (%.1f examples/sec; %.3f'
              'sec/batch)' % (step, num_iter,
                              examples_per_sec, sec_per_batch))
        start_time = time.time()

    # Compute precision @ 1.
    precision_at_1 = count_top_1 / total_sample_count
    recall_at_5 = count_top_5 / total_sample_count
    print('precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
          (precision_at_1, recall_at_5, total_sample_count))
    return precision_at_1, recall_at_5

    # Start the queue runners.
    """
    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                           start=True))

        if train:
            num_iter = int(math.ceil(1281167 / helper.SPLIT_BATCH_SIZE))
            #num_iter = int(math.ceil(50000 / helper.SPLIT_BATCH_SIZE))
            print('starting evaluation on (%s).' % ('training'))
        else:
            #num_iter = int(math.ceil(50000 / helper.SPLIT_BATCH_SIZE))
            num_iter = int(math.ceil(50000 / helper.BATCH_SIZE))
            #num_iter = int(math.ceil(10000 / helper.SPLIT_BATCH_SIZE))
            print('starting evaluation on (%s).' % ('validation'))

        # Counts the number of correct predictions.
        count_top_1 = 0.0
        count_top_5 = 0.0
        #total_sample_count = num_iter * helper.SPLIT_BATCH_SIZE
        total_sample_count = num_iter * helper.BATCH_SIZE
        step = 0


        start_time = time.time()
        while step < num_iter and not coord.should_stop():
          top_1, top_5 = sess.run([top_1_op, top_5_op], feed_dict={dropout: 1.0, is_training: False})
          count_top_1 += np.sum(top_1)
          count_top_5 += np.sum(top_5)
          step += 1
          if step % 20 == 0:
            duration = time.time() - start_time
            sec_per_batch = duration / 20.0
            examples_per_sec = helper.BATCH_SIZE / sec_per_batch
            #examples_per_sec = helper.SPLIT_BATCH_SIZE / sec_per_batch
            print('[%d batches out of %d] (%.1f examples/sec; %.3f'
                  'sec/batch)' % (step, num_iter,
                                  examples_per_sec, sec_per_batch))
            start_time = time.time()

        # Compute precision @ 1.
        precision_at_1 = count_top_1 / total_sample_count
        recall_at_5 = count_top_5 / total_sample_count
        print('precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
              (precision_at_1, recall_at_5, total_sample_count))
        return precision_at_1, recall_at_5


    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
    """

def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Get images and labels for CIFAR-10.
        #dataset = CIFARData(subset='train')
        dataset = ImagenetData(subset='train')
        assert dataset.data_files()

        #test_set = CIFARData(subset='validation')
        test_set = ImagenetData(subset='validation')
        assert test_set.data_files()

        epoch1 = .5 * helper.MAX_EPOCHS
        epoch2 = .75 * helper.MAX_EPOCHS
        step1 = dataset.num_examples_per_epoch() * epoch1 // (helper.BATCH_SIZE)
        step2 = dataset.num_examples_per_epoch() * epoch2 // (helper.BATCH_SIZE)
        print('Reducing learning rate at step '+str(step1)+' and step '+str(step2)+' and ending at '+str(helper.MAX_STEPS))

        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # Learning rate
        lr = .1

        #learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        dropout = tf.placeholder(tf.float32, shape=[], name='dropout')
        is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        boundaries = [step1, step2]
        values = [lr, lr/10, lr/100]

        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values, name=None)

        decayed_lr = tf.train.polynomial_decay(lr, global_step, helper.MAX_STEPS, end_learning_rate=0.0001, power=4.0, cycle=False, name=None)

        # Create an optimizer that performs gradient descent.
        with tf.name_scope('Optimizer'):
            opt = tf.train.MomentumOptimizer(learning_rate=decayed_lr, momentum=0.9, use_nesterov=True)
            #opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)

        tf.summary.scalar('decayed_learning_rate', decayed_lr)
        tf.summary.scalar('learning_rate', learning_rate)



        # Override the number of preprocessing threads to account for the increased
        # number of GPU towers.
        num_preprocess_threads = helper.NUM_THREADS * helper.N_GPUS
        distorted_images, distorted_labels = image_processing.distorted_inputs(dataset, batch_size=helper.SPLIT_BATCH_SIZE, num_preprocess_threads=num_preprocess_threads)

        #images, labels = image_processing.inputs(dataset, batch_size=helper.BATCH_SIZE, num_preprocess_threads=num_preprocess_threads)
        test_images, test_labels = image_processing.inputs(test_set, batch_size=helper.SPLIT_BATCH_SIZE, num_preprocess_threads=num_preprocess_threads)

        input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Split the batch of images and labels for towers.
        #images_splits = tf.split(axis=0, num_or_size_splits=helper.N_GPUS, value=distorted_images)
        #labels_splits = tf.split(axis=0, num_or_size_splits=helper.N_GPUS, value=distorted_labels)

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [distorted_images, distorted_labels], capacity=2 * helper.N_GPUS)

        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(helper.N_GPUS):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (helper.TOWER_NAME, i)) as scope:
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        image_batch, label_batch = batch_queue.dequeue()
                        loss = tower_loss(scope, image_batch, label_batch, dropout=dropout, is_training=is_training)
                        #loss = tower_loss(scope, images_splits[i], labels_splits[i], dropout=dropout, is_training=is_training)

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        tf.get_variable_scope().reuse_variables()

                        grads = opt.compute_gradients(loss)

                        tower_grads.append(grads)


        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summaries for the input processing and global_step.
        summaries.extend(input_summaries)

        # Apply the gradients to adjust the shared variables.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(helper.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            # Group all updates to into a single train op.
            #train_op = apply_gradient_op
            train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Add histograms for trainable variables.
        #for var in tf.trainable_variables():
        #    summaries.append(tf.summary.histogram(var.op.name, var))

        for grad, var in grads:
            summaries.append(tf.summary.histogram(var.op.name, var))
            #summaries.append(tf.summary.histogram(var.op.name + '_gradient', grad))

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        cross_entropy_op = tf.reduce_mean(tf.get_collection('cross_entropies'), name='cross_entropy')

        accuracy_op = tf.reduce_mean(tf.get_collection('accuracy'), name='accuracies')
        summaries.append(tf.summary.scalar('cross_entropy', cross_entropy_op))
        summaries.append(tf.summary.scalar('accuracy', accuracy_op))

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()


        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        if RESTORE == True:
            ckpt = tf.train.get_checkpoint_state(SAVE_POINT)
            saver.restore(sess, ckpt.model_checkpoint_path)

            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/imagenet_train/model.ckpt-0,
            # extract global_step from it.
            restored_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Successfully loaded model from %s at step=%s.' %
                (ckpt.model_checkpoint_path, restored_step))
            step = int(restored_step)
            range_step = range(step,helper.MAX_STEPS)
            tf.get_variable_scope().reuse_variables()
            global_step = tf.get_variable('global_step', trainable=False)
        else:
            range_step = range(helper.MAX_STEPS)


        summary_writer = tf.summary.FileWriter('summary', graph=sess.graph)
        num_params = helper.count_params() / 1e6
        print('Total number of params = %.2fM' % num_params)
        print("training")
        top1_error = [-1.0,-1.0]
        top1_step = 0
        top5_error = [-1.0,-1.0]
        top5_step = 0

        for step in range_step:

            start_time = time.time()
            _, loss_value, cross_entropy_value, accuracy_value = sess.run([train_op, loss, cross_entropy_op, accuracy_op], feed_dict={dropout: 0.8, is_training: True})#, options=run_options, run_metadata=run_metadata)#, learning_rate: lr})
            duration = time.time() - start_time

            if step == step1 or step == step2:
                print('Decreasing Learning Rate')
                lr /= 10

            if step % 10 == 0:
                num_examples_per_step = helper.BATCH_SIZE
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration

                format_str = ('step %d, loss = %.2f, cross entropy = %.2f, accuracy = %.2f, %.3f sec/batch')
                print (format_str % (step, loss_value, cross_entropy_value, accuracy_value, sec_per_batch))

                """
                # Create the Timeline object, and write it to a json
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('timeline.json', 'w') as f:
                    f.write(ctf)
                """

            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict={dropout:0.8, is_training: False})#, learning_rate: lr})
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 5000 == 0 or (step + 1) == helper.MAX_STEPS:
                if step != 0:
                    checkpoint_path = SAVE_POINT + 'model.ckpt'
                    saver.save(sess, checkpoint_path, global_step=step)
                    print('Model saved')

                    #evaluate(distorted_images, distorted_labels, sess, dropout=dropout, is_training=is_training, train=True)
                    top1, top5 = evaluate(test_images, test_labels, sess, dropout=dropout, is_training=is_training, train=False)
                    if top1 > top1_error[0]:
                        top1_error[0] = top1
                        top1_error[1] = top5
                        top1_step = step
                    if top5 > top5_error[1]:
                        top5_error[0] = top1
                        top5_error[1] = top5
                        top5_step = step
                    print("Best top1 model achieved top1: %.4f, top5: %.4f at step %d" %(top1_error[0], top1_error[1], top1_step))
                    print("Best top5 model achieved top1: %.4f, top5: %.4f at step %d" %(top5_error[0], top5_error[1], top5_step))



if __name__ == '__main__':
    train()
