import tensorflow as tf
from glob import glob
from Cluster import Cluster

from sklearn.model_selection import train_test_split
import numpy as np

class Classifier(object):
    def __init__(self, base_path, batch_size=12, epochs=100):
        self.base_path = base_path

        self.clouds = glob(base_path + "*")
        self.clouds.sort()
        temp_clusters = []
        for cloud in self.clouds:
            temp_clusters.append(Cluster(cloud))

        # Avoid features with nans
        self.clusters = []
        max_length = -10000.
        for cluster in temp_clusters:
            flag = True
            for feature in cluster.features:
                if np.isnan(feature):
                    flag = False
            if flag:
                self.clusters.append(cluster)
                if cluster.length > max_length:
                    max_length = cluster.length
        print(max_length)

        z_comparison = 0.0
        care_count = 0
        max_z = -10000000.
        for cluster in self.clusters:
            if cluster.label != 0:
                if cluster.z > max_z:
                    max_z = cluster.z
                care_count = care_count + 1
        print(max_z)

        self.train_clusters, self.test_clusters = train_test_split(self.clusters, test_size=0.25, random_state=42)

        self.num_gpus = 3
        self.num_classes = 4
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight = .8
        self.length_weight = (1. - self.class_weight)/2
        self.z_weight = (1. - self.class_weight)/2
        self.iterations = 50000

        self.optimizer = tf.train.AdamOptimizer()
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.log_device_placement = False
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def build_vars(self, input_shape, output_shape, wd=.0001):
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

    def composite_function(self, input, name, output_shape, relu=True):
        input_shape = input.get_shape().as_list()[1]

        with tf.variable_scope(name) as in_scope:
            matrix, bias = self.build_vars(input_shape, output_shape)

            output = tf.matmul(input, matrix, name='matmul') + bias

            if relu:
                output = tf.nn.relu(output, name='act')
            else:
                output = tf.nn.sigmoid(output, name='act')

        return output

    def graph(self, input):
        layer1 = self.composite_function(input, 'layer1', output_shape=150)

        class_output = self.composite_function(layer1,  'class_output', output_shape=self.num_classes)

        length_output = self.composite_function(layer1,  'length_output', output_shape=1, relu=False)

        z_output = self.composite_function(layer1,  'z_output', output_shape=1, relu=False)

        return class_output, length_output, z_output

    def compute_loss(self, predicted_class, predicted_length, predicted_z, label, length, z, scope):
        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predicted_class, labels=label)
        class_loss = tf.reduce_mean(class_loss)

        predicted_length = tf.squeeze(predicted_length)
        length_loss = tf.losses.mean_squared_error(labels = length, predictions=predicted_length)

        predicted_z = tf.squeeze(predicted_z)
        z_loss = tf.losses.mean_squared_error(labels = z, predictions=predicted_z)

        var_coll = tf.get_collection('losses', scope)
        var_loss = tf.add_n(var_coll)

        loss = (self.class_weight * class_loss) + (self.length_weight * length_loss) + (self.z_weight * z_loss) + var_loss

        return loss

    def average_gradients(self, gradients):
        average_grads = []
        for grad_vars in zip(*gradients):
            grads = []
            for grad, var in grad_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(grad, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            temp_grad = tf.concat(axis=0, values=grads)
            temp_grad = tf.reduce_mean(temp_grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


    def compute_gradients(self, iterator):
        d_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.num_gpus):
                with tf.device('/gpu:'+str(i)):
                    with tf.name_scope('%s_%d' % ('Device', i)) as scope:
                        next_feature, next_label, next_length, next_z = iterator.get_next()
                        predicted_class, predicted_length, predicted_z = self.graph(next_feature)
                        tf.get_variable_scope().reuse_variables()
                        loss = self.compute_loss(predicted_class, predicted_length, predicted_z, next_label, next_length, next_z, scope)
                        grad = self.optimizer.compute_gradients(loss)

                        d_grads.append(grad)

        avg_grads = self.average_gradients(d_grads)

        return avg_grads, loss

    def build_input_queue(self, clusters):
        features = []
        labels = []
        lengths = []
        zs = []
        for cluster in clusters:
            features.append(cluster.features)
            labels.append(cluster.label)
            lengths.append(cluster.length)
            zs.append(cluster.z)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels, lengths, zs))
        dataset = dataset.repeat(None)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()

        return iterator

    def test_accuracy(self, it, iterator):
        class_acc = 0.
        length_acc = 0.
        z_acc = 0.
        N = int(len(self.test_clusters)/self.batch_size)

        tf.get_variable_scope().reuse_variables()
        next_feature, next_label, next_length, next_z = iterator.get_next()
        predicted_class, predicted_length, predicted_z = self.graph(next_feature)

        top_pred = tf.argmax(predicted_class, axis=1)
        top_length = tf.squeeze(predicted_length)
        top_z = tf.squeeze(predicted_z)

        correct_prediction = tf.equal(top_pred, tf.cast(next_label, tf.int64))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        length_op = tf.losses.absolute_difference(labels = next_length, predictions = top_length)

        z_op = tf.losses.absolute_difference(labels = next_z, predictions = top_z)

        for i in range(N):
            acc, l, z = self.sess.run([accuracy_operation, length_op, z_op])
            class_acc += (acc * self.batch_size)
            length_acc += (l * self.batch_size)
            z_acc += (z * self.batch_size)

        class_acc /= len(self.test_clusters)
        length_acc /= len(self.test_clusters)
        z_acc /= len(self.test_clusters)
        length_acc *= self.clusters[0].MAX_LENGTH
        z_acc *= self.clusters[0].MAX_Z
        print("Iteration: %d, Class acc = %.5f, Length diff = %.5f, Z diff = %.5f" % (it, class_acc, length_acc, z_acc))

        return class_acc

    def train(self):
        train_iterator = self.build_input_queue(self.train_clusters)
        test_iterator = self.build_input_queue(self.test_clusters)

        grads, loss = self.compute_gradients(train_iterator)

        apply_gradient_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)

        self.saver = tf.train.Saver()

        self.init = tf.global_variables_initializer()

        self.sess.run(self.init)

        for i in range(self.iterations):
            _ = self.sess.run([apply_gradient_op])

            if i % 10000 == 0:
                self.test_accuracy(i, test_iterator)
        test_acc = self.test_accuracy(i, test_iterator)
        train_acc = self.test_accuracy(i, train_iterator)

        return test_acc

    def save(self):
        tensors = tf.trainable_variables()
        names_and_vals = []
        for tensor in tensors:
            tensor_name = tensor.name[:-2]
            tensor_value = tensor.eval(session=self.sess)
            temp_name_and_val = [tensor_name, tensor_value]
            names_and_vals.append(temp_name_and_val)

        for name_and_val in names_and_vals:
            name = name_and_val[0] + '.txt'
            name = name.replace('/', '_')
            val = name_and_val[1]
            np.savetxt(name, val)

        self.saver.save(self.sess, './model', global_step=self.global_step)
