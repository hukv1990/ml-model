# coding=utf-8

from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.append(r'D:\ml\datasets')

import tensorflow as tf
from base_net import BaseNet
from tensorflow.examples.tutorials.mnist import  input_data

class LeNet(BaseNet):
    def __init__(self, batch_size):
        super(LeNet, self).__init__(batch_size)

        self.num_examples_per_epoch_for_train = 60000
        self.num_examples_per_epoch_for_test = 10000
        self.num_epochs_per_decay = 5
        self.initial_learning_rate = 0.01
        self.learning_rate_decay_ractor = 0.6
        self.moving_avg_decay = 0.99

        tf.logging.set_verbosity(tf.logging.ERROR)
        self.dataset = input_data.read_data_sets(r'D:\ml\datasets\mnist', one_hot=True)

    def inputs(self, training=True):
        with tf.variable_scope('inputs'):
            self.images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x-input')
            self.labels = tf.placeholder(tf.float32, shape=[None, 10], name='y-input')
            self.training = tf.placeholder(tf.bool, name='training')
        return self.images, self.labels, self.training

    def _get_layer_variable(self, shape, stddev=0.01):
        w = tf.get_variable('w',
                            shape=shape,
                            dtype=tf.float32,
                            # initializer=tf.truncated_normal_initializer(stddev=stddev))
                            initializer=tf.glorot_normal_initializer(seed=1))
        b = tf.get_variable('b',
                            shape=[shape[-1]],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
        return w, b

    def inference(self, inputs):
        with tf.variable_scope('inference'):
            A0 = tf.reshape(inputs, [-1, 28, 28, 1])
            with tf.variable_scope('conv1'):
                w, b = self._get_layer_variable([5, 5, 1, 32])
                Z1 = tf.nn.bias_add(tf.nn.conv2d(A0, w, strides=[1, 1, 1, 1], padding='SAME'), b)
                A1 = tf.nn.relu(Z1)

            pool1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

            with tf.variable_scope('conv2'):
                w, b = self._get_layer_variable([5, 5, 32, 64])
                Z2 = tf.nn.bias_add(tf.nn.conv2d(pool1, w, strides=[1, 1, 1, 1], padding='SAME'), b)
                A2 = tf.nn.relu(Z2)

            pool2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

            pool_shape = pool2.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            A_fc0 = tf.reshape(pool2, [-1, nodes])

            with tf.variable_scope('fc1'):
                w, b = self._get_layer_variable([nodes, 512])
                Z_fc1 = tf.matmul(A_fc0, w) + b
                A_fc1 = tf.nn.relu(Z_fc1)

            A_fc1_dropout = tf.cond(self.training,
                                    lambda:tf.nn.dropout(A_fc1, 0.75),
                                    lambda:A_fc1)

            # with tf.variable_scope('fc2'):
            #     w, b = _get_layer_variable([512, 512])
            #     Z_fc2 = tf.matmul(A_fc1_dropout, w) + b
            #     A_fc2 = tf.nn.relu(Z_fc2)
            # if is_train:
            #     A_fc2_dropout = tf.nn.dropout(A_fc2, 0.75)
            # else:
            #     A_fc2_dropout = A_fc2

            with tf.variable_scope('fc3'):
                w, b = self._get_layer_variable([512, 10])
                Z_fc3 = tf.matmul(A_fc1_dropout, w) + b
        return Z_fc3

    def train(self, loss, global_step):
        num_batches_per_epoch = self.num_examples_per_epoch_for_train / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay)
        lr = tf.train.exponential_decay(
            learning_rate=self.initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=self.learning_rate_decay_ractor,
            staircase=True)
        tf.summary.scalar('learning_rate', lr)
        # gradient_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
        # gradient_op = tf.train.AdagradOptimizer(lr).minimize(loss, global_step=global_step)
        gradient_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(self.moving_avg_decay, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([gradient_op, variable_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op

    def loss(self, logits, labels):
        cross_entroy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        cross_entroy_mean = tf.reduce_mean(cross_entroy)

        def add_to_train_collection():
            tf.add_to_collection('losses', cross_entroy_mean)
            losses = tf.add_n(tf.get_collection('losses'))

            return losses

        def add_to_test_collection():
            return cross_entroy_mean

        losses = tf.cond(self.training,
                         true_fn=add_to_train_collection,
                         false_fn=add_to_test_collection,
                         name='train_cond')
        tf.summary.scalar('losses', losses)
        return losses

    def accuracy(self, logits, labels):
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1)),tf.float32))


        tf.summary.scalar('acc', acc)
        return acc

    def predict(self, logits):
        result = tf.argmax(logits, -1)
        return result

    def evaluation(self):pass


if __name__ == '__main__':
    net = LeNet(32)
    net.inputs()