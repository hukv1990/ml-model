#coding=utf-8

from __future__ import print_function
# from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim

import sys
sys.path.append(r'D:\ml\dataset_inf')

from base_net import BaseNet
from cifar10 import Cifar10

class AlexNet(BaseNet):
    def __init__(self, batch_size, num_classes):
        super(AlexNet, self).__init__(batch_size)

        self.num_examples_per_epoch_for_train = 50000
        self.num_examples_per_epoch_for_test = 10000
        self.num_classes = num_classes

        self.num_epochs_per_decay = 5
        self.initial_learning_rate = 1e-3
        self.learning_rate_decay_ractor = 0.6
        self.moving_avg_decay = 0.99


        self.dataset_train = Cifar10(batch_size,
                                     training=True,
                                     path=r'D:\ml\datasets\cifar-10-batches-bin')
        self.dataset_valid = Cifar10(10000,
                                     training=False,
                                     path=r'D:\ml\datasets\cifar-10-batches-bin')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

    def inputs(self):
        def train_tensor():
            return self.dataset_train.get_tensor()

        def valid_tensor():
            return self.dataset_valid.get_tensor()

        with tf.variable_scope('inputs'):
            images, labels = tf.cond(self.is_training,
                              true_fn=train_tensor,
                              false_fn=valid_tensor)

        return images, labels

    def inference(self, images):
        with tf.variable_scope('alexnet', [images]):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                x = slim.conv2d(images, 64, [3,3], 1, scope='conv1')
                x = slim.max_pool2d(x, [3,3], 2, scope='pool1')

                x = slim.conv2d(x, 128, [5, 5], scope='conv2')
                x = slim.conv2d(x, 128, [5, 5], scope='conv3')
                x = slim.max_pool2d(x, [3, 3], 2, scope='pool2')
                x = slim.conv2d(x, 256, [3, 3], scope='conv4')
                x = slim.conv2d(x, 256, [3, 3], scope='conv5')

                x = slim.max_pool2d(x, [3, 3], 2, scope='pool3')

                x = slim.flatten(x, scope='flatten')
                x = slim.fully_connected(x, 1024)
                x = slim.dropout(x, 0.5,is_training=self.is_training)
                # x = slim.fully_connected(x, 1024)
                # x = slim.dropout(x, 0.5, is_training=self.is_training)
                x = slim.fully_connected(x, self.num_classes, activation_fn=None)
        return x

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
        cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        tf.add_to_collection('losses', cross_entropy_mean)
        losses = tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('losses', losses)
        return losses

    def accuracy(self, logits, labels):
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1)),tf.float32))
        return acc

    def predict(self, logits):
        result = tf.argmax(logits, -1)
        return result

    def evaluation(self, *args, **kwargs):pass