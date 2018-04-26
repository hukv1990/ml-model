#!/opt/conda/bin/python
#coding=utf-8

import tensorflow as tf
import numpy as np
from resnets_utils import random_mini_batches
import h5py
import os


class ResNetPruning(object):
    def __init__(self, data_dir='./resnet50_dataset', batch_size=32):
        self.DATA_DIR = data_dir
        self.BATCH_SIZE = batch_size
        self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
        self.NUM_EXAMPLES_PER_EPOCH_FOR_DEV = 120

        self.MOVING_AVERAGE_DECAY = 0.999
        self.NUM_EPOCHS_PER_DECAY = 5.0
        self.LEARNING_RATE_DECAY_FACTOR = 0.75
        self.INITIAL_LEARNING_RATE = 1e-4

        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8

    def inputs(self, eval_data=True):
        _set_name = 'test' if eval_data else 'train'
        _dataset = h5py.File(os.path.join(self.DATA_DIR, '{0}_signs.h5'.format(_set_name)), "r")
        _set_x_orig = np.array(_dataset["{0}_set_x".format(_set_name)][:]) # your train set features
        _set_y_orig = np.array(_dataset["{0}_set_y".format(_set_name)][:]) # your train set labels
        _set_y_orig = _set_y_orig.reshape(_set_y_orig.shape[0], -1)
        return _set_x_orig / 255.0, _set_y_orig

    def destoried_inputs(self):
        train_dataset = h5py.File(os.path.join(self.DATA_DIR, 'train_signs.h5'), "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
        train_set_y_orig = train_set_y_orig.reshape(train_set_y_orig.shape[0], -1)
        return random_mini_batches(train_set_x_orig / 255.0, train_set_y_orig, self.BATCH_SIZE, 0)

    def create_placeholder(self):
        X = tf.placeholder(tf.float32, [None, 64, 64, 3], name='x-input')
        Y = tf.placeholder(tf.int64, [None], name='y-input')
        return X, Y

    def _conv2d(self, input):
        pass

    def _weight_variable2(self, shape, name=None):
        initial = tf.truncated_normal(shape, stddev=1e-2)
        return tf.Variable(initial, name=name)

    def _weight_variable(self, shape, name):
        w = tf.get_variable(name,
                            shape=shape,
                            dtype=tf.float32,
                            # initializer=tf.contrib.layers.xavier_initializer_conv2d()
                            # initializer=tf.contrib.layers.variance_scaling_initializer()
                            # initializer=tf.truncated_normal_initializer(stddev=5e-2)
                            # initializer=tf.glorot_normal_initializer()
                            initializer=tf.glorot_uniform_initializer()
                            )
        return w

    def _bias_variable(self, shape, value=0.):
        return tf.Variable(tf.constant(value, tf.float32, shape))

    def _conv_block(self, X_input, kernel_size, in_filter,
        out_filters, stage_block, training, stride=2):
        '''
        axis : [None, w, h, channels]在对应的channel层上做批次归一化, 也可以是-1
        stride 为什么要改变，而且默认为2
        '''
        block_name = 'res' + stage_block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            # first
            W1 = self._weight_variable([1, 1, in_filter, f1], 'W1')
            X = tf.nn.conv2d(X_input, W1, strides=[1, stride, stride, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            # second
            W2 = self._weight_variable([kernel_size,kernel_size, f1, f2], 'W2')
            X = tf.nn.conv2d(X, W2, strides=[1,1,1,1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #third
            W3 = self._weight_variable([1, 1, f2, f3], 'W3')
            X = tf.nn.conv2d(X, W3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            # short cut
            W_shortcut = self._weight_variable([1, 1, in_filter, f3], 'W_shortcut')
            X_shortcut = tf.nn.conv2d(X_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

            # final layer
            X = tf.add(X_shortcut, X)
            X = tf.nn.relu(X)

        return X

    def _identity_block(self, X_input, kernel_size, in_filter, out_filters, stage_block, training):
        block_name = 'res' + stage_block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            # first
            W1 = self._weight_variable([1, 1, in_filter, f1], 'W1')
            X = tf.nn.conv2d(X_input, W1, strides=[1,1,1,1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            # second
            W2 = self._weight_variable([kernel_size, kernel_size, f1, f2], 'W2')
            X= tf.nn.conv2d(X, W2, strides=[1,1,1,1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            # third
            W3 = self._weight_variable([1, 1, f2, f3], 'W3')
            X = tf.nn.conv2d(X, W3, strides=[1,1,1,1], padding='VALID')

            # final layer
            X = tf.add(X_shortcut, X)
            X = tf.nn.relu(X)
        return X

    def inference(self, X_input, training):
        X = tf.pad(X_input, tf.constant([[0,0], [3,3], [3,3], [0,0]]), 'CONSTANT')
        with tf.variable_scope('inference'):
            # training = tf.placeholder(tf.bool, name='training')
            # stage 1
            with tf.variable_scope('stage1'):
                W = self._weight_variable([7,7,3,64], 'W')
                X = tf.nn.conv2d(X, W, strides=[1,2,2,1], padding='VALID')
                X = tf.layers.batch_normalization(X, axis=3, training=training)
                X = tf.nn.relu(X)
                X = tf.nn.max_pool(X, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

            with tf.variable_scope('stage2'):
                X = self._conv_block(X, 3, 64, [64,64,256], '2a', training, stride=1)
                X = self._identity_block(X, 3, 256, [64, 64, 256], '2b', training)
                X = self._identity_block(X, 3, 256, [64, 64, 256], '2c', training)

            with tf.variable_scope('stage3'):
                X = self._conv_block(X, 3, 256, [128,128,512], '3a', training)
                X = self._identity_block(X, 3, 512, [128,128,512], '3b', training)
                X = self._identity_block(X, 3, 512, [128, 128, 512], '3c', training)
                X = self._identity_block(X, 3, 512, [128, 128, 512], '3d', training)

            with tf.variable_scope('stage4'):
                X = self._conv_block(X, 3, 512, [256,256,1024], '4a', training)
                X = self._identity_block(X, 3, 1024, [256,256,1024], '4b', training)
                X = self._identity_block(X, 3, 1024, [256, 256, 1024], '4c', training)
                X = self._identity_block(X, 3, 1024, [256, 256, 1024], '4d', training)
                X = self._identity_block(X, 3, 1024, [256, 256, 1024], '4e', training)
                X = self._identity_block(X, 3, 1024, [256, 256, 1024], '4f', training)

            with tf.variable_scope('stage5'):
                X = self._conv_block(X, 3, 1024, [512,512,2048], '5a', training)
                X = self._identity_block(X, 3, 2048, [512,512,2048], '5b', training)
                X = self._identity_block(X, 3, 2048, [512, 512, 2048], '5c', training)

                X = tf.nn.avg_pool(X, [1,2,2,1], strides=[1,1,1,1], padding='VALID')

            flatten = tf.layers.flatten(X)
            logits = tf.layers.dense(flatten, units=6)

        return tf.reshape(logits, [-1, 6])

    def loss(self, logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', cross_entropy_mean)
        losses = tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('losses', losses)
        return losses

    def accuracy(self, logits, labels):
        prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy_op = tf.reduce_mean(tf.cast(prediction, tf.float32))
        return accuracy_op

    def train(self, total_loss, global_step):
        num_batches_per_epoch = self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / self.BATCH_SIZE
        decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)

        lr = tf.train.exponential_decay(
        learning_rate=self.INITIAL_LEARNING_RATE,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=self.LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
        tf.summary.scalar('learning rate', lr)

        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # optimizer = tf.train.GradientDescentOptimizer(lr)
                optimizer = tf.train.AdamOptimizer(lr)
                # optimizer = tf.train.RMSPropOptimizer(lr)
                optimizer_op = optimizer.minimize(total_loss, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([optimizer_op, variable_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op

if __name__ == '__main__':
    # X, y = inputs(True)
    # print(X.shape, y.shape)
    # mini_batches = destoried_inputs()
    # mini_batch = mini_batches[0]
    # X, y = mini_batch
    # print(X.shape, y.shape)

    resnet = ResNetPruning()
    X, y = resnet.inputs()
    pass
