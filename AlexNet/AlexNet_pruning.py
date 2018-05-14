#!/opt/conda/bin/python
#coding=utf-8

import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from cifar10 import cifar10_input

DATA_DIR = './cifar-10-batches-bin'
BATCH_SIZE = 64
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_DEV = 10000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 5.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.75  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0005  # Initial learning rate.

def create_placeholder():
    X = tf.placeholder(tf.float32, [None, 24, 24, 3])
    y = tf.placeholder(tf.int32, [None])
    return X, y

def inputs(eval_data=True, data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    return cifar10_input.inputs(eval_data, data_dir, batch_size)

def destoried_inputs(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    return cifar10_input.distorted_inputs(data_dir, batch_size)


def _weight_variable(shape):
    return tf.get_variable('w',
                           shape=shape,
                           dtype=tf.float32,
                           # initializer=tf.contrib.layers.xavier_initializer_conv2d()
                           initializer=tf.contrib.layers.variance_scaling_initializer()
                           # initializer=tf.truncated_normal_initializer(stddev=5e-2)
                           # initializer=tf.glorot_normal_initializer()
                           # initializer=tf.glorot_uniform_initializer()
                           )

def _conv2d(shape, padding='SAME', name=None):
    def _func(input):
        with tf.variable_scope(name):
            w = _weight_variable(shape)
            b = tf.Variable(tf.constant(0.0, tf.float32, [shape[-1]]))
            Zx = tf.nn.bias_add(tf.nn.conv2d(input, w, [1,1,1,1], padding=padding), b)
            Ax = tf.nn.relu(Zx)
        return Ax
    return _func

def _max_pool(ksize, strides, padding='SAME', name = None):
    def _func(input):
        return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding, name=name)
    return _func


def inference(X, is_train=False):
    # 24x24
    A1 = _conv2d(shape=[3, 3, 3, 64], padding='SAME', name='conv1')(X)
    A2 = _conv2d(shape=[3, 3, 64, 64], padding='SAME', name='conv2')(A1)
    pool1 = _max_pool(ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')(A2)

    # 12x12
    A3 = _conv2d([3, 3, 64, 128], 'SAME', 'conv3')(pool1)
    A4 = _conv2d([3, 3, 128, 128], 'SAME', 'conv4')(A3)
    pool2 = _max_pool([1, 3, 3, 1], [1, 2, 2, 1], 'SAME', 'pool2')(A4)

    # 6x6
    A5 = _conv2d([3, 3, 128, 256], 'SAME', 'conv5')(pool2)
    A6 = _conv2d([3, 3, 256, 256], 'SAME', 'conv6')(A5)
    pool3 = _max_pool([1, 3, 3, 1], [1, 2, 2, 1], 'SAME', 'pool3')(A6)

    # 3x3
    _shape = pool3.get_shape().as_list()
    nodes = _shape[1] * _shape[2] * _shape[3]
    A_fc0 = tf.reshape(pool3, [-1, nodes])

    A_fc0_dropout = tf.nn.dropout(A_fc0, 0.5)
    with tf.variable_scope('fc1'):
        w = _weight_variable([nodes, 10])
        b = tf.Variable(tf.constant(0., tf.float32, [10]))
        Z_fc1 = tf.matmul(A_fc0_dropout, w) + b
    return Z_fc1


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    losses = tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('losses', losses)
    return losses



def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(
        learning_rate=INITIAL_LEARNING_RATE,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    tf.summary.scalar('learning rate', lr)


    # optimizer = tf.train.GradientDescentOptimizer(lr)
    optimizer = tf.train.AdamOptimizer(lr)
    # optimizer = tf.train.RMSPropOptimizer(lr)
    optimizer_op = optimizer.minimize(total_loss, global_step=global_step)

    variable_averages =  tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([optimizer_op, variable_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op


if __name__ == '__main__':
    pass
