#coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cifar10 import cifar10_input

DATA_DIR = './cifar-10-batches-bin'
BATCH_SIZE = 128
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 5.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001  # Initial learning rate.

def create_placeholder():
    X = tf.placeholder(tf.float32, [None, 24, 24, 3])
    y = tf.placeholder(tf.int32, [None, ])
    return X, y

def inputs(eval_data=True, data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    return cifar10_input.inputs(eval_data, data_dir, batch_size)

def destoried_inputs(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    return cifar10_input.distorted_inputs(data_dir, batch_size)


def _weight_variable(shape):
    return tf.get_variable('w',
                           shape=shape,
                           dtype=tf.float32,
                           initializer=tf.glorot_normal_initializer())


def inference(X, is_train=False):
    with tf.variable_scope('conv1'):
        w = _weight_variable([5, 5, 3, 64])
        b = tf.Variable(tf.constant(0.0, tf.float32, [64]))
        Z1 = tf.nn.bias_add(tf.nn.conv2d(X, w, [1,1,1,1], padding='SAME'), b)
        A1 = tf.nn.relu(Z1)

    pool1 = tf.nn.max_pool(
        A1,
        ksize=[1,3,3,1],
        strides=[1,2,2,1],
        padding='SAME',
        name='pool1')

    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2'):
        w = _weight_variable([5,5,64,64])
        b = tf.Variable(tf.constant(0.1, tf.float32, [64]))
        Z2 = tf.nn.bias_add(tf.nn.conv2d(norm1, w, strides=[1,1,1,1], padding='SAME'), b)
        A2 = tf.nn.relu(Z2)
    norm2 = tf.nn.lrn(A2, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(
        norm2,
        ksize=[1,3,3,1],
        strides=[1,2,2,1],
        padding='SAME',
        name='pool2')

    _shape = pool2.get_shape().as_list()
    nodes = _shape[1] * _shape[2] * _shape[3]
    A_fc0 = tf.reshape(pool2, [-1, nodes])
    with tf.variable_scope('fc1'):
        w = _weight_variable([nodes, 384])
        b = tf.Variable(tf.constant(0.1, tf.float32, [384]))
        Z_fc1 = tf.matmul(A_fc0, w) + b
        A_fc1 = tf.nn.relu(Z_fc1)

    A_fc1_dropout = tf.nn.dropout(A_fc1, 0.75) if is_train else A_fc1

    with tf.variable_scope('fc2'):
        w = _weight_variable([384, 192])
        b = tf.Variable(tf.constant(0.1, tf.float32, [192]))
        Z_fc2 = tf.matmul(A_fc1_dropout, w) + b
        A_fc2 = tf.nn.relu(Z_fc2)

    A_fc2_dropout = tf.nn.dropout(A_fc2, 0.75) if is_train else A_fc2

    with tf.variable_scope('fc3'):
        w = _weight_variable([192, 10])
        b = tf.Variable(tf.constant(0.0, tf.float32, [10]))
        Z_fc3 = tf.matmul(A_fc2_dropout, w) + b
    return Z_fc3



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
    optimizer_op = optimizer.minimize(total_loss, global_step=global_step)

    variable_averages =  tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([optimizer_op, variable_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op


if __name__ == '__main__':
    pass
