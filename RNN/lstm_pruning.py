#!/opt/conda/bin/python
#coding=utf-8

import tensorflow as tf
# import numpy as np
from lstm_inputs import LSTM_Input

class LSTM_Pruning(object):
    def __init__(self, batch_size=128):
        self.batch_size = batch_size

        self.MOVING_AVERAGE_DECAY = 0.999
        self.NUM_EPOCHS_PER_DECAY = 10.0
        self.LEARNING_RATE_DECAY_FACTOR = 0.75
        self.INITIAL_LEARNING_RATE = 1e-4

        # self.TF_VERSION = tf.__version__
        # print('self.TF_VERSION = ', self.TF_VERSION)

        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8

        self.lstm_input = LSTM_Input(batch_size=self.batch_size)
        self.n_input = 28 # input layer n
        self.n_steps = 28 # length
        self.n_hidden = 128 # hidden layers' features
        self.n_classes = 10 # output num

        self.X = None
        self.istate = None
        self.Y = None

    @staticmethod
    def print_all_variables():
        for var in tf.global_variables():
            print(var)

    def inputs(self, data, batch_size=None):
        return self.lstm_input.inputs(data, batch_size)

    def distoried_inputs(self, batch_size=None):
        return self.lstm_input.distorted_inputs(batch_size)

    def create_placeholder(self):
        self.X = tf.placeholder(tf.float32, [None, 28, 28], name='x-input')
        self.istate = tf.placeholder(tf.float32, [None, 2 * self.n_hidden])
        self.Y = tf.placeholder(tf.float32, [None], name='y-input')

    @staticmethod
    def _get_weights(shape, name='w'):
        return tf.get_variable(name , shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))

    @staticmethod
    def _get_bias(shape, name='b'):
        return tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.constant_initializer(0.1))


    def inference(self, images):
        images = tf.transpose(images, [1, 0, 2])
        images = tf.reshape(images, [-1, self.n_input])
        with tf.variable_scope('inference'):
            w1 = self._get_weights([self.n_input, self.n_hidden], 'w1')
            b1 = self._get_bias([self.n_hidden], 'b1')
            X = tf.matmul(images, w1) + b1

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

            # X = tf.cast(X, tf.int32)
            X = tf.split(X, self.n_steps)
            outputs, states = tf.nn.static_rnn(lstm_cell, X, dtype=tf.float32)

            wo = self._get_weights([self.n_hidden, self.n_classes], 'wo')
            bo = self._get_bias([self.n_classes], 'bo')
            logits = tf.matmul(outputs[-1], wo) + bo
        return logits


    @staticmethod
    def loss(logits, labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('losses', cross_entropy_mean)
        return cross_entropy_mean

    @staticmethod
    def accuracy(logits, labels):
        labels = tf.cast(labels, tf.int64)
        prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy_op = tf.reduce_mean(tf.cast(prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy_op)
        return accuracy_op

    @staticmethod
    def predict(logits):
        return tf.argmax(logits, 1)


    def train(self, loss, global_step):
        num_batches_per_epoch = self.lstm_input.num_examples_per_epoch_for_train / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)

        lr = tf.train.exponential_decay(
            learning_rate=self.INITIAL_LEARNING_RATE,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=self.LEARNING_RATE_DECAY_FACTOR,
            staircase=True)
        tf.summary.scalar('learning rate', lr)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(lr)
            optimizer_op = optimizer.minimize(loss, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([optimizer_op, variable_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op


def main(argv=None):
    pass


if __name__ == '__main__':
    tf.app.run(main, argv=None)

