#!/opt/conda/bin/python
#coding=utf-8

import tensorflow as tf
import numpy as np
import time
import six
from ResNet_input import ResNetInput_Cifar10


class ResNetPurning(object):
    def __init__(self, batch_size=32, training=True):
        self.num_class = 10
        self.batch_size = batch_size

        self.MOVING_AVERAGE_DECAY = 0.999
        self.NUM_EPOCHS_PER_DECAY = 5.0
        self.LEARNING_RATE_DECAY_FACTOR = 0.6
        self.INITIAL_LEARNING_RATE = 1e-1

        self.TF_VERSION = tf.__version__
        print('self.TF_VERSION = ', self.TF_VERSION)

        self.config = tf.ConfigProto(
            device_count={"CPU": 8}, # limit to num_cpu_core CPU usage
            inter_op_parallelism_threads = 2,
            intra_op_parallelism_threads = 10)

        # be True only in training
        self.training = training

        self.cifar10 = ResNetInput_Cifar10(batch_size)
        self.initializer = tf.contrib.layers.variance_scaling_initializer()

    # @staticmethod
    # def _weight_variable(shape, name, kernel_initializer=None, kernel_regularizer=None):
    #     w = tf.get_variable(name,
    #                         shape=shape,
    #                         dtype=tf.float32,
    #                         # initializer=tf.contrib.layers.xavier_initializer_conv2d()
    #                         initializer=tf.contrib.layers.variance_scaling_initializer()
    #                         # initializer=tf.truncated_normal_initializer(stddev=5e-2)
    #                         # initializer=tf.glorot_normal_initializer()
    #                         # initializer=tf.glorot_uniform_initializer()
    #     )
    #
    #     return w

    def create_placeholder(self):
        X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x-input')
        Y = tf.placeholder(tf.int32, [None], name='y-input')
        return X, Y


    def _conv2d(self, inputs ,filters, kernel_size, strides, padding,
                kernel_initializer=None,
                kernel_regularizer=None):

        if kernel_initializer is None:
            kernel_initializer = self.initializer

        _inputs_layers = inputs.get_shape().as_list()[-1]

        filter_shape = [kernel_size[0], kernel_size[1], _inputs_layers, filters]
        w = tf.get_variable('w',
                            shape=filter_shape,
                            initializer=kernel_initializer)
        b = tf.get_variable('b', [filters], initializer=tf.constant_initializer(0.))
        x = tf.nn.conv2d(inputs, w,
                         strides=[1, strides[0], strides[1], 1],
                         padding=padding)
        return tf.nn.bias_add(x, b)


    def _bn_relu(self, inputs):
        norm = tf.layers.batch_normalization(inputs, axis=-1, training=self.training)
        return tf.nn.relu(norm)

    def _conv_bn_relu(self, filters, kernel_size,
                      strides=(1,1), padding='SAME',
                      kernel_initializer=None, kernel_regularizer=None):
        def f(inputs):
            conv = self._conv2d(inputs=inputs,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)
            return self._bn_relu(conv)
        return f

    def _bn_relu_conv(self, filters, kernel_size,
                      strides=(1,1),
                      padding='SAME',
                      kernel_initializer=None,
                      kernel_regularizer=None):
        def f(inputs):
            activation = self._bn_relu(inputs)
            return self._conv2d(inputs=activation,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)
        return f

    def _short_cut(self, inputs, residual):
        inputs_shape = inputs.get_shape().as_list()
        residual_shape = residual.get_shape().as_list()
        stride_width = int(round(inputs_shape[1] / residual_shape[1]))
        stride_height = int(round(inputs_shape[2] / residual_shape[2]))
        equal_channels = inputs_shape[3] == residual_shape[3]

        shortcut = inputs
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = self._conv2d(inputs=inputs,
                                    filters=residual_shape[3],
                                    kernel_size=[1,1],
                                    strides=[stride_width, stride_height],
                                    padding='VALID')
        return tf.add(shortcut, residual)

    def _residual_block(self, block_function, filters, repetitions, is_first_layer=False):
        def f(inputs):
            for i in range(repetitions):
                init_strides = (1,1)
                if i == 0 and not is_first_layer:
                    init_strides = (2,2)
                inputs = block_function(filters=filters, init_strides=init_strides,
                                       is_first_block_of_first_layer=(is_first_layer and i == 0))(inputs)
                return inputs
        return f

    def basic_block(self, filters, init_strides=(1,1), is_first_block_of_first_layer=False):

        def f(inputs):
            with tf.variable_scope('conv0'):
                if is_first_block_of_first_layer:
                    conv1 = self._conv2d(inputs=inputs,
                                         kernel_size=(3,3),
                                         filters=filters,
                                         strides=init_strides,
                                         padding='SAME')
                else:
                    conv1 = self._bn_relu_conv(filters=filters,
                                               kernel_size=(3,3),
                                               strides=init_strides)(inputs)
            with tf.variable_scope('conv1'):
                residual = self._bn_relu_conv(filters=filters, kernel_size=(3,3),
                                          strides=init_strides)(conv1)
            with tf.variable_scope('conv_shortcut'):
                shortcut = self._short_cut(inputs=inputs, residual=residual)
            return shortcut
        return f

    def bottleneck(self, filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
        """Bottleneck architecture for > 34 layer resnet.
        Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

        Returns:
            A final conv layer of filters * 4
        """

        def f(inputs):
            with tf.variable_scope('conv0'):
                if is_first_block_of_first_layer:
                    # don't repeat bn->relu since we just did bn->relu->maxpool
                    conv_1_1 = self._conv2d(inputs, filters=filters, kernel_size=(1, 1),
                                            strides=init_strides,
                                            padding="same")
                else:
                    conv_1_1 = self._bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                             strides=init_strides)(inputs)
            with tf.variable_scope('conv1'):
                conv_3_3 = self._bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)

            with tf.variable_scope('conv2'):
                residual = self._bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)

            with tf.variable_scope('conv_shortcut'):
                shortcut = self._short_cut(inputs, residual)
            return shortcut

        return f

    def _get_block(self, identifier):
        if isinstance(identifier, six.string_types):
            res = globals().get(identifier)
            if not res:
                raise ValueError('Invalid {}'.format(identifier))
            return res
        return identifier

    @staticmethod
    def MaxPooling2D(pool_size, strides, padding):
        ksize=[1, pool_size[0], pool_size[1], 1]
        strides = [1, strides[0], strides[1], 1]
        def f(inputs):
            ret = tf.nn.max_pool(inputs, ksize, strides, padding)
            return ret
        return f

    @staticmethod
    def AveragePooling2D(pool_size, strides, padding="SAME"):
        ksize = [1, pool_size[0], pool_size[1], 1]
        strides = [1, strides[0], strides[1], 1]
        def f(inputs):
            ret = tf.nn.avg_pool(inputs, ksize, strides, padding)
            return ret
        return f

    @staticmethod
    def int_shape(inputs):
        return inputs.get_shape().as_list()

    def build(self, inputs, block_fn, repetitions):
        # if len(inputs_shape) != 3:
        #     raise  Exception("inputs shape should be a tuple (rows, cols, channels).")

        with tf.variable_scope('inference'):

            block_fn = self._get_block(block_fn)

            with tf.variable_scope('stage_0'):
                conv1 = self._conv_bn_relu(filters=64, kernel_size=(7,7), strides=(2,2))(inputs)
                pool1 = self.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='SAME')(conv1)

            block = pool1
            filters = 64
            for i, r in enumerate(repetitions):
                with tf.variable_scope('stage_{0}'.format(i+1)):
                    block = self._residual_block(block_fn,
                                                 filters=filters,
                                                 repetitions=r,
                                                 is_first_layer=(i==0))(block)
                filters *= 2

            # Last activation
            block = self._bn_relu(block)

            block_shape = self.int_shape(block)
            pool2 = self.AveragePooling2D(pool_size=(block_shape[1], block_shape[2]),
                                          strides=(1,1))(block)
            flatten1 = tf.contrib.layers.flatten(pool2)
            dense = tf.layers.dense(flatten1, units=self.num_class,
                                         activation=tf.nn.softmax,
                                         kernel_initializer=self.initializer)
        return dense

    def build_resnet_18(self, inputs):
        return self.build(inputs, self.basic_block, [2,2,2,2])

    def build_resnet_34(self, input_shape):
        return self.build(input_shape, self.basic_block, [3, 4, 6, 3])

    def build_resnet_50(self, input_shape):
        return self.build(input_shape, self.basic_block, [3, 4, 6, 3])

    def build_resnet_101(self, input_shape):
        return self.build(input_shape, self.basic_block, [3, 4, 23, 3])


    def build_resnet_152(self, input_shape):
        return self.build(input_shape, self.basic_block, [3, 8, 36, 3])

    @staticmethod
    def calc_loss(logits, labels):
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
        num_batches_per_epoch = self.cifar10.num_examples_per_epoch_for_train / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)

        # lr = tf.train.exponential_decay(
        #     learning_rate=self.INITIAL_LEARNING_RATE,
        #     global_step=global_step,
        #     decay_steps=decay_steps,
        #     decay_rate=self.LEARNING_RATE_DECAY_FACTOR,
        #     staircase=True)
        lr = tf.constant(self.INITIAL_LEARNING_RATE)
        tf.summary.scalar('learning rate', lr)
        # lr = tf.maximum(lr, 1e-6)


        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.GradientDescentOptimizer(lr)
                # optimizer = tf.train.AdamOptimizer(lr)
                # optimizer = tf.train.RMSPropOptimizer(lr)
                optimizer_op = optimizer.minimize(loss, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([optimizer_op, variable_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op

def main(argv=None):
    resnet = ResNetPurning()
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    logits = resnet.build_resnet_18(X)



if __name__ == '__main__':
    tf.app.run(main=main, argv=None)