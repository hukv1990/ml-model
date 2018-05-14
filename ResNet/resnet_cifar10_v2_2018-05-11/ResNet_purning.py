#coding=utf-8

import tensorflow as tf
from ResNet_input import ResNetInput_Cifar10
import numpy as np


# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------


class ResNetPurning(ResNetInput_Cifar10):
    def __init__(self, training, batch_size=32):
        self.__version__ = 1
        super(ResNetPurning, self).__init__(batch_size=batch_size)
        self.batch_size = batch_size

        # self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
        # self.NUM_EXAMPLES_PER_EPOCH_FOR_DEV = 120

        self.MOVING_AVERAGE_DECAY = 0.99
        self.NUM_EPOCHS_PER_DECAY = 1.0
        self.LEARNING_RATE_DECAY_FACTOR = 0.75
        self.INITIAL_LEARNING_RATE = 1e-3

        self.TF_VERSION = tf.__version__
        print('self.TF_VERSION = ', self.TF_VERSION)

        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8

        # be True only in training
        self.training = training

        self.graph = tf.Graph()

        self.cifar10 = ResNetInput_Cifar10(batch_size)

    @staticmethod
    def create_placeholder():
        x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x-input')
        y = tf.placeholder(tf.int64, [None], name='y-input')
        return x, y


    @staticmethod
    def _weight_variable(shape, name):
        w =  tf.get_variable(name,
                             shape=shape,
                             dtype=tf.float32,
                             # initializer=tf.contrib.layers.xavier_initializer_conv2d()
                             # initializer=tf.contrib.layers.variance_scaling_initializer()
                             # initializer=tf.truncated_normal_initializer(stddev=5e-2)
                             # initializer=tf.glorot_normal_initializer()
                             initializer=tf.glorot_uniform_initializer()
                             )
        return w

    @staticmethod
    def _conv2d(inputs, num_filters, kernel_size=3, strides=1, padding='SAME', name=None):
        _input_layers = inputs.get_shape().as_list()[-1]
        # print('_iput_layers = ', _input_layers)
        w = ResNetPurning._weight_variable(
            name = ''.join(('w_', name)),
            shape=(kernel_size, kernel_size, _input_layers, num_filters)
        )
        x = tf.nn.conv2d(inputs, w,
                         strides=[1, strides, strides, 1],
                         padding=padding,
                         name=name)
        return x

    def _batch_norm(self, inputs, axis=-1):
        return tf.layers.batch_normalization(inputs, axis=-1, training=self.training)


    def resnet_layer(self, inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation=tf.nn.relu,
                     batch_normalization=True,
                     conv_first=True,
                     name=None):
        """2D Convolution-Batch Normalization-Activation stack builder
            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    activation-bn-conv (False)
            # Returns
                x (tensor): tensor as input to the next layer
        """


        conv = lambda x : \
            self._conv2d(x, num_filters, kernel_size, strides, padding='SAME', name=name)

        x = inputs

        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = self._batch_norm(x)
            if activation is not None:
                x = activation(x)
        else:
            if batch_normalization:
                x = self._batch_norm(x)
            if activation is not None:
                x = activation(x)
            x = conv(x)

        return x

    def resnet_v1(self, inputs, depth):
        logits = None
        with tf.variable_scope('inference'):
            if (depth - 2) % 6 != 0:
                raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

            num_filters = 16
            num_res_blocks = int((depth - 2) / 6)

            x = self.resnet_layer(inputs=inputs, name='conv0')

            for stack in range(3):
                with tf.variable_scope('stage_{0}'.format(stack)):
                    for res_block in range(num_res_blocks):
                        strides = 1
                        if stack > 0 and res_block == 0:
                            strides = 2
                        with tf.variable_scope('conv_{0}'.format(res_block)):
                            y = self.resnet_layer(inputs = x,
                                                  num_filters=num_filters,
                                                  strides=strides,
                                                  name='conv1')
                            # print('s1', y.get_shape())
                            y = self.resnet_layer(inputs=y,
                                                  num_filters=num_filters,
                                                  activation=None,
                                                  name='conv2')
                            # print('s2', y.get_shape())
                            if stack > 0 and res_block == 0:
                                x = self.resnet_layer(inputs=x,
                                                      num_filters=num_filters,
                                                      kernel_size=1,
                                                      strides=strides,
                                                      activation=None,
                                                      batch_normalization=False,
                                                      name='conv_res')
                                # print('s3', x.get_shape())
                            x = tf.add(x, y)
                            x = tf.nn.relu(x)
                    num_filters *= 2

            x = tf.nn.avg_pool(x, [1,8,8,1], [1,1,1,1], padding='VALID')

            y = tf.contrib.layers.flatten(x)
            logits = tf.layers.dense(y, units=10,
                                     activation=tf.nn.softmax,
                                     kernel_initializer=tf.glorot_uniform_initializer())
        return logits

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

        lr = tf.train.exponential_decay(
            learning_rate=self.INITIAL_LEARNING_RATE,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=self.LEARNING_RATE_DECAY_FACTOR,
            staircase=True)
        tf.summary.scalar('learning rate', lr)
        lr = tf.maximum(lr, 1e-6)

        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # optimizer = tf.train.GradientDescentOptimizer(lr)
                optimizer = tf.train.AdamOptimizer(lr)
                # optimizer = tf.train.RMSPropOptimizer(lr)
                optimizer_op = optimizer.minimize(loss, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([optimizer_op, variable_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op


def main(argv=None):
    resnet = ResNetPurning()
    resnet.resnet_v1()

if __name__ == '__main__':
    tf.app.run(main=main, argv=None)





