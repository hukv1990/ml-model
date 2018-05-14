#!/opt/conda/bin/python
#coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ResNet_input import ResNetInput_Cifar10

from ResNet_input import ResNetInput_HandSigns


class ResNetPruning(object):
    def __init__(self, batch_size=64):
        self.BATCH_SIZE = batch_size
        self.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
        self.NUM_EXAMPLES_PER_EPOCH_FOR_DEV = 120

        self.MOVING_AVERAGE_DECAY = 0.999
        self.NUM_EPOCHS_PER_DECAY = 10.0
        self.LEARNING_RATE_DECAY_FACTOR = 0.75
        self.INITIAL_LEARNING_RATE = 1e-4

        self.TF_VERSION = tf.__version__
        print('self.TF_VERSION = ', self.TF_VERSION)

        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8

        self.resnet_input_model = 0  # 0: hand  1:cifar

        if self.resnet_input_model == 0:
            self.resnet_input = ResNetInput_HandSigns(batch_size)
        else:
            self.resnet_input = ResNetInput_Cifar10(batch_size)

        self.Model = self._get_model()

    def print_all_variables(self):
        for var in tf.global_variables():
            print(var)

    def inputs(self, eval_data='dev'):
        return self.resnet_input.inputs(eval_data)

    def destoried_inputs(self):
        return self.resnet_input.distorted_inputs()

    def create_placeholder(self):
        if self.resnet_input_model == 0:
            _shape = [None, 64, 64, 3]
        else :
            _shape = [None, 32, 32, 3]

        X = tf.placeholder(tf.float32, _shape, name='x-input')
        Y = tf.placeholder(tf.int64, [None], name='y-input')
        return X, Y

    def _conv2d(self, x_input, filter_shape, name, strides, padding):
        w = self._weight_variable(shape=filter_shape, name ='w_'+name)
        return tf.nn.conv2d(x_input, w, strides=strides, padding=padding)

    def _weight_variable2(self, shape, name=None):
        initial = tf.truncated_normal(shape, stddev=1e-2)
        return tf.Variable(initial, name=name)

    def _weight_variable(self, shape, name):
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

    def _bias_variable(self, shape, value=0.):
        return tf.Variable(tf.constant(value, tf.float32, shape))

    def _conv_block_v1(self, X_input, in_filter, out_filters, stage_block, training, stride=2):
        block_name = 'res_' + stage_block

        f0 = in_filter
        f1,f2 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            # first
            X = self._conv2d(X_input, [3,3,f0,f1], 'conv_1', strides=[1,stride,stride,1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            # second
            X = self._conv2d(X, [3,3,f1, f2], 'conv_2', strides=[1,1,1,1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            # short cut
            X_shortcut = self._conv2d(X_shortcut, [1, 1, f0, f2], 'shortcut', strides=[1,stride, stride,1], padding='SAME')
            X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, training=training)

            X = tf.add(X_shortcut, X)
            X = tf.nn.relu(X)

        return X


    def _conv_block_v2(self, X_input, in_filter, out_filters, stage_block, training, stride=2):
        """
        axis : [None, w, h, channels]在对应的channel层上做批次归一化, 也可以是-1
        stride 为什么要改变，而且默认为2
        """
        block_name = 'res_' + stage_block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            # first
            X = self._conv2d(X_input, [1, 1, in_filter, f1], 'conv_1', strides=[1, stride, stride, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            # second
            X = self._conv2d(X, [3,3, f1, f2], 'conv_2', strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #third
            X = self._conv2d(X, [1, 1, f2, f3], 'conv_3', strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            # short cut
            X_shortcut = self._conv2d(X_shortcut, [1, 1, in_filter, f3], 'shortcut', strides=[1, stride, stride, 1], padding='VALID')
            X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, training=training)

            # final layer
            X = tf.add(X_shortcut, X)
            X = tf.nn.relu(X)
        return X

    def _identity_block_v1(self, X_input, in_filter, out_filters, stage_block, training, stride=1):
        block_name = 'res_' + stage_block

        f0 = in_filter
        f1, f2 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            # first
            X = self._conv2d(X_input, [3, 3, f0, f1], 'conv_1', strides=[1, stride, stride, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            # second
            X = self._conv2d(X, [3, 3, f1, f2], 'conv_2', strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            # final layer
            X = tf.add(X_shortcut, X)
            X = tf.nn.relu(X)

        return X

    def _identity_block_v2(self, X_input, in_filter, out_filters, stage_block, training):
        block_name = 'res_' + stage_block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            # first
            X = self._conv2d(X_input, [1,1,in_filter,f1], 'id_1', strides=[1,1,1,1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            # second
            X = self._conv2d(X, [3, 3, f1, f2], 'id_2', strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            # third
            X = self._conv2d(X, [1, 1, f2, f3], 'id_3', strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            # final layer
            X = tf.add(X_shortcut, X)
            X = tf.nn.relu(X)
        return X

    def _get_model(self):
        class Model(object):
            pass
        Model.conv1 = {'kernel':[7,7,3,64], 'name':'conv','strides':[1,2,2,1], 'padding':'VALID'}
        Model.conv2_x = {'stage': 2, 'in_filter':64, 'convx':[[64,64,256]]*3, 'strides':1, 'version':2}
        Model.conv3_x = {'stage': 3, 'in_filter':256, 'convx':[[128,128,512]]*3, 'strides':2, 'version':2}
        Model.conv4_x = {'stage': 4, 'in_filter':512, 'convx':[[256,256,1024]]*3, 'strides':2, 'version':2}
        Model.conv5_x = {'stage': 5, 'in_filter':1024, 'convx':[[512,512,2048]]*3, 'strides':2, 'version':2}
        return Model

    def block(self, X_input, model, training):
        stage = model['stage']
        in_filter = model['in_filter']
        convx = model['convx']
        strides = model['strides']
        if model['version'] == 2:
            _conv_block = self._conv_block_v2
            _identity_block = self._identity_block_v2
        else:
            _conv_block = self._conv_block_v1
            _identity_block = self._identity_block_v1
        X = X_input

        with tf.variable_scope('stage_{0}'.format(stage)):
            if in_filter != convx[-1][-1]:
                X = _conv_block(X, in_filter, convx[0], 'a', training, stride=strides)
            else:
                X = _identity_block(X, in_filter, convx[0], 'a', training)

            for i in range(1, len(convx)):
                X = _identity_block(X, convx[i-1][-1], convx[i], chr(0x61+i), training)
        return X


    def inference(self, X_input, training):
        """
        使用self.Model定义了ResNet结构
        """
        # print('input : ', X_input.get_shape())
        X = tf.pad(X_input, tf.constant([[0,0], [3,3], [3,3], [0,0]]), 'CONSTANT')
        # print('pad : ', X.get_shape())
        with tf.variable_scope('inference'):
            with tf.variable_scope('stage_1'):
                X = self._conv2d(X,self.Model.conv1['kernel'],
                                 self.Model.conv1['name'],
                                 self.Model.conv1['strides'],
                                 self.Model.conv1['padding'])
                # print('conv1 : ', X.get_shape())
                X = tf.layers.batch_normalization(X, axis=3, training=training)
                X = tf.nn.relu(X)
                # print('stage1 : ', X.get_shape())
                X = tf.nn.max_pool(X, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            # print('stage1 : ', X.get_shape())

            # stage 2
            X = self.block(X, self.Model.conv2_x, training)
            # print('stage2 : ', X.get_shape())
            # stage 3
            X = self.block(X, self.Model.conv3_x, training)
            # print('stage3 : ', X.get_shape())
            # stage 4
            X = self.block(X, self.Model.conv4_x, training)
            # print('stage4 : ', X.get_shape())
            # stage 5
            X = self.block(X, self.Model.conv5_x, training)

            # print(X.get_shape())

            X = tf.nn.avg_pool(X, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
            if self.TF_VERSION <= '1.6.0':
                flatten = tf.contrib.layers.flatten(X)
            else:
                flatten = tf.layers.flatten(X)
            # print(flatten.get_shape())
            logits = tf.layers.dense(flatten, units=6)
        return tf.reshape(logits, [-1, 6])

    def inference2(self, X_input, training):
        X = tf.pad(X_input, tf.constant([[0,0], [3,3], [3,3], [0,0]]), 'CONSTANT')
        with tf.variable_scope('inference'):
            # training = tf.placeholder(tf.bool, name='training')
            # stage 1
            with tf.variable_scope('stage1'):
                X = self._conv2d(X, [7,7,3,64], 'conv', strides=[1,2,2,1], padding='VALID')
                X = tf.layers.batch_normalization(X, axis=3, training=training)
                X = tf.nn.relu(X)
                X = tf.nn.max_pool(X, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

            with tf.variable_scope('stage2'):
                X = self._conv_block_v2(X, 64, [64,64,256], '2a', training, stride=1)
                X = self._identity_block_v2(X, 256, [64, 64, 256], '2b', training)
                X = self._identity_block_v2(X, 256, [64, 64, 256], '2c', training)

            with tf.variable_scope('stage3'):
                X = self._conv_block_v2(X, 256, [128,128,512], '3a', training)
                X = self._identity_block_v2(X, 512, [128,128,512], '3b', training)
                X = self._identity_block_v2(X, 512, [128, 128, 512], '3c', training)
                X = self._identity_block_v2(X, 512, [128, 128, 512], '3d', training)

            with tf.variable_scope('stage4'):
                X = self._conv_block_v2(X, 512, [256,256,1024], '4a', training)
                X = self._identity_block_v2(X, 1024, [256, 256, 1024], '4b', training)
                X = self._identity_block_v2(X, 1024, [256, 256, 1024], '4c', training)
                X = self._identity_block_v2(X, 1024, [256, 256, 1024], '4d', training)
                X = self._identity_block_v2(X, 1024, [256, 256, 1024], '4e', training)
                X = self._identity_block_v2(X, 1024, [256, 256, 1024], '4f', training)

            with tf.variable_scope('stage5'):
                X = self._conv_block_v2(X, 1024, [512,512,2048], '5a', training)
                X = self._identity_block_v2(X, 2048, [512,512,2048], '5b', training)
                X = self._identity_block_v2(X, 2048, [512, 512, 2048], '5c', training)

            X = tf.nn.avg_pool(X, [1,2,2,1], strides=[1,1,1,1], padding='VALID')

            if self.TF_VERSION <= '1.6.0':
                flatten = tf.contrib.layers.flatten(X)
            else:
                flatten = tf.layers.flatten(X)
            # print(flatten.get_shape())
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
        tf.summary.scalar('accuracy', accuracy_op)
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
    # resnet = ResNetPruning()
    # X, y = resnet.inputs()
    # y = y.reshape(y.shape[0])
    #
    # while True:
    #     _ = np.random.randint(0, len(X))
    #     image, label = X[_], y[_]
    #     print('label = ', label)
    #     plt.imshow(image), plt.show()

    resnet = ResNetPruning()
    resnet.inference()
    pass