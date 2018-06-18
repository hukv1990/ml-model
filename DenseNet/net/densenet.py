# -*- coding: utf-8 -*-
"""DenseNet models for Keras.

# Reference paper

- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

# Reference implementation

- [Torch DenseNets]
  (https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import keras
import keras.backend as K
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Concatenate
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.models import Input
from keras.models import Model




def dense_block(x, blocks, name, bottle_neck=True):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1), bottle_neck=bottle_neck)
    return x


def transition_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False,
               name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(inputs, growth_rate, name, bottle_neck = False):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = inputs
    if bottle_neck:
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                name=name + '_0_bn')(x)
        x = Activation('relu', name=name + '_0_relu')(x)
        x = Conv2D(4 * growth_rate, 1, use_bias=False,
                    name=name + '_1_conv')(x)
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)
    x = Conv2D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([inputs, x])
    return x


def DenseNet(blocks,
             input_shape=None,
             classes=1000):
    """Instantiates the DenseNet architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with
    TensorFlow, Theano, and CNTK. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    inputs = Input(input_shape)

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2', bottle_neck=True)
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3', bottle_neck=True)
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4', bottle_neck=True)
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5', bottle_neck=True)

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='fc')(x)

    model = Model(inputs, x, name='densenet')

    return model

def DenseNetBC(blocks,
             input_shape=None,
             classes=1000):
    """Instantiates the DenseNet architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with
    TensorFlow, Theano, and CNTK. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    inputs = Input(input_shape)

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    # x = ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    x = Conv2D(16, 7, strides=2, use_bias=False, padding='same',name='conv1/conv')(inputs)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2', bottle_neck=False)
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3', bottle_neck=False)
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4', bottle_neck=False)
    # x = transition_block(x, 0.5, name='pool4')
    # x = dense_block(x, blocks[3], name='conv5')

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='fc')(x)

    model = Model(inputs, x, name='densenet')

    return model


def DenseNet121(input_shape, classes):
    return DenseNet([6, 12, 24, 16],
                    input_shape,
                    classes)

def DenseNet169(input_shape, classes):
    return DenseNet([6, 12, 32, 32],
                    input_shape,
                    classes)

def DenseNet201(input_shape, classes):
    return DenseNet([6, 12, 48, 32],
                    input_shape,
                    classes)

def DenseNetBC40(input_shape, classes):
    return DenseNetBC([6, 6, 6],
                    input_shape,
                    classes)

