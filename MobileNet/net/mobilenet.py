#coding=utf-8

import keras
from keras.layers import SeparableConv2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import GlobalAvgPool2D
from keras.layers import ZeroPadding2D
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import DepthwiseConv2D
from keras.layers import Dropout
from keras.layers import Reshape

from keras.models import Input, Model
from keras.applications import mobilenet
import keras.backend as K


class MobileNet(object):

    def __init__(self,
                 input_shape=None,
                 alpha=1.0,
                 depth_multipliter=1,
                 dropout=1e-3,
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 pooling=None,
                 classes=1000):
        self.input_shape = input_shape
        self.alpha = alpha
        self.depth_multipliter = depth_multipliter
        self.dropout = dropout
        self.include_top = include_top
        self.weights = weights
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.classes = classes

        self.rows = input_shape[0]
        self.cols = input_shape[1]

        if self.alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

    @staticmethod
    def relu6(x):
        return K.relu(x, max_value=6)

    def _conv_block(self, filters, kernel=(3,3), strides=(1,1)):
        def func(inputs):
            _filters = int(filters * self.alpha)
            x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
            x = Conv2D(_filters, kernel,
                       padding='valid',
                       use_bias=False,
                       strides=strides,
                       name='conv1')(x)
            x = BatchNormalization(axis=-1, name='conv1_bn')(x)
            return Activation(self.relu6, name='conv1_relu')(x)
        return func

    def _depthwise_conv_block(self, filters,
                              strides=(1, 1),
                              block_id=1):
        def func(inputs):
            pointwise_conv_filters = int(filters * self.alpha)
            channel_axis = -1
            x = ZeroPadding2D(padding=(1, 1), name='conv_pad_%d' % block_id)(inputs)
            x = DepthwiseConv2D((3, 3),
                                padding='valid',
                                depth_multiplier=self.depth_multipliter,
                                strides=strides,
                                use_bias=False,
                                name='conv_dw_%d' % block_id)(x)
            x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
            x = Activation(self.relu6, name='conv_dw_%d_relu' % block_id)(x)

            x = Conv2D(pointwise_conv_filters, (1, 1),
                       padding='same',
                       use_bias=False,
                       strides=(1, 1),
                       name='conv_pw_%d' % block_id)(x)
            x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
            return x
        return func

    def build_net_mnist(self):
        inputs = Input(self.input_shape)
        x = self._conv_block(32, strides=(2,2))(inputs)
        x = self._depthwise_conv_block(64, block_id=1)(x)

        # x = self._depthwise_conv_block(128, strides=(2, 2), block_id=2)(x)
        # x = self._depthwise_conv_block(128, block_id=3)(x)

        # x = self._depthwise_conv_block(256, strides=(2, 2), block_id=4)(x)
        # x = self._depthwise_conv_block(256, block_id=5)(x)

        x = self._depthwise_conv_block(512, strides=(2, 2), block_id=6)(x)
        x = self._depthwise_conv_block(512, block_id=7)(x)
        x = self._depthwise_conv_block(512, block_id=8)(x)
        x = self._depthwise_conv_block(512, block_id=9)(x)
        x = self._depthwise_conv_block(512, block_id=10)(x)
        x = self._depthwise_conv_block(512, block_id=11)(x)

        x = self._depthwise_conv_block(1024, strides=(2, 2), block_id=12)(x)
        x = self._depthwise_conv_block(1024, block_id=13)(x)

        shape = (1, 1, int(1024 * self.alpha))

        x = GlobalAvgPool2D()(x)
        x = Reshape(shape, name='reshape_1')(x)
        x = Dropout(self.dropout, name='dropout')(x)
        x = Conv2D(self.classes, (1, 1),
                   padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
        x = Reshape((self.classes,), name='reshape_2')(x)
        model = Model(inputs, x, name='mobilenet_%0.2f_%s' % (self.alpha, self.rows))
        return model

    def build_net(self):
        inputs = Input(self.input_shape)
        x = self._conv_block(32, strides=(2,2))(inputs)
        x = self._depthwise_conv_block(64, block_id=1)(x)

        x = self._depthwise_conv_block(128, strides=(2, 2), block_id=2)(x)
        x = self._depthwise_conv_block(128, block_id=3)(x)

        x = self._depthwise_conv_block(256, strides=(2, 2), block_id=4)(x)
        x = self._depthwise_conv_block(256, block_id=5)(x)

        x = self._depthwise_conv_block(512, strides=(2, 2), block_id=6)(x)
        x = self._depthwise_conv_block(512, block_id=7)(x)
        x = self._depthwise_conv_block(512, block_id=8)(x)
        x = self._depthwise_conv_block(512, block_id=9)(x)
        x = self._depthwise_conv_block(512, block_id=10)(x)
        x = self._depthwise_conv_block(512, block_id=11)(x)

        x = self._depthwise_conv_block(1024, strides=(2, 2), block_id=12)(x)
        x = self._depthwise_conv_block(1024, block_id=13)(x)

        shape = (1, 1, int(1024 * self.alpha))

        x = GlobalAvgPool2D()(x)
        x = Reshape(shape, name='reshape_1')(x)
        x = Dropout(self.dropout, name='dropout')(x)
        x = Conv2D(self.classes, (1, 1),
                   padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
        x = Reshape((self.classes,), name='reshape_2')(x)
        model = Model(inputs, x, name='mobilenet_%0.2f_%s' % (self.alpha, self.rows))
        return model

if __name__ == '__main__':
    model = MobileNet((224,224,3)).build_net()
    # model = mobilenet.MobileNet((224,224,3))
    model.summary()
