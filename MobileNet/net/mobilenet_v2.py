#coding=utf-8

from keras.applications import mobilenet
import keras
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import Add
from keras.layers import GlobalAvgPool2D
from keras.layers import Dense

from keras.models import Input, Model
from keras import backend as K

class MobileNetV2(object):
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

        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')


    @staticmethod
    def relu6(x):
        return K.relu(x, max_value=6)

    def _conv_block(self, filters, kernel=(3,3), strides=(1,1)):
        def func(inputs):
            first_block_filters = self._make_divisible(filters * self.alpha, 8)
            x = Conv2D(first_block_filters, kernel,
                       padding='same',
                       use_bias=False,
                       strides=strides,
                       kernel_initializer='he_normal',
                       name='conv1')(inputs)
            x = BatchNormalization(axis=-1, name='conv1_bn')(x)
            return Activation(self.relu6, name='conv1_relu')(x)
        return func

    def _make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _inverted_res_block(self, filters, expansion, strides, block_id):
        def func(inputs):
            in_channels = K.int_shape(inputs)[-1]
            pointwise_conv_filters = int(filters * self.alpha)
            pointwise_filters = self._make_divisible(pointwise_conv_filters, 8)
            x = inputs
            prefix = 'block{}_'.format(block_id)

            if block_id:
                #Expand
                x = Conv2D(expansion * in_channels,
                           kernel_size=(1,1),
                           padding='same',
                           use_bias=False,
                           activation=None,
                           name=prefix+'expand')(x)
                x = BatchNormalization(name=prefix+'expand_BN')(x)
                x = Activation(self.relu6, name=prefix+'expand_relu')(x)
            else:
                prefix = 'expanded_conv_'

            # Depthwise
            x = DepthwiseConv2D(kernel_size=(3,3),
                                strides=strides,
                                padding='same',
                                name=prefix+'depthwise')(x)
            x = BatchNormalization(name=prefix+'depthwise_BN')(x)
            x = Activation(self.relu6, name=prefix+'depthwise_relu')(x)

            #project
            x = Conv2D(pointwise_filters,
                       kernel_size=(1,1),
                       padding='same',
                       use_bias=False,
                       activation=None,
                       name=prefix+'project')(x)
            x = BatchNormalization(name=prefix+'project_BN')(x)

            if in_channels == pointwise_filters and strides == 1:
                return Add(name=prefix+'ass')([inputs, x])
            return x
        return func

    def build_net_mnist(self):

        inputs = Input(self.input_shape)
        x = self._conv_block(32, kernel=(3,2), strides=(2,2))(inputs)
        x = self._inverted_res_block(16, strides=(1,1), expansion=1, block_id=0)(x)

        # x = self._inverted_res_block(24, strides=(2,2), expansion=6, block_id=1)(x)
        # x = self._inverted_res_block(24, strides=(1,1), expansion=6, block_id=2)(x)

        x = self._inverted_res_block(32, strides=(2, 2), expansion=6, block_id=3)(x)
        x = self._inverted_res_block(32, strides=(1, 1), expansion=6, block_id=4)(x)
        x = self._inverted_res_block(32, strides=(1, 1), expansion=6, block_id=5)(x)

        x = self._inverted_res_block(64, strides=(2, 2), expansion=6, block_id=6)(x)
        x = self._inverted_res_block(64, strides=(1, 1), expansion=6, block_id=7)(x)
        x = self._inverted_res_block(64, strides=(1, 1), expansion=6, block_id=8)(x)
        x = self._inverted_res_block(64, strides=(1, 1), expansion=6, block_id=9)(x)

        # x = self._inverted_res_block(96, strides=(2, 2), expansion=6, block_id=10)(x)
        # x = self._inverted_res_block(96, strides=(1, 1), expansion=6, block_id=11)(x)
        # x = self._inverted_res_block(96, strides=(1, 1), expansion=6, block_id=12)(x)
        #
        # x = self._inverted_res_block(160, strides=(2, 2), expansion=6, block_id=13)(x)
        # x = self._inverted_res_block(160, strides=(1, 1), expansion=6, block_id=14)(x)
        # x = self._inverted_res_block(160, strides=(1, 1), expansion=6, block_id=15)(x)

        x = self._inverted_res_block(96, strides=(1, 1), expansion=6, block_id=16)(x)

        x = Conv2D(1280,kernel_size=(1,1),use_bias=False,name='Conv_1')(x)
        x = BatchNormalization(epsilon=1e-3,
                                      momentum=0.999,
                                      name='Conv_1_bn')(x)
        x = Activation(self.relu6, name='out_relu')(x)

        x = GlobalAvgPool2D()(x)
        x = Dense(self.classes, activation='softmax',use_bias=True, name='Logits')(x)

        model = Model(inputs, x, name='mobilenetv2_%0.2f_%s' % (self.alpha, self.rows))
        return model

    def build_net(self):

        inputs = Input(self.input_shape)
        x = self._conv_block(32, kernel=(3,2), strides=(2,2))(inputs)
        x = self._inverted_res_block(16, strides=(1,1), expansion=1, block_id=0)(x)

        x = self._inverted_res_block(24, strides=(2,2), expansion=6, block_id=1)(x)
        x = self._inverted_res_block(24, strides=(1,1), expansion=6, block_id=2)(x)

        x = self._inverted_res_block(32, strides=(2, 2), expansion=6, block_id=3)(x)
        x = self._inverted_res_block(32, strides=(1, 1), expansion=6, block_id=4)(x)
        x = self._inverted_res_block(32, strides=(1, 1), expansion=6, block_id=5)(x)

        x = self._inverted_res_block(64, strides=(2, 2), expansion=6, block_id=6)(x)
        x = self._inverted_res_block(64, strides=(1, 1), expansion=6, block_id=7)(x)
        x = self._inverted_res_block(64, strides=(1, 1), expansion=6, block_id=8)(x)
        x = self._inverted_res_block(64, strides=(1, 1), expansion=6, block_id=9)(x)

        x = self._inverted_res_block(96, strides=(2, 2), expansion=6, block_id=10)(x)
        x = self._inverted_res_block(96, strides=(1, 1), expansion=6, block_id=11)(x)
        x = self._inverted_res_block(96, strides=(1, 1), expansion=6, block_id=12)(x)

        x = self._inverted_res_block(160, strides=(2, 2), expansion=6, block_id=13)(x)
        x = self._inverted_res_block(160, strides=(1, 1), expansion=6, block_id=14)(x)
        x = self._inverted_res_block(160, strides=(1, 1), expansion=6, block_id=15)(x)

        x = self._inverted_res_block(320, strides=(1, 1), expansion=6, block_id=16)(x)

        x = Conv2D(1280,kernel_size=(1,1),use_bias=False,name='Conv_1')(x)
        x = BatchNormalization(epsilon=1e-3,
                                      momentum=0.999,
                                      name='Conv_1_bn')(x)
        x = Activation(self.relu6, name='out_relu')(x)

        x = GlobalAvgPool2D()(x)
        x = Dense(self.classes, activation='softmax',use_bias=True, name='Logits')(x)

        model = Model(inputs, x, name='mobilenetv2_%0.2f_%s' % (self.alpha, self.rows))
        return model

if __name__ == '__main__':
    net = MobileNetV2((224,224,3))
    model = net.build_net()
    model.summary()