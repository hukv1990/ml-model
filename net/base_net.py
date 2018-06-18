# coding=utf-8

import tensorflow as tf
import abc
import six

@six.add_metaclass(abc.ABCMeta)
class BaseNet(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    @abc.abstractmethod
    def inputs(self, *args, **kwargs): pass

    @abc.abstractmethod
    def inference(self, *args, **kwargs): pass

    @abc.abstractmethod
    def train(self, *args, **kwargs): pass

    @abc.abstractmethod
    def loss(self, *args, **kwargs): pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs): pass

    @abc.abstractmethod
    def evaluation(self, *args, **kwargs): pass


class LeCun(BaseNet):
    def __init__(self):
        pass

    def _build_model(self):
        print(__name__)


if __name__ == '__main__':
    net = LeCun()
