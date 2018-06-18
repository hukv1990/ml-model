# coding=utf-8

import six
import abc

@six.add_metaclass(abc.ABCMeta)
class BaseDataset(object):
    def __init__(self, batch_size,
                 image_shape,
                 train_path = None,
                 test_path = None):
        self.batch_size = batch_size
        self.image_width = image_shape[0]
        self.image_height = image_shape[1]
        self.image_ndim = image_shape[2]
        if train_path:
            self.train_path = train_path

        if test_path:
            self.test_path = test_path

    @abc.abstractmethod
    def _build_dataset(self): pass

    @abc.abstractmethod
    def _dataset_parser(self): pass

    @abc.abstractmethod
    def get_tensor(self): pass