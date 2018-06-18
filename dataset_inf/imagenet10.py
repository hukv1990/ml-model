#coding=utf-8

import tensorflow as tf

from collections import namedtuple


class ImageNet10(object):
    def __init__(self, batch_size, training=True, image_augment=True, path=None):
        self.batch_size = batch_size

        self.image_width = 150
        self.image_height = 150
        self.image_ndim = 3

        self.num_per_epoches_for_train = 400
        self.num_per_epoches_for_test = 100
        self.num_parallel_calls = 16

        self.training = training
        self.image_augment = image_augment

        self.image_bytes = self.image_width * self.image_height * self.image_ndim
        self.label_bytes = 1

        # self.file_paths = tf.constant(self._getall_filepath(path))
        self.file_paths, self.labels = self._getall_data(path)
        self.dataset = self._create_dataset()

        iterator = self.dataset.make_one_shot_iterator()
        self.images, self.labels = iterator.get_next()

    def _create_dataset(self):
        pass




if __name__ == '__main__':
    batch_size = 32
    imagenet = ImageNet10(batch_size)