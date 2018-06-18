#coding=utf-8

import tensorflow as tf
import os
import numpy as np


class Cifar10(object):
    def __init__(self, batch_size, training=True, image_augment=True, path=None):
        self.batch_size = batch_size
        self.image_width = 32
        self.image_height = 32
        self.image_ndim = 3

        self.num_per_epoches_for_train = 50000
        self.num_per_epoches_for_test = 10000
        self.num_parallel_calls = 16
        self.num_classes = 10

        self.training = training
        self.image_augment = image_augment

        self.image_bytes = self.image_width * self.image_height * self.image_ndim
        self.label_bytes = 1
        record_bytes = self.image_bytes + self.label_bytes

        # self.file_paths = tf.constant(self._getall_filepath(path))
        self.file_paths = self._getall_filepath(path)
        self.dataset = self._dataset(record_bytes=record_bytes)

        iterator = self.dataset.make_one_shot_iterator()
        self.images, self.labels = iterator.get_next()

    def _dataset(self, record_bytes):
        dataset = tf.data.FixedLengthRecordDataset(self.file_paths, record_bytes=record_bytes)
        dataset = dataset.prefetch(buffer_size=2000)
        dataset = dataset.map(self._dataset_parser, num_parallel_calls=self.num_parallel_calls)

        if self.training:
            if self.image_augment:
                dataset = dataset.map(self._image_augment)
            # self.dataset = self.dataset.prefetch(buffer_size=self.batch_size)
            # dataset = dataset.shuffle(buffer_size=int(0.4 * self.num_per_epoches_for_train))
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)

        return dataset

    def random_crop(self, image, ratio=0.1):
        resize_ratio = tf.add(tf.random_uniform([1], -ratio, 0), 1)
        resize_height = tf.cast(resize_ratio * self.image_height, tf.int32)
        resize_width = tf.cast(resize_ratio * self.image_width, tf.int32)
        resize_ndim = tf.constant([self.image_ndim], dtype=tf.int32)
        random_size = tf.concat([resize_height, resize_width, resize_ndim], axis=0)
        image = tf.random_crop(image, size=random_size)
        return image

    def random_rotate(self, image, ratio=10):
        paddings = tf.constant([[5, 5], [5, 5], [0,0]])
        # paddings = tf.expand_dims(paddings, axis=-1)
        # paddings = tf.tile(paddings, [1,1,3])
        image = tf.pad(image, paddings=paddings, mode="REFLECT")
        pad_size = 5
        # image = tf.image.resize_image_with_crop_or_pad(image,
        #                                        self.image_width + pad_size,
        #                                        self.image_height + pad_size)


        # images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
        #          (NHWC), (num_rows, num_columns, num_channels) (HWC), or
        #          (num_rows, num_columns) (HW). The rank must be statically known (the
        #          shape is not `TensorShape(None)`.
        # angles: A scalar angle to rotate all images by, or (if images has rank 4)
        #          a vector of length num_images, with an angle for each image in the batch.
        # interpolation: Interpolation mode. Supported values: "NEAREST", "BILINEAR".
        #       name: The name of the op.
        angle = tf.random_uniform([1], minval=-ratio, maxval=ratio)
        angle = tf.divide(angle, 180.) * 3.141592
        image = tf.contrib.image.rotate(image, angle)
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       self.image_width,
                                                       self.image_height)
        return image

    def _image_augment(self, image, label):
        # 随机裁剪
        image = self.random_crop(image, ratio=0.1)

        # 随机旋转
        image = self.random_rotate(image, 10)

        # 随机左右翻转
        image = tf.image.random_flip_left_right(image)

        # tf.contrib.image.rotate
        image = tf.image.resize_images(image, [self.image_width, self.image_height])
        return image, label


    def _dataset_parser(self, raw_bytes):
        raw_bytes = tf.decode_raw(raw_bytes, tf.uint8)
        label = tf.slice(raw_bytes, [0], [self.label_bytes], name='decode_label')
        image = tf.slice(raw_bytes, [self.label_bytes], [self.image_bytes], name='decode_image')

        label = tf.cast(label, tf.int32)
        image = tf.cast(image, tf.float32)

        image = tf.reshape(image, [self.image_ndim, self.image_width, self.image_height])
        image = tf.divide(image, 255.)
        image = tf.transpose(image, [1, 2, 0])
        label = tf.one_hot(label, self.num_classes, 1, 0)

        # image.set_shape([self.image_width, self.image_height, self.image_ndim])
        label = tf.reshape(label, [self.num_classes])
        return image, label

    def _getall_filepath(self, path):
        if not os.path.exists(path):
            raise FileExistsError('path {} does not exist')
        if self.training:
            file_paths = [os.path.join(path, 'data_batch_{}.bin'.format(i)) for i in range(1,6)]
        else:
            file_paths = [os.path.join(path, 'test_batch.bin')]
        return file_paths

    def get_tensor(self):
        return self.images, self.labels

if __name__ == '__main__':
    cifar10 = Cifar10(batch_size=128,
                      training=True,
                      image_augment=True,
                      path=r'D:\ml\datasets\cifar-10-batches-bin')

    iterator = cifar10.dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.device('/cpu:0'):
        with tf.Session(config=config) as sess:
            while True:
                x_train, y_train = sess.run([images, labels])
                print(x_train.shape, y_train.shape)

                print(y_train[0])
                import matplotlib.pyplot as plt
                plt.figure(figsize=(5,5)), plt.imshow(x_train[0]),plt.show()


