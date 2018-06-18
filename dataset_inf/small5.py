#coding=utf-8

import tensorflow as tf
import os
import numpy as np


class SmallFive(object):
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
        record_bytes = self.image_bytes + self.label_bytes

        # self.file_paths = tf.constant(self._getall_filepath(path))
        self.file_paths, self.labels = self._getall_data(path)
        self.dataset = self._create_dataset()

        iterator = self.dataset.make_one_shot_iterator()
        self.images, self.labels = iterator.get_next()

    def _getall_data(self, path):
        if self.training:
            images_dir = os.path.join(path, 'train')
        else:
            images_dir = os.path.join(path, 'test')
        if not os.path.exists(images_dir):
            raise FileExistsError('path {} does not exist'.format(images_dir))

        images_dir = os.path.abspath(images_dir)

        file_paths = [os.path.join(images_dir, file_name)  for file_name in os.listdir(images_dir)]
        labels = [int(file_name[:-4]) // 100 for file_name in os.listdir(images_dir)]
        return file_paths, labels

    def _dataset_parser(self, image_path, label):
        image_string = tf.read_file(image_path)
        image = tf.image.decode_image(image_string, channels=3)

        label = tf.cast(label, tf.int32)
        image = tf.cast(image, tf.float32)

        image = tf.divide(image, 255.)
        image.set_shape([None, None, 3])
        image = tf.image.resize_images(image, (self.image_height, self.image_width))

        # label.set_shape([1])
        label = tf.reshape(label, [1])
        return image, label


    def _create_dataset(self):
        file_paths = tf.constant(self.file_paths)
        labels = tf.constant(self.labels, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = dataset.prefetch(1000)
        dataset = dataset.map(self._dataset_parser, num_parallel_calls=self.num_parallel_calls)
        if self.training:
            if self.image_augment:
                dataset = dataset.map(self._image_augment)
            dataset = dataset.shuffle(buffer_size=200)

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
        paddings = tf.constant([[20, 20], [20, 20], [0,0]])
        # paddings = tf.expand_dims(paddings, axis=-1)
        # paddings = tf.tile(paddings, [1,1,3])
        image = tf.pad(image, paddings=paddings, mode="SYMMETRIC")
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
        return image, label

    def get_tensor(self):
        return self.images, self.labels


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    small5 = SmallFive(32, path='re', training=True)
    images, labels = small5.get_tensor()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.device('/cpu:0'):
        with tf.Session(config=config) as sess:
            while True:
                x_train, y_train = sess.run([images, labels])
                print(x_train.shape, y_train.shape)

                for i in range(32):
                    print(y_train[i])
                    plt.imshow(x_train[i]),plt.show()
