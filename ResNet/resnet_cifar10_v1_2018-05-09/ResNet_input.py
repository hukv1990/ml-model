#!/opt/conda/bin/python
#coding=utf-8

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import numpy as np
from hand_classfiser.resnets_utils import random_mini_batches


class ResNetInput_HandSigns(object):
    def __init__(self, batch_size, data_dir='./hand_classfiser/hand_classfiser_data'):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.image_size = 64
        self.num_classes = 10
        self.num_examples_per_epoch_for_train = 1000
        self.num_examples_per_epoch_for_dev = 120

    def inputs(self, eval_data):
        _set_name = 'test' if eval_data else 'train'
        _dataset = h5py.File(os.path.join(self.data_dir, '{0}_signs.h5'.format(_set_name)), "r")
        _set_x_orig = np.array(_dataset["{0}_set_x".format(_set_name)][:])  # your train set features
        _set_y_orig = np.array(_dataset["{0}_set_y".format(_set_name)][:])  # your train set labels
        _set_y_orig = _set_y_orig.reshape(_set_y_orig.shape[0], -1)
        return _set_x_orig / 255.0, _set_y_orig

    def distorted_inputs(self):
        train_dataset = h5py.File(os.path.join(self.data_dir, 'train_signs.h5'), "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
        train_set_y_orig = train_set_y_orig.reshape(train_set_y_orig.shape[0], -1)
        return random_mini_batches(train_set_x_orig / 255.0, train_set_y_orig, self.batch_size, 0)


class ResNetInput_Cifar10(object):
    def __init__(self, batch_size, data_dir='./cifar-10-batches-bin'):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.image_size = 32
        self.num_classes = 10
        self.num_examples_per_epoch_for_train = 50000
        self.num_examples_per_epoch_for_dev = 10000

    def read_cifar10(self, filename_queue):

        class CIFAR10Record():
            pass

        result = CIFAR10Record()
        result.height = self.image_size
        result.width = self.image_size
        result.ndim = 3

        image_bytes = result.height * result.width * result.ndim
        label_bytes = 1

        record_bytes = label_bytes + image_bytes

        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        result.key, value = reader.read(filename_queue)

        record_bytes = tf.decode_raw(value, tf.uint8)

        result.label = tf.cast(
            tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32
        )

        depth_major = tf.reshape(
            tf.strided_slice(record_bytes,[label_bytes], [label_bytes + image_bytes]),
            [result.ndim, result.height, result.width]
        )

        # Convert from [depth, height, width] to [height, width, depth].
        result.uint8image = tf.transpose(depth_major, [1,2,0])

        return result

    def destroy_image(self, image):
        """
        destroy images
        """
        return image

    def _generate_image_and_label_batch(self, image, label, min_queue_examples, batch_size, shuffle=False):
        """
        generage batch using tensorflow interface.
        """
        num_preprocess_threads = 16
        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples
            )
        else:
            images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size
            )
        tf.summary.image('images', images)
        return images, tf.reshape(label_batch, [batch_size])

    def distorted_inputs(self):
        file_names = [
            os.path.join(self.data_dir, 'data_batch_{0}.bin'.format(i)) for i in range(1,6)
        ]
        for f in file_names:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file : ', f)

        filename_queue = tf.train.string_input_producer(file_names)

        read_input = self.read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        destroyed_image = self.destroy_image(reshaped_image)

        destroyed_image.set_shape([self.image_size, self.image_size, 3])
        read_input.label.set_shape([1])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(
            self.num_examples_per_epoch_for_train * min_fraction_of_examples_in_queue)
        print('Filling queue with %d CIFAR images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)
        return self._generate_image_and_label_batch(
            destroyed_image,
            read_input.label,
            min_queue_examples,
            self.batch_size,
            shuffle=True
        )


    def inputs(self, eval_data):
        if eval_data != 'dev':
            file_names = [
                os.path.join(self.data_dir, 'data_batch_{0}.bin'.format(i)) for i in range(1,6)
            ]
            num_examples_per_epoch = self.num_examples_per_epoch_for_train
        else:
            file_names = [os.path.join(self.data_dir, 'test_batch.bin')]
            num_examples_per_epoch = self.num_examples_per_epoch_for_dev

        for f in file_names:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file ' + f)

        filename_queue = tf.train.string_input_producer(file_names)
        read_input = self.read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)
        destroyed_image = self.destroy_image(reshaped_image)

        destroyed_image.set_shape([self.image_size, self.image_size, 3])
        read_input.label.set_shape([1])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(
            num_examples_per_epoch * min_fraction_of_examples_in_queue)
        return self._generate_image_and_label_batch(
            destroyed_image,
            read_input.label,
            min_queue_examples,
            self.batch_size,
            shuffle=False
        )


if __name__ == '__main__':
    resnet_input = ResNetInput_Cifar10(10)
    _input = resnet_input.inputs(True)
    image, label = _input.uint8image, _input.label
    with tf.Session() as sess:
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while True:
            _image, _label = sess.run([image, label])
            print('label = ', _label, 'image = ', _image.max())
            plt.imshow(_image.reshape(32,32,3)), plt.show()

        coord.request_stop()
        coord.join()
