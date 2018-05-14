#!/opt/conda/bin/python
#coding=utf-8

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np

class LSTM_Input(object):
    def __init__(self, data_dir = './mnist_data', batch_size = 128):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.image_size = 28
        self.num_classes = 10
        self.num_examples_per_epoch_for_train = 55000
        self.num_examples_per_epoch_for_dev = 10000


        self.train_filename = r'mnist_train.tfrecord'
        self.test_filename = r'mnist_test.tfrecord'
        self.dev_filename = r'mnist_dev.tfrecord'

        self.train_path = os.path.join(self.data_dir, self.train_filename)
        self.dev_path = os.path.join(self.data_dir, self.dev_filename)
        self.test_path = os.path.join(self.data_dir, self.test_filename)

    def _generate_image_and_label_batch(self, image, label, min_queue_examples, batch_size, shuffle):
        num_preprocess_threads = 16
        if shuffle:
            images, labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples
            )
        else:
            images, labels = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size
            )
        return images, tf.reshape(labels, [batch_size])


    def inputs(self, eval = 'dev', batch_size = None):
        assert eval in ['dev', 'test', 'train'], "eval shoud be one of ['dev', 'test', 'train']"

        filename_queue = tf.train.string_input_producer([
            os.path.join(self.data_dir, "mnist_{0}.tfrecord".format(eval))
        ])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            })
        image = tf.cast(tf.decode_raw(features['image'], tf.uint8), tf.float32)
        label = tf.cast(features['label'], tf.int64)
        image = tf.divide(image, 255.0)
        image = tf.reshape(image, [self.image_size, self.image_size])

        # image.set_shape([784, ])
        #     label.set_shape([])
        min_queue_examples = int(50000 * 0.4)

        if batch_size is None: batch_size = self.batch_size

        return self._generate_image_and_label_batch(
            image,
            label,
            min_queue_examples,
            batch_size,
            shuffle=False)

    def distorted_inputs(self, batch_size=None):
        filename_queue = tf.train.string_input_producer([self.train_path])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            })
        image = tf.cast(tf.decode_raw(features['image'], tf.uint8), tf.float32)
        label = tf.cast(features['label'], tf.int64)
        image = tf.divide(image, 255.0)
        image = tf.reshape(image, [self.image_size, self.image_size])

        # image.set_shape([28, 28])
        #     label.set_shape([])

        min_queue_examples = int(self.num_examples_per_epoch_for_train * 0.4)

        if batch_size is None: batch_size = self.batch_size

        return self._generate_image_and_label_batch(
            image,
            label,
            min_queue_examples,
            batch_size,
            shuffle=True)

    def _mnist_to_tfrecord(mnist_path):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets(mnist_path, dtype=tf.uint8, one_hot=False)

        def save_tfrecord(data, filename):
            images = data.images
            labels = data.labels
            num_examples = data.num_examples

            writer = tf.python_io.TFRecordWriter(filename)
            for index in range(num_examples):
                image_raw = images[index].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[index]]))
                }))
                writer.write(example.SerializeToString())
            writer.close()

        save_tfrecord(mnist.train, os.path.join(mnist_path, 'mnist_train.tfrecord'))
        save_tfrecord(mnist.validation, os.path.join(mnist_path, 'mnist_dev.tfrecord'))
        save_tfrecord(mnist.test, os.path.join(mnist_path, 'mnist_test.tfrecord'))


def random_sample(images, labels):
    _ = np.random.randint(0, images.shape[0], 1).squeeze()
    image, label = images[_], labels[_]
    image = np.reshape(image, (28,28))
    print('label = ', label)
    plt.imshow(image), plt.show()


def main(argv=None):
    lstm_input = LSTM_Input()
    images, labels = lstm_input.distorted_inputs()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        X, y = sess.run([images, labels])
        print(X.shape, y.shape)

        random_sample(X, y)

        coord.request_stop()
        coord.join()


if __name__ == '__main__':
    tf.app.run(main, argv=None)
