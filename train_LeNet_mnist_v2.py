# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function

import sys

import tensorflow as tf
import numpy as np
import time

from net.lenet import LeNet
from tensorflow.examples.tutorials.mnist import  input_data

MODEL_PATH = './model/LeNet/lenet_mnist.ckpt'
TRAIN_LOG_DIR = './logs/LeNet/train'
VALID_LOG_DIR = './logs/LeNet/valid'

tf.logging.set_verbosity(tf.logging.ERROR)

dataset = input_data.read_data_sets(r'D:\ml\datasets\mnist', one_hot=True)

def train(num_epochs=10000):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        net = LeNet(32)
        images, labels, training = net.inputs()

        logits = net.inference(images)
        losses = net.loss(logits, labels)
        train_op = net.train(losses, global_step)
        accuracy = net.accuracy(logits, labels)

        saver = tf.train.Saver()
        writer_train = tf.summary.FileWriter(TRAIN_LOG_DIR)
        writer_valid = tf.summary.FileWriter(VALID_LOG_DIR)
        init_op = tf.global_variables_initializer()
        merged = tf.summary.merge_all()

        test_images = dataset.test.images
        test_labels = dataset.test.labels
        test_images = np.reshape(test_images, [-1, 28, 28, 1])
        feed_dict_valid = {images : test_images,
                           labels : test_labels,
                           training : False}

        with tf.Session() as sess:

            sess.run(init_op)

            for step in range(num_epochs):
                x_train, y_train = dataset.train.next_batch(net.batch_size)
                x_train = np.reshape(x_train, [-1, 28, 28, 1])
                feed_dict = {images: x_train, labels: y_train, training: True}

                _ = sess.run([train_op], feed_dict=feed_dict)

                if step % 100 == 0:
                    loss_train, acc_train, summary_train = sess.run([losses, accuracy, merged], feed_dict=feed_dict)
                    writer_train.add_summary(summary_train, step)

                    loss_valid, acc_valid, summary_valid = sess.run([losses, accuracy, merged], feed_dict=feed_dict_valid)
                    writer_valid.add_summary(summary_valid, step)

                    format_str = ('step {0}, loss = {1:.3e} , train acc = {2:.2f}, valid loss = {3:.3e}, valid acc = {4:.2f}')
                    print(format_str.format(step, loss_train, acc_train, loss_valid, acc_valid))

                    saver.save(sess, MODEL_PATH, global_step=global_step)
        writer_train.close()
        writer_valid.close()


def main(argv=None):
    train(10000)

if __name__ == '__main__':
    tf.app.run(main=main)