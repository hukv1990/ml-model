#coding=utf-8

from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.append('../net')
sys.path.append('../dataset_inf')

import tensorflow as tf
import numpy as np
from alexnet import AlexNet

class AlexNetCifar10(object):
    def __init__(self, batch_size = 128,
                 num_classes = 10,
                 model_path='./model/alexnet_cifar10.ckpt',
                 train_log_dir = './logs/train',
                 valid_log_dir='./logs/valid'):
        self.model_path = model_path
        self.train_log_dir = train_log_dir
        self.valid_log_dir = valid_log_dir

        self.batch_size = batch_size
        self.num_classes = num_classes

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

    def train(self, num_epochs=10000):
        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()
            net = AlexNet(self.batch_size, self.num_classes)
            images, labels = net.inputs()
            logits = net.inference(images)
            losses = net.loss(logits, labels)
            train_op = net.train(losses, global_step)
            accuracy = net.accuracy(logits, labels)

            saver = tf.train.Saver()
            writer_train = tf.summary.FileWriter(self.train_log_dir)
            writer_valid = tf.summary.FileWriter(self.valid_log_dir)
            init_op = tf.global_variables_initializer()
            merged = tf.summary.merge_all()

            with tf.Session(config=self.config) as sess:
                sess.run(init_op)
                writer_train.add_graph(sess.graph)

                for step in range(num_epochs):
                    _ = sess.run([train_op], feed_dict={net.is_training: True})

                    if step % 100 == 0:
                        loss_train, acc_train, summary_train \
                            = sess.run([losses, accuracy, merged], feed_dict={net.is_training: True})
                        writer_train.add_summary(summary_train, step)
                        loss_valid, acc_valid, summary_valid \
                            = sess.run([losses, accuracy, merged], feed_dict={net.is_training: False})
                        writer_valid.add_summary(summary_valid, step)

                        format_str = ('step {0}, loss = {1:.3e} , train acc = {2:.2f}, '
                                      'valid loss = {3:.3e}, valid acc = {4:.2f}')
                        print(format_str.format(step, loss_train, acc_train, loss_valid, acc_valid))

                        saver.save(sess, self.model_path, global_step=global_step)
            writer_train.close()
            writer_valid.close()


def main(argv=None):
    net = AlexNetCifar10()
    net.train(10000)

if __name__ == '__main__':
    tf.app.run(main=main)