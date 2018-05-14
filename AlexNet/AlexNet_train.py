#!/opt/conda/bin/python
#coding=utf-8

import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import time
import AlexNet_pruning as alex
import re
import argparse
import sys

LOG_DIR_TRAIN = './logs/train'
MODEL_PATH = './model/AlexNex_cifar10.ckpt'

tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8

def train(num_epochs = 10000):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        train_images, train_labels = alex.destoried_inputs()
        X, y = alex.create_placeholder()
        logits = alex.inference(X, is_train=True)
        loss = alex.loss(logits=logits, labels=y)
        train_op = alex.train(loss, global_step)

        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        # _loss_exp = []
        with tf.Session(config=tf_config) as sess:
            sess.run(init_op)
            train_writer = tf.summary.FileWriter(LOG_DIR_TRAIN, sess.graph)
            coord = tf.train.Coordinator()
            _ = tf.train.start_queue_runners(sess=sess, coord=coord)

            for step in range(num_epochs):
                start_time = time.time()
                _images, _labels = sess.run([train_images, train_labels])
                _ = sess.run([train_op], feed_dict={X:_images, y:_labels})
                duration = time.time() - start_time

                # print loss value
                if step % 100 == 0:
                    summary, loss_val = sess.run([summary_op, loss], feed_dict={X:_images, y:_labels})
                    train_writer.add_summary(summary, global_step=step)

                    examples_per_sec = alex.BATCH_SIZE / duration
                    sec_per_batch = float(duration)
                    format_str = ('step %d, loss = %.4e (%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (step, loss_val, examples_per_sec, sec_per_batch))

                # save model
                if step % 1000 == 0:
                    saver.save(sess, MODEL_PATH, global_step=global_step)
            coord.request_stop()
            coord.join()
            train_writer.close()


def restore_train(num_epochs = 10000):
    model_dir = re.match(r'(.*)/.*$', MODEL_PATH).group(1)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if not (ckpt and ckpt.model_checkpoint_path):
        print('No checkpoint file found.')
        return
    _step = int(ckpt.model_checkpoint_path.split('-')[-1])

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()# tf.Variable(_step, trainable=False)

        train_images, train_labels = alex.destoried_inputs()
        X, y = alex.create_placeholder()
        logits = alex.inference(X, is_train=True)
        loss = alex.loss(logits=logits, labels=y)
        train_op = alex.train(loss, global_step)

        summary_op = tf.summary.merge_all()
        # init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        # _loss_exp = []

        with tf.Session(config=tf_config) as sess:
            # sess.run(init_op)
            saver.restore(sess, ckpt.model_checkpoint_path)
            train_writer = tf.summary.FileWriter(LOG_DIR_TRAIN, sess.graph)
            coord = tf.train.Coordinator()
            _ = tf.train.start_queue_runners(sess=sess, coord=coord)

            for step in range(_step, num_epochs):
                start_time = time.time()
                _images, _labels = sess.run([train_images, train_labels])
                _ = sess.run([train_op], feed_dict={X: _images, y: _labels})
                duration = time.time() - start_time

                # print loss value
                if step % 100 == 0:
                    summary, loss_val = sess.run([summary_op, loss], feed_dict={X: _images, y: _labels})
                    train_writer.add_summary(summary, global_step=step)

                    examples_per_sec = alex.BATCH_SIZE / duration
                    sec_per_batch = float(duration)
                    format_str = ('step %d, loss = %.3e (%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (step, loss_val, examples_per_sec, sec_per_batch))

                # save model
                if step % 1000 == 0:
                    saver.save(sess, MODEL_PATH, global_step=global_step)

            coord.request_stop()
            coord.join()
            train_writer.close()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_epoches', help='num of epoches', type=int)
    parser.add_argument('-r', '--restore', help='restore to train', action='store_true')
    args = parser.parse_args()
    print('args = ', args)
    if args.restore:
        _train_func = restore_train
    else:
        _train_func = train

    if args.num_epoches != None:
        _train_func(args.num_epoches)

if __name__ == '__main__':
    tf.app.run(main=main, argv=None)