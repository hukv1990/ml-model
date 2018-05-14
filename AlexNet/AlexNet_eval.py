#!/opt/conda/bin/python
#coding=utf-8

import tensorflow as tf
import AlexNet_pruning as alex
import time
import math
import numpy as np

EVAL_INTERVAL_SECS = 2
LOG_DIR_DEV = './logs/dev'
LOG_DIR_TRAIN = './logs/train'
MODEL_PATH = './model'



def eval_once(ckpt, saver, writer, top_k_op, summary_op, loss_op, X, y):

    images, labels = alex.inputs()

    with tf.Session() as sess:

        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('-')[-1]

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        num_iter = int(math.ceil(alex.NUM_EXAMPLES_PER_EPOCH_FOR_DEV / alex.BATCH_SIZE))
        true_count = 0
        total_sample_count = num_iter * alex.BATCH_SIZE
        step = 0
        _summary = None
        while step < num_iter and not coord.should_stop():
            _images, _labels = sess.run([images, labels])
            predictions, _summary = sess.run([top_k_op, summary_op],
                                      feed_dict={X:_images, y:_labels})
            true_count += np.sum(predictions)

            step += 1

        precision = true_count / total_sample_count
        summary = tf.Summary()
        summary.ParseFromString(_summary)
        summary.value.add(tag='accuracy', simple_value=precision)
        writer.add_summary(summary, global_step)
        print('step = {0}, dev accuracy = {1:.4f}'.format(global_step, precision))

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def train_once(ckpt, saver, writer, top_k_op, summary_op, X, y):

    images, labels = alex.inputs(eval_data=False)

    with tf.Session() as sess:

        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('-')[-1]

        coord = tf.train.Coordinator()
        try:
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            num_iter = int(math.ceil(alex.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / alex.BATCH_SIZE))
            true_count = 0
            total_sample_count = num_iter * alex.BATCH_SIZE
            step = 0
            while step < num_iter and not coord.should_stop():
                _images, _labels = sess.run([images, labels])
                predictions = sess.run(top_k_op,
                                       feed_dict={X:_images, y:_labels})

                true_count += np.sum(predictions)
                step += 1

            precision = true_count / total_sample_count
            summary = tf.Summary()
            # summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='accuracy', simple_value=precision)
            writer.add_summary(summary, global_step)
            print('step = {0}, train accuracy = {1:.4f}'.format(global_step, precision))
        except:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluation():
    with tf.Graph().as_default():

        X, y = alex.create_placeholder()

        logits = alex.inference(X)
        loss_op = alex.loss(logits, y)
        top_k_op = tf.nn.in_top_k(logits, y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(
            alex.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        summary_op = tf.summary.merge_all()
        dev_writer = tf.summary.FileWriter(LOG_DIR_DEV)
        train_writer = tf.summary.FileWriter(LOG_DIR_TRAIN)

        _former_ckpt = None
        while True:

            ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
            if not (ckpt and ckpt.model_checkpoint_path):
                print('No checkpoint file found.')
                time.sleep(EVAL_INTERVAL_SECS)
                continue

            if _former_ckpt == ckpt.model_checkpoint_path:
                time.sleep(EVAL_INTERVAL_SECS)
                continue

            _former_ckpt = ckpt.model_checkpoint_path

            eval_once(ckpt, saver, dev_writer, top_k_op, summary_op, loss_op, X, y)
            # train_once(ckpt, saver, train_writer, top_k_op, summary_op, X, y)


def main(argv=None):
    evaluation()


if __name__ == '__main__':
    tf.app.run(main=main, argv=None)