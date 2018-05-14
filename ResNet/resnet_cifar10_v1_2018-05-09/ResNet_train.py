#!/opt/conda/bin/python
#coding=utf-8

import tensorflow as tf
from ResNet_purning import ResNetPruning
import numpy as np
import time
import argparse

class ResNetTrain(ResNetPruning):
    def __init__(self, log_dir = './logs/train', model_path = './model/ResNet.ckpt'):
        ResNetPruning.__init__(self)
        self.LOG_DIR_TRAIN = log_dir
        self.MODEL_PATH = model_path


    def restore_train(self, num_epoches=1000):
        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()
            X, Y = self.create_placeholder()
            logits = self.inference(X, training=True)
            loss = self.loss(logits=logits, labels=Y)
            train_op = self.train(loss, global_step)

            summary_op = tf.summary.merge_all()
            variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY)
            variable_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variable_to_restore)

            train_writer = tf.summary.FileWriter(self.LOG_DIR_TRAIN)
            train_writer.add_graph(tf.get_default_graph())

            mini_batches = self.destoried_inputs()
            _len_of_batches = len(mini_batches)

            with tf.Session(config=self.config) as sess:
                ckpt = tf.train.get_checkpoint_state(self.MODEL_DIR)
                if not (ckpt and ckpt.model_checkpoint_path):
                    print('No checkpoint file found.')
                    time.sleep(self.EVAL_INTERVAL_SECS)
                    return

                # sess.run(tf.global_variables_initializer())
                saver.restore(sess, ckpt.model_checkpoint_path)

                for step in range(step, num_epoches):

                    X_mini_batch, Y_mini_batch = mini_batches[step % _len_of_batches]
                    Y_mini_batch = np.int64(Y_mini_batch.reshape(Y_mini_batch.shape[0]))

                    start_time = time.time()
                    _ = sess.run(train_op, feed_dict={X: X_mini_batch, Y: Y_mini_batch})
                    duration = time.time() - start_time

                    if step % 100 == 0:
                        summary, loss_val = sess.run([summary_op, loss], feed_dict={X: X_mini_batch, Y: Y_mini_batch})
                        train_writer.add_summary(summary, global_step=step)

                        examples_per_sec = self.BATCH_SIZE / duration
                        sec_per_batch = float(duration)
                        format_str = ('step %d, loss = %.4e (%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % (step, loss_val, examples_per_sec, sec_per_batch))

                    if step % 100 == 0:
                        saver.save(sess, self.MODEL_PATH, global_step=global_step)

            train_writer.close()

    def __call__(self, num_epoches=1000):
        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()
            X, Y = self.create_placeholder()
            logits = self.inference(X, training=True)
            # exit(0)
            loss = self.loss(logits=logits, labels=Y)
            train_op = self.train(loss, global_step)
            accuracy = self.accuracy(logits, Y)

            summary_op = tf.summary.merge_all()

            saver = tf.train.Saver()

            train_writer = tf.summary.FileWriter(self.LOG_DIR_TRAIN)
            train_writer.add_graph(tf.get_default_graph())
            if self.resnet_input_model == 0:
                mini_batches = self.destoried_inputs()
                _len_of_batches = len(mini_batches)
            else:
                images, labels, = self.destoried_inputs()


            with tf.Session(config=self.config) as sess:
                sess.run(tf.global_variables_initializer())
                if self.resnet_input_model == 1:
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                for step in range(num_epoches):

                    if self.resnet_input_model == 0:
                        X_mini_batch, Y_mini_batch = mini_batches[step % _len_of_batches]
                        Y_mini_batch = np.int64(Y_mini_batch.reshape(Y_mini_batch.shape[0]))
                    else:
                        X_mini_batch, Y_mini_batch = sess.run([images, labels])

                    start_time = time.time()
                    _ = sess.run(train_op, feed_dict={X:X_mini_batch, Y:Y_mini_batch})
                    duration = time.time() - start_time

                    if step % 10 == 0:
                        # summary, loss_val = sess.run([summary_op, loss], feed_dict={X: X_mini_batch, Y: Y_mini_batch})
                        summary, loss_val, acc_val = sess.run([summary_op, loss, accuracy],
                                                              feed_dict={X: X_mini_batch, Y: Y_mini_batch})
                        train_writer.add_summary(summary, global_step=step)

                        examples_per_sec = self.BATCH_SIZE / duration
                        sec_per_batch = float(duration)
                        format_str = ('step %d, loss = %.4e (%.1f examples/sec; %.3f sec/batch), accuracy = %.4f')
                        print(format_str % (step, loss_val, examples_per_sec, sec_per_batch, acc_val))

                        # images, labels = self.inputs(eval_data=True)
                        # labels = labels.reshape(labels.shape[0])
                        #
                        # _step, _loss, _accuracy = sess.run([global_step, loss, accuracy], feed_dict={X: images, Y: labels})
                        # format_str = 'step = {0}, dev loss = {1:.4e}, dev accuracy = {2:.4f}'
                        # print(format_str.format(_step, _loss, _accuracy))

                    if step % 10 == 0:
                        saver.save(sess, self.MODEL_PATH, global_step=global_step)

                if self.resnet_input_model == 1:
                    coord.request_stop()
                    coord.join()

            train_writer.close()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_epoches', type=int, help='num of epoches all.')

    args = parser.parse_args()

    if args.num_epoches:
        ResNetTrain()(num_epoches = args.num_epoches)
    else:
        ResNetTrain()()


if __name__ == '__main__':
    tf.app.run(main=main, argv=None)