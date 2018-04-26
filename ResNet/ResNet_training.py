#!/opt/conda/bin/python
#coding=utf-8

import tensorflow as tf
from ResNet_pruning import ResNetPruning
import numpy as np
import time
import argparse

class ResNetTrain(ResNetPruning):
    def __init__(self, log_dir = './logs/train', model_path = './model/ResNet.ckpt'):
        ResNetPruning.__init__(self)
        self.LOG_DIR_TRAIN = log_dir
        self.MODEL_PATH = model_path

    def __call__(self, num_epoches=10000):
        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()
            X, Y = self.create_placeholder()
            logits = self.inference(X, training=True)
            loss = self.loss(logits=logits, labels=Y)
            train_op = self.train(loss, global_step)

            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()

            train_writer = tf.summary.FileWriter(self.LOG_DIR_TRAIN)
            train_writer.add_graph(tf.get_default_graph())

            mini_batches = self.destoried_inputs()
            _len_of_batches = len(mini_batches)

            # print('len of mini batches = ', len(mini_batches))
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for step in range(num_epoches):

                    X_mini_batch, Y_mini_batch = mini_batches[step % _len_of_batches]
                    Y_mini_batch = np.int64(Y_mini_batch.reshape(Y_mini_batch.shape[0]))

                    start_time = time.time()
                    _ = sess.run(train_op, feed_dict={X:X_mini_batch, Y:Y_mini_batch})
                    duration = time.time() - start_time

                    if step % 1 == 0:
                        summary, loss_val = sess.run([summary_op, loss], feed_dict={X: X_mini_batch, Y: Y_mini_batch})
                        train_writer.add_summary(summary, global_step=step)

                        examples_per_sec = self.BATCH_SIZE / duration
                        sec_per_batch = float(duration)
                        format_str = ('step %d, loss = %.4e (%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % (step, loss_val, examples_per_sec, sec_per_batch))

                    if step % 10 == 0:
                        saver.save(sess, self.MODEL_PATH, global_step=global_step)

            train_writer.close()

def main(argv=None):
    resnet = ResNetTrain()
    resnet()

if __name__ == '__main__':
    tf.app.run(main=main, argv=None)
