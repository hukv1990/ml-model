#!/opt/conda/bin/python
#coding=utf-8

import tensorflow as tf
from lstm_pruning import LSTM_Pruning
import numpy as np
import time


class LSTM_Train(LSTM_Pruning):
    def __init__(self, log_dir = './logs/train', model_path = './model/lstm.ckpt'):
        LSTM_Pruning.__init__(self, batch_size=128)
        self.log_dir = log_dir
        self.model_path = model_path

        # model
        self.graph = tf.Graph()
        self.model()

    def model(self):
        with self.graph.as_default():
            self.global_step = tf.train.get_or_create_global_step()
            self.create_placeholder()
            self.logits = self.inference(self.X)
            self.loss_op = self.loss(self.logits, self.Y)
            self.train_op = self.train(self.loss_op, self.global_step)

            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()
            self.train_writer = tf.summary.FileWriter(self.log_dir, graph=self.graph)

    def start_train(self, num_epoches=1000):
        with self.graph.as_default():
            images, labels = self.distoried_inputs()

            with tf.Session(config=self.config) as sess:
                sess.run(tf.global_variables_initializer())
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                for step in range(num_epoches):
                    x_batch, y_batch = sess.run([images, labels])
                    # print(x_batch.shape)

                    start_time = time.time()
                    _ = sess.run(self.train_op, feed_dict={self.X: x_batch, self.Y: y_batch})
                    duration = time.time() - start_time

                    if step % 100 == 0:
                        summary, loss_val = sess.run([self.summary_op, self.loss_op],
                                                     feed_dict={self.X: x_batch, self.Y: y_batch})
                        self.train_writer.add_summary(summary, global_step=step)

                        examples_per_sec = self.batch_size / duration
                        sec_per_batch = float(duration)
                        format_str = ('step %d, loss = %.4e (%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % (step, loss_val, examples_per_sec, sec_per_batch))

                    if step % 1000 == 0:
                        self.saver.save(sess, self.model_path, global_step=self.global_step)

                self.train_writer.close()
                coord.request_stop()
                coord.join()


def main(argv=None):
    lstm_train = LSTM_Train()
    lstm_train.start_train(20001)



if __name__ == '__main__':
    tf.app.run(main, argv=None)
