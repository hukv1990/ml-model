#!/opt/conda/bin/python
#coding=utf-8

import tensorflow as tf
from lstm_pruning import LSTM_Pruning
import numpy as np
import time


class LSTM_Eval(LSTM_Pruning):
    def __init__(self, log_dir = './logs/dev', model_dir = './model'):
        LSTM_Pruning.__init__(self)
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.eval_inter_secs = 2

        # model and graph
        self.graph = tf.Graph()
        self.model()

    def model(self):
        self.create_placeholder()
        self.logits = self.inference(self.X)
        self.loss_op = self.loss(self.logits, self.Y)

        variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()

        self.saver = tf.train.Saver(variable_to_restore)

        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir)
        self.accuracy_op = self.accuracy(self.logits, self.Y)

    def start_eval(self):
        _former_ckpt = None
        images, labels = self.inputs('dev', batch_size=self.lstm_input.num_examples_per_epoch_for_dev)
        while True:
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if not (ckpt and ckpt.model_checkpoint_path):
                print('No checkpoint file found.')
                time.sleep(self.eval_inter_secs)
                continue

            if _former_ckpt == ckpt.model_checkpoint_path:
                time.sleep(self.eval_inter_secs)
                continue

            _former_ckpt = ckpt.model_checkpoint_path
            with tf.Session(config=self.config) as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                x_batch, y_batch = sess.run([images, labels])

                global_step = ckpt.model_checkpoint_path.split('-')[-1]
                _accuracy, _loss, _summary = sess.run([self.accuracy_op, self.loss_op, self.summary_op],
                                                      feed_dict={self.X: x_batch, self.Y: y_batch})
                self.writer.add_summary(_summary, global_step)
                print('step = {0}, dev loss = {1:.4e}, dev accuracy = {2:.4f}'.format(global_step, _loss, _accuracy))
                coord.request_stop()
                coord.join()


    def predict_one(self):
        images, labels = self.inputs('test', batch_size=1)
        self.predict(self.logits)


def main(argv=None):
    lstm_model = LSTM_Eval()
    lstm_model.start_eval()


if __name__ == '__main__':
    tf.app.run(main, argv=None)