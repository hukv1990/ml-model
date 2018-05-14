#coding=utf-8

import tensorflow as tf
from ResNet_purning import ResNetPurning
import time
import numpy as np
import math
from tqdm import trange

class ResNetEval(ResNetPurning):
    def __init__(self, batch_size=32,
                 train_log_dir = './logs/train',
                 dev_log_dir = './logs/dev',
                 model_dir = './model'):
        super(ResNetEval, self).__init__(training=False, batch_size=batch_size)

        self.train_log_dir = train_log_dir
        self.dev_log_dir = dev_log_dir
        self.model_dir = model_dir
        self.eval_inter_secs = 2
        self.graph = tf.Graph()

        # model graph
        with self.graph.as_default():
            self.model()

    def model(self):
        self.X, self.Y = self.create_placeholder()
        self.logits = self.build_resnet_18(self.X)
        self.loss = self.calc_loss(logits=self.logits, labels=self.Y)
        self.accuracy_op = self.accuracy(self.logits, self.Y)

        variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()

        self.saver = tf.train.Saver(variable_to_restore)
        self.summary_op = tf.summary.merge_all()
        self.dev_writer = tf.summary.FileWriter(self.dev_log_dir)
        self.train_writer = tf.summary.FileWriter(self.train_log_dir)
        self.dev_images, self.dev_labels = self.cifar10.inputs(eval_data='dev')
        self.train_images, self.train_labels = self.cifar10.inputs(eval_data='test')

    def eval_once(self, sess, global_step, eval_data='dev'):
        if eval_data == 'dev':
            n_times = int(math.floor(self.cifar10.num_examples_per_epoch_for_dev / self.cifar10.batch_size))
            images = self.dev_images
            labels = self.dev_labels
        else:
            n_times = int(math.floor(self.cifar10.num_examples_per_epoch_for_train / self.cifar10.batch_size))
            images = self.train_images
            labels = self.train_labels
        # print('n_times = ', n_times)
        loss_list = []
        acc_list = []
        # n_times = 1

        for i in range(n_times):
            x_batch, y_batch = sess.run([images, labels])
            _accuracy, _loss = sess.run([self.accuracy_op, self.loss],
                                        feed_dict={self.X: x_batch, self.Y: y_batch})
            acc_list.append(_accuracy)
            loss_list.append(_loss)

        _accuracy = np.sum(acc_list) / n_times
        _loss = np.sum(loss_list) / n_times
        summary = tf.Summary()
        summary.value.add(tag='accuracy', simple_value=_accuracy)
        if eval_data == 'dev':
            summary.value.add(tag='losses', simple_value=_loss)
            self.dev_writer.add_summary(summary, global_step)
        else:
            self.train_writer.add_summary(summary, global_step)
        print('step = {0}, {1} loss = {2:.4e}, {1} accuracy = {3:.4f}'.format(global_step, eval_data, _loss, _accuracy))


    def start_eval(self):
        _former_ckpt = None
        with self.graph.as_default():
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
                    global_step = ckpt.model_checkpoint_path.split('-')[-1]

                    self.eval_once(sess, global_step, eval_data='dev')
                    self.eval_once(sess, global_step, eval_data='test')

                    coord.request_stop()
                    coord.join()

def main(argv=None):
    resnet_eval = ResNetEval()
    resnet_eval.start_eval()

if __name__ == '__main__':
    tf.app.run(main=main, argv=None)