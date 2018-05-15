#coding=utf-8

import tensorflow as tf
from ResNet_purning import ResNetPurning
import time
import argparse
import os

class ResNetTrain(ResNetPurning):
    def __init__(self, log_dir = './logs/train', model_dir = './model/resnet_cifar10.ckpt'):
        super(ResNetTrain, self).__init__(training=True)
        self.global_step = tf.train.get_or_create_global_step()
        self.log_dir = log_dir
        self.model_dir = model_dir

        # with self.graph.as_default():
        self.model()

    # def model(self):
    #     self.X, self.Y = self.create_placeholder()
    #     self.logits = self.resnet_v1(self.X, 20)
    #     self.loss = self.calc_loss(logits=self.logits, labels=self.Y)
    #     self.accuracy_op = self.accuracy(self.logits, self.Y)
    #
    #     variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY)
    #     variable_to_restore = variable_averages.variables_to_restore()
    #
    #     self.saver = tf.train.Saver(variable_to_restore)
    #     self.summary_op = tf.summary.merge_all()
    #     self.writer = tf.summary.FileWriter(self.log_dir)

    def model(self):
        self.X, self.Y = self.create_placeholder()
        self.logits = self.resnet_v1(self.X, 20)
        self.loss = self.calc_loss(self.logits, self.Y)
        self.train_op = self.train(self.loss, self.global_step)

        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter(self.log_dir, graph=self.graph)

    def start_train(self, num_epoches = 10000, restore=None):
        # with self.graph.as_default():
            step_start = 0
            images, labels = self.cifar10.distorted_inputs()
            with tf.Session(config=self.config) as sess:
                if restore == True:
                    ckpt = tf.train.get_checkpoint_state(os.path.split(self.model_dir)[0])
                    if ckpt and ckpt.model_checkpoint_path:
                        self.saver.restore(sess, ckpt.model_checkpoint_path)
                        step_start = int(ckpt.model_checkpoint_path.split('-')[-1]) - 1
                    else:
                        print('No checkpoint file found. process exit.')
                        return
                else:
                    sess.run(tf.global_variables_initializer())
                coord = tf.train.Coordinator()
                tf.train.start_queue_runners(sess=sess, coord=coord)

                for step in range(step_start, num_epoches):
                    x_batch, y_batch = sess.run([images, labels])

                    start_time = time.time()
                    _ = sess.run([self.train_op], feed_dict={self.X:x_batch, self.Y:y_batch})
                    duration = time.time() - start_time

                    if step % 10 == 0:
                        summary, loss_val = sess.run([self.summary_op, self.loss],
                                                     feed_dict={self.X: x_batch, self.Y: y_batch})
                        self.train_writer.add_summary(summary, global_step=step)

                        examples_per_sec = self.batch_size / duration
                        sec_per_batch = float(duration)
                        format_str = ('step %d, loss = %.4e (%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % (step, loss_val, examples_per_sec, sec_per_batch))

                    if step % 1000 == 0:
                        self.saver.save(sess, self.model_dir, global_step=self.global_step)

                coord.request_stop()
                coord.join()


def main(argv=None):
    parse = argparse.ArgumentParser()
    parse.add_argument('-n', '--num', type=int, help='max epoches')
    parse.add_argument('-r', '--restore', action="store_true", help='continue to train from restore')

    argv = parse.parse_args()

    if argv.num == None:
        argv.num = 10000

    resnet_train = ResNetTrain()
    resnet_train.start_train(argv.num, restore=argv.restore)

if __name__ == '__main__':
    tf.app.run(main=main, argv=None)