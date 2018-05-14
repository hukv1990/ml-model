#!/opt/conda/bin/python
#coding=utf-8

import tensorflow as tf
import numpy as np
from ResNet_purning import ResNetPruning
import time
import matplotlib.pyplot as plt
import argparse


class ResNetEval(ResNetPruning):
    def __init__(self, eval_choice='dev',
                 log_dir_dev='./logs/dev',
                 log_dir_train='./logs/train',
                 model_dir = './model'):
        ResNetPruning.__init__(self)
        self.LOG_DIR_DEV = log_dir_dev
        self.LOG_DIR_TRAIN = log_dir_train
        self.MODEL_DIR = model_dir
        self.EVAL_INTERVAL_SECS = 2
        self.eval_choice = eval_choice

    def eval_once(self, ckpt, saver, writer, accuracy_op, summary_op, loss_op, X, Y):
        images, labels = self.inputs(eval_data='dev')
        labels = labels.reshape(labels.shape[0])

        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)

            global_step = ckpt.model_checkpoint_path.split('-')[-1]
            _loss, _accuracy, _summary = sess.run([loss_op, accuracy_op, summary_op],
                                                  feed_dict={X:images, Y:labels})

            writer.add_summary(_summary, global_step)
            print('step = {0}, dev loss = {1:.4e}, dev accuracy = {2:.4f}'.format(global_step, _loss, _accuracy))


    def __call__(self):
        with tf.Graph().as_default():
            X, Y = self.create_placeholder()
            logits = self.inference(X, training=True)
            loss_op = self.loss(logits, Y)
            accuracy_op = self.accuracy(logits, Y)

            variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY)
            variable_to_restore = variable_averages.variables_to_restore()
            # for var in variable_to_restore:
            #     print(var)
            # exit(0)
            saver = tf.train.Saver(variable_to_restore)
            # saver = tf.train.Saver()

            summary_op = tf.summary.merge_all()
            if self.eval_choice == True:
                writer = tf.summary.FileWriter(self.LOG_DIR_DEV)
            else:
                writer = tf.summary.FileWriter(self.LOG_DIR_TRAIN)

            _former_ckpt = None
            while True:
                ckpt = tf.train.get_checkpoint_state(self.MODEL_DIR)
                if not (ckpt and ckpt.model_checkpoint_path):
                    print('No checkpoint file found.')
                    time.sleep(self.EVAL_INTERVAL_SECS)
                    continue

                if _former_ckpt == ckpt.model_checkpoint_path:
                    time.sleep(self.EVAL_INTERVAL_SECS)
                    continue

                _former_ckpt = ckpt.model_checkpoint_path
                self.eval_once(ckpt, saver, writer, accuracy_op, summary_op, loss_op, X, Y)

    def predict(self):
        images, labels = self.inputs(eval_data=True)
        _ = np.random.randint(0, len(images))

        _image, _label = images[_], labels[_]
        print('random num = {0}, label = {1}'.format(_, _label))

        _x = _image.reshape(-1, 64,64,3)
        _y = _label

        X, y = self.create_placeholder()
        logits = self.inference(X, training=False)
        predict = tf.argmax(logits, 1)

        variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
        ckpt = tf.train.get_checkpoint_state(self.MODEL_DIR)
        if not (ckpt and ckpt.model_checkpoint_path):
            print('No checkpoint file found.')
            return

        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)
            _predict = sess.run(predict, feed_dict={X:_x, y:_y})
            print('predict = ', _predict)
        plt.imshow(_image), plt.show()


def main(argv=None):
    eval_choice = True
    parse = argparse.ArgumentParser()
    parse.add_argument('-e', '--eval', choices=['test', 'dev'], help='dev or test eval.')
    parse.add_argument('-p', '--random_predict', action="store_true", help='random predict from dev set.')

    args = parse.parse_args()
    print('args = ', args)

    # predict and return
    if args.random_predict == True:
        resnet_eval = ResNetEval()
        resnet_eval.predict()
        return

    # eval test or dev
    if args.eval:
        eval_choice  = args.eval
    else:
        eval_choice = 'dev'
    ResNetEval(eval_choice=eval_choice)()

if __name__ == '__main__':
    # resnet_eval = ResNetEval()
    # resnet_eval.predict()
    tf.app.run(main=main, argv=None)