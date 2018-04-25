#coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import AlexNet_pruning as alex

LOG_DIR_TRAIN = './logs/train'
MODEL_PATH = './model/AlexNex_cifar10.ckpt'

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
        with tf.Session() as sess:
            sess.run(init_op)
            train_writer = tf.summary.FileWriter(LOG_DIR_TRAIN, sess.graph)
            coord = tf.train.Coordinator()
            _ = tf.train.start_queue_runners(sess=sess, coord=coord)

            for step in range(num_epochs):
                _train_X, _train_y = sess.run([train_images, train_labels])
                _, summary, loss_val = sess.run([train_op, summary_op, loss], feed_dict={X:_train_X, y:_train_y})
                train_writer.add_summary(summary)

                # print loss value
                if step % 10 == 0:
                    print('step {0}, loss = {1:.4e}().'.format(step, loss_val))

                # save model
                if step % 100 == 0:
                    saver.save(sess, MODEL_PATH, global_step=global_step)
            coord.request_stop()
            coord.join()
            train_writer.close()


def main(argv=None):
    train(1000)

if __name__ == '__main__':
    tf.app.run(main=main, argv=None)