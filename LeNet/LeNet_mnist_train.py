# coding: utf-8

# In[1]:


import tensorflow as tf
import time
import sys
import argparse
import LeNet_mnist_pruning as mnist
import numpy as np


# In[2]:


BATCH_SIZE = mnist.BATCH_SIZE
MODEL_PATH = './model/lenet_mnist.ckpt'
LOG_DIR = './logs/train'


# In[3]:

def train(num_epochs=10000):
    with tf.Graph().as_default():

        global_step = tf.train.get_or_create_global_step()

        images, labels = mnist.inputs()

        logits = mnist.inference(images, is_train=True)

        loss = mnist.loss(logits, labels)

        train_op = mnist.train(loss, global_step)

        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for step in range(num_epochs):
                start_time = time.time()

                _ = sess.run([train_op])

                duration = time.time() - start_time
                if step % 100 == 0:
                    loss_value, summary = sess.run([loss, merged])
                    writer.add_summary(summary, step)

                    format_str = ('step %d, loss = %.3e (%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (step, loss_value, BATCH_SIZE / duration, float(duration)))

                    saver.save(sess, MODEL_PATH, global_step=global_step)

            coord.request_stop()
            coord.join()
            writer.close()

def restore_train(num_epochs=10000, model_path='./model/lenet_mnist.ckpt-8001'):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        images, labels = mnist.inputs()
        logits = mnist.inference(images)
        loss = mnist.loss(logits, labels)
        train_op = mnist.train(loss, global_step)

        # init_op = tf.global_variables_initializer()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # sess.run(init_op)
            saver.restore(sess, model_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for step in range(num_epochs):
                start_time = time.time()
                _ = sess.run(train_op)
                duration = time.time() - start_time

                if step % 100 == 0:
                    examples_per_sec = BATCH_SIZE / duration
                    sec_per_batch = float(duration)
                    loss_value = sess.run(loss)
                    format_str = ('step %d, loss = %.3e (%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

                if step % 1000 == 0:
                    saver.save(sess, MODEL_PATH, global_step=global_step)
            coord.request_stop()
            coord.join()


def main(argv=None):
    train(20000)


# In[5]:


if __name__ == '__main__':
    tf.app.run(main=main)