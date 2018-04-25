#coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    a = tf.constant(0.5, dtype=tf.float32)
    b = tf.constant(5.5, dtype=tf.float32)
    c = a + b

    with tf.Session() as sess:
        result = sess.run(c)
        print(result)