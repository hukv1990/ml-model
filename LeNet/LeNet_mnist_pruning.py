
# coding: utf-8

# In[8]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os


# In[27]:


NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 55000
BATCH_SIZE = 128

MOVING_AVERAGE_DECAY = 0.999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 5.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.6  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01  # Initial learning rate.


# In[19]:


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size
        )
    return images, tf.reshape(labels, [batch_size])


# In[23]:


def _input_batch(filenames, batch_size=BATCH_SIZE):
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image' : tf.FixedLenFeature([], tf.string),
            'label' : tf.FixedLenFeature([], tf.int64)
        })
    image = tf.cast(tf.decode_raw(features['image'], tf.uint8), tf.float32)
    label = tf.cast(features['label'], tf.int64)
    image = tf.divide(image, 255.0)
    
    image.set_shape([784, ])
#     label.set_shape([])
    min_queue_examples = int(50000 * 0.4)
    return _generate_image_and_label_batch(
        image, 
        label, 
        min_queue_examples, 
        batch_size, 
        shuffle=True)


# In[21]:

def inputs(filenames = ['./mnist_data/mnist_train.tfrecord'], batch_size=BATCH_SIZE):
    images, labels = _input_batch(filenames, batch_size=batch_size)
    return images, labels


def create_placeholder():
    with tf.variable_scope('inputs'):
        X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x-input')
        y = tf.placeholder(tf.int64, shape=[None], name='y-input')
    return X, y

# In[24]:


def _get_layer_variable(shape, stddev=0.01):
    w = tf.get_variable('w', 
                        shape=shape, 
                        dtype=tf.float32,
                        # initializer=tf.truncated_normal_initializer(stddev=stddev))
                        initializer=tf.glorot_normal_initializer(seed=1))
    b = tf.get_variable('b',
                       shape=[shape[-1]],
                       dtype=tf.float32,
                       initializer=tf.constant_initializer(0.1))
    return w, b


# In[25]:


def inference(images, is_train=False):
    A0 = tf.reshape(images, [-1, 28, 28, 1])
    with tf.variable_scope('conv1'):
        w, b = _get_layer_variable([5,5,1,32])
        Z1 = tf.nn.bias_add(tf.nn.conv2d(A0, w, strides=[1,1,1,1], padding='SAME'), b)
        A1 = tf.nn.relu(Z1)
        
    pool1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
    
    with tf.variable_scope('conv2'):
        w, b = _get_layer_variable([5,5,32,64])
        Z2 = tf.nn.bias_add(tf.nn.conv2d(pool1, w, strides=[1,1,1,1], padding='SAME'), b)
        A2 = tf.nn.relu(Z2)
    
    pool2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    A_fc0 = tf.reshape(pool2, [-1, nodes])
    
    with tf.variable_scope('fc1'):
        w, b = _get_layer_variable([nodes, 512])
        Z_fc1 = tf.matmul(A_fc0, w) + b
        A_fc1 = tf.nn.relu(Z_fc1)
    if is_train:
        A_fc1_dropout = tf.nn.dropout(A_fc1, 0.75)
    else:
        A_fc1_dropout = A_fc1

    # with tf.variable_scope('fc2'):
    #     w, b = _get_layer_variable([512, 512])
    #     Z_fc2 = tf.matmul(A_fc1_dropout, w) + b
    #     A_fc2 = tf.nn.relu(Z_fc2)
    # if is_train:
    #     A_fc2_dropout = tf.nn.dropout(A_fc2, 0.75)
    # else:
    #     A_fc2_dropout = A_fc2

    with tf.variable_scope('fc3'):
        w, b = _get_layer_variable([512, 10])
        Z_fc3 = tf.matmul(A_fc1_dropout, w) + b
    return Z_fc3


# In[26]:


def loss(logits, labels):
    cross_entroy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entroy_mean = tf.reduce_mean(cross_entroy)
    tf.add_to_collection('losses', cross_entroy_mean)
    losses = tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('losses', losses)
    return losses

# In[33]:


def train(loss, global_step):
    
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(
        learning_rate=INITIAL_LEARNING_RATE,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    tf.summary.scalar('learning_rate', lr)
    # gradient_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
    # gradient_op = tf.train.AdagradOptimizer(lr).minimize(loss, global_step=global_step)
    gradient_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op


# In[34]:


def _mnist_to_tfrecord(mnist_path):
    mnist = input_data.read_data_sets(mnist_path, dtype=tf.uint8, one_hot=False)
    def save_tfrecord(data, filename):
        images = data.images
        labels = data.labels
        num_examples = data.num_examples
        
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[index]]))
            }))
            writer.write(example.SerializeToString())
        writer.close()
    save_tfrecord(mnist.train, os.path.join(mnist_path, 'mnist_train.tfrecord'))   
    save_tfrecord(mnist.validation, os.path.join(mnist_path, 'mnist_dev.tfrecord'))   
    save_tfrecord(mnist.test, os.path.join(mnist_path, 'mnist_test.tfrecord'))   


# In[18]:


if __name__ == '__main__':
    _mnist_to_tfrecord('../data/mnist')

    # mnist = input_data.read_data_sets('../data/mnist', dtype=tf.uint8, one_hot=False)
    # img = mnist.train.images[0]
    # print(img.max(), img.min())
    # plt.imshow(img.reshape((28,28)), 'gray'), plt.show()

