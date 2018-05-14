
# coding: utf-8

# In[4]:


#coding=utf-8
import time
import tensorflow as tf
import LeNet_mnist_pruning as mnist
import LeNet_mnist_train
import numpy as np


# In[5]:

NUM_EXAMPLES_PER_EPOCH_FOR_DEV = 5000
EVAL_INTERVAL_SECS = 1
MODEL_PATH = './model'
LOG_DIR = './logs/dev'
# In[7]:


def evaluate():
    with tf.Graph().as_default():
        images, labels = mnist.inputs(['./mnist_data/mnist_dev.tfrecord'], NUM_EXAMPLES_PER_EPOCH_FOR_DEV)
        logits = mnist.inference(images, is_train=False)

        loss = mnist.loss(logits, labels)

        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(top_k_op, tf.float32))
        tf.summary.scalar('acc', accuracy)
        
        variable_averages = tf.train.ExponentialMovingAverage(mnist.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()

        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(variable_to_restore)
        _former_path = None
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(LOG_DIR)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            while True:
                ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    if _former_path != ckpt.model_checkpoint_path:
                        _former_path = ckpt.model_checkpoint_path
                        saver.restore(sess, ckpt.model_checkpoint_path)

                        global_step = ckpt.model_checkpoint_path.split('-')[-1]

                        dev_acc, summary = sess.run([accuracy, summary_op])
                        writer.add_summary(summary, global_step)

                        _str_format = time.strftime("%Y-%m-%d %H:%M:%S")
                        _str_format += ': After {0} training step(s), dev accuracy = {1:.4f}'
                        print(_str_format.format(global_step, dev_acc))

                else :
                    print(time.strftime("%Y-%m-%d %H:%M:%S") + " ： No check point file found")
                time.sleep(EVAL_INTERVAL_SECS)
            coord.request_stop()
            coord.join()

# In[8]:
def predict(image):
    image = np.reshape(image, [-1, 28, 28, 1])
    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x-input')
        logits = mnist.inference(X, is_train=False)
        predict = tf.argmax(logits, 1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        _former_path = None

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                if _former_path != ckpt.model_checkpoint_path:
                    _former_path = ckpt.model_checkpoint_path
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    predictions = sess.run(predict, feed_dict={X:image})

    return predictions

def main(argv=None):
    # '''
    evaluate()
    # '''

    '''
    import os
    import cv2
    _path = r"E:\MachineVision\ml\LeNet\self_test\output"
    for i in range(10):
        file_path = os.path.join(_path, "{0}.bmp".format(i))
        img = cv2.imread(file_path, 0)
        ret = predict(img)
        print("i = {0}, ret = {1}".format(i, ret))
    '''

# In[9]:


if __name__ == '__main__':
    tf.app.run()

