#coding=utf-8

import tensorflow as tf
import AlexNet_pruning as alex
import time

EVAL_INTERVAL_SECS = 5
LOG_DIR_DEV = './logs/dev'
MODEL_PATH = './model'

_former_model = None

def eval_once(dict_op):
    global _former_model
    saver = dict_op['saver']
    loss_op = dict_op['loss']
    accuracy_op = dict_op['accuracy']

    ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
    if not (ckpt and ckpt.model_checkpoint_path):
        print('No checkpoint found')
        return
    if _former_model == ckpt.model_checkpoint_path:
        return
    else:
        _former_model = ckpt.model_checkpoint_path

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        _ = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = _former_model.split('-')[-1]
        accuracy = sess.run(accuracy_op)
        print('step = {0}, acc = {1:.4f}'.format(global_step, accuracy))

        coord.request_stop()
        coord.join()


def evaluation():
    with tf.Graph().as_default():
        dev_images, dev_labels = alex.inputs()
        logits = alex.inference(dev_images)
        loss_op = alex.loss(logits, dev_labels)
        top_k_op = tf.nn.in_top_k(logits, dev_labels, 1)
        accuracy_op = tf.reduce_mean(tf.cast(top_k_op, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(
            alex.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variable_to_restore)
        dict_op = {
            'loss' : loss_op,
            'accuracy' : accuracy_op,
            'saver' : saver
        }
        while True:
            eval_once(dict_op)
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    evaluation()

if __name__ == '__main__':
    tf.app.run(main=main, argv=None)