#!/usr/bin/env python3
"""Evaluate the output of a trained neural network."""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluate the output of a neural network.

    Args:
        X: numpy.ndarray with the input data to evaluate.
        Y: numpy.ndarray with the one-hot labels for X.
        save_path: location to load the model from.

    Returns:
        The network's prediction, accuracy, and loss, respectively.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        feed = {x: X, y: Y}
        prediction = sess.run(y_pred, feed_dict=feed)
        acc = sess.run(accuracy, feed_dict=feed)
        cost = sess.run(loss, feed_dict=feed)
    return prediction, acc, cost
