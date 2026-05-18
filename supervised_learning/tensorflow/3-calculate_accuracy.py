#!/usr/bin/env python3
"""Calculate the accuracy of a network's prediction."""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculate the decimal accuracy of a prediction.

    Args:
        y: placeholder for the one-hot labels of the input data.
        y_pred: tensor containing the network's predictions.

    Returns:
        A tensor containing the decimal accuracy of the prediction.
    """
    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
