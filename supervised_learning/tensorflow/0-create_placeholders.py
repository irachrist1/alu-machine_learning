#!/usr/bin/env python3
"""Create placeholders for the neural network."""
import tensorflow as tf


def create_placeholders(nx, classes):
    """Return two placeholders, x and y, for the neural network.

    Args:
        nx: the number of feature columns in the data.
        classes: the number of classes in the classifier.

    Returns:
        Two placeholders named ``x`` and ``y``. ``x`` is for the input data
        and ``y`` is for the one-hot labels.
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")
    return x, y
