#!/usr/bin/env python3
"""Calculate the softmax cross-entropy loss of a prediction."""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculate the softmax cross-entropy loss of a prediction.

    Args:
        y: placeholder for the one-hot labels of the input data.
        y_pred: tensor containing the network's predictions.

    Returns:
        A tensor containing the loss of the prediction.
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
