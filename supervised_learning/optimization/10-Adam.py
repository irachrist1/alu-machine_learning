#!/usr/bin/env python3
"""TensorFlow Adam optimizer module."""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Create an Adam optimization training operation.

    Args:
        loss: loss of the network.
        alpha: learning rate.
        beta1: weight used for the first moment.
        beta2: weight used for the second moment.
        epsilon: small number to avoid division by zero.

    Returns:
        The Adam optimization operation.
    """
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return optimizer.minimize(loss)
