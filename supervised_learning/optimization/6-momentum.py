#!/usr/bin/env python3
"""TensorFlow momentum optimizer module."""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Create a momentum optimization training operation.

    Args:
        loss: loss of the network.
        alpha: learning rate.
        beta1: momentum weight.

    Returns:
        The momentum optimization operation.
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
