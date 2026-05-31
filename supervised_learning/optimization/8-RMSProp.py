#!/usr/bin/env python3
"""TensorFlow RMSProp optimizer module."""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Create an RMSProp optimization training operation.

    Args:
        loss: loss of the network.
        alpha: learning rate.
        beta2: RMSProp weight.
        epsilon: small number to avoid division by zero.

    Returns:
        The RMSProp optimization operation.
    """
    optimizer = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
