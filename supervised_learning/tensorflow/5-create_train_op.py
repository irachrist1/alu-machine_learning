#!/usr/bin/env python3
"""Create the training operation for the network."""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Create the gradient descent training operation.

    Args:
        loss: the loss of the network's prediction.
        alpha: the learning rate.

    Returns:
        An operation that trains the network using gradient descent.
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
