#!/usr/bin/env python3
"""TensorFlow learning rate decay module."""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Create a learning rate decay operation using inverse time decay.

    Args:
        alpha: original learning rate.
        decay_rate: weight that determines the decay rate.
        global_step: number of gradient descent passes elapsed.
        decay_step: passes before alpha is decayed further.

    Returns:
        The learning rate decay operation.
    """
    return tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
